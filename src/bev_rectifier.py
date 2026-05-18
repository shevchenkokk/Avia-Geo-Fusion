"""BEV-ректификатор (вид сверху) для стека восприятия.

Вычисляет плоскую гомографию, проецирующую наклонный кадр самолёта
(pitch 30–50° ниже горизонта) в синтетический надирный вид — именно
такой вид ожидают матчер и структурный анализатор.

Математика:
    tilt = 90° + pitch_deg
    R_pitch = поворот вокруг оси X на tilt
    R_roll  = поворот вокруг оси Z на roll_deg
    R       = R_pitch @ R_roll
    f_nadir = out_width * agl_m / ground_span_m
    K_nadir = интринсики синтетической надир-камеры с фокусом f_nadir
    H_src_to_bev = K_nadir @ R^T @ K_rect^{-1}
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class BevRectifier:
    K_rect: np.ndarray             # интринсики выпрямленного кадра 3×3
    image_size: tuple[int, int]    # (w, h) входного кадра
    pitch_deg: float               # угол оптической оси ниже горизонта (отрицательный = вниз)
    roll_deg: float = 0.0
    agl_m: float = 620.0           # крейсерская высота AGL; матчер поглощает остаточный масштаб
    ground_span_m: float = 300.0   # сторона охвата BEV в метрах
    out_size: tuple[int, int] = (800, 800)
    # Точка тела, попадающая в центр BEV. None → автоматически точка пересечения
    # оптической оси с землёй. При pitch=-30° AGL=620 → view_y≈1073 м вперёд.
    # При (0,0) в центр попала бы точка прямо под самолётом — вне поля зрения.
    view_centre_body_xy_m: Optional[tuple[float, float]] = None
    H: Optional[np.ndarray] = None  # кэшированная гомография (заполняется .build())

    @classmethod
    def build(
        cls,
        K_rect: np.ndarray,
        image_size: tuple[int, int],
        pitch_deg: float,
        roll_deg: float = 0.0,
        agl_m: float = 620.0,
        ground_span_m: float = 300.0,
        out_size: tuple[int, int] = (800, 800),
        view_centre_body_xy_m: Optional[tuple[float, float]] = None,
    ) -> "BevRectifier":
        obj = cls(
            K_rect=np.asarray(K_rect, dtype=np.float64).reshape(3, 3),
            image_size=image_size,
            pitch_deg=float(pitch_deg),
            roll_deg=float(roll_deg),
            agl_m=float(agl_m),
            ground_span_m=float(ground_span_m),
            out_size=out_size,
            view_centre_body_xy_m=view_centre_body_xy_m,
        )
        obj.H = obj._compute_homography()
        return obj

    def _compute_homography(self) -> np.ndarray:
        out_w, out_h = self.out_size
        f_nadir = (out_w * self.agl_m) / self.ground_span_m

        if self.view_centre_body_xy_m is not None:
            view_x, view_y = self.view_centre_body_xy_m
        else:
            p_abs = abs(self.pitch_deg)
            if p_abs < 89.9:
                view_y = self.agl_m / math.tan(math.radians(p_abs))
            else:
                view_y = 0.0
            view_x = 0.0

        # Смещаем главную точку так, чтобы view_centre проецировался в центр BEV:
        #   pixel = (cx + f * Xb / AGL, cy + f * Yb / AGL) = (out_w/2, out_h/2)
        cx = out_w / 2.0 - f_nadir * view_x / self.agl_m
        cy = out_h / 2.0 - f_nadir * view_y / self.agl_m
        K_nadir = np.array([
            [f_nadir, 0.0, cx],
            [0.0, f_nadir, cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        tilt = math.radians(90.0 + self.pitch_deg)
        c, s = math.cos(tilt), math.sin(tilt)
        R_pitch = np.array([
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ], dtype=np.float64)
        if abs(self.roll_deg) > 1e-9:
            r = math.radians(self.roll_deg)
            cr, sr = math.cos(r), math.sin(r)
            R_roll = np.array([
                [cr, -sr, 0.0],
                [sr, cr, 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            R = R_pitch @ R_roll
        else:
            R = R_pitch
        return K_nadir @ R.T @ np.linalg.inv(self.K_rect)

    def warp(self, frame: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(
            frame, self.H, self.out_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0) if frame.ndim == 3 else 0,
        )

    def warp_mask(self, mask_uint8: np.ndarray) -> np.ndarray:
        """Проецирует маску самолёта в BEV.

        INTER_NEAREST сохраняет бинарность; BORDER_CONSTANT=0 гарантирует, что
        непроецированные зоны BEV не попадут в keypoints матчера.
        """
        return cv2.warpPerspective(
            mask_uint8, self.H, self.out_size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def warp_points(self, pts_rect: np.ndarray) -> np.ndarray:
        if pts_rect is None or len(pts_rect) == 0:
            return np.empty((0, 2), dtype=np.float64)
        src = np.asarray(pts_rect, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(src, self.H)
        return out.reshape(-1, 2)

    def view_centre_body_m(self) -> tuple[float, float]:
        """Точка тела (forward, right) в метрах, проецируемая в центр BEV.

        Соглашение: первый компонент — forward (x_body), второй — right (y_body).
        Для default-конфига (None) центр BEV соответствует пересечению оптической
        оси с землёй: (agl/tan(|pitch|), 0).
        """
        if self.view_centre_body_xy_m is not None:
            view_x, view_y = self.view_centre_body_xy_m
            return (float(view_y), float(view_x))
        p_abs = abs(self.pitch_deg)
        if p_abs < 89.9:
            forward_m = self.agl_m / math.tan(math.radians(p_abs))
        else:
            forward_m = 0.0
        return (float(forward_m), 0.0)
