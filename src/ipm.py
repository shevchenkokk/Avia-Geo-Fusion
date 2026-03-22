"""
Inverse Perspective Mapping для кадра GoPro.

Мотивация. Матчер (LightGlue/LoFTR) ищет локально-похожие патчи. Между
наклонным кадром самолёта (GoPro под pitch ~-30°) и строго надирным спутником
локальная геометрия разная: одна и та же крыша на кадре самолёта — трапеция, на
спутнике — прямоугольник. RANSAC-гомография это компенсирует на уровне
глобальной модели, но внутри самого матчера каждая кандидатная пара уже
отфильтрована по дескриптору, поэтому большинство «правильных» пар теряется
ещё до RANSAC. Отсюда скачки и низкий inlier-ratio.

Решение. Перед матчингом «выпрямляем» кадр самолёта в псевдо-надир по грубой
модели (pinhole-камера + плоская земля + известный pitch). После warp'а
drone_frame становится геометрически-похож на ортофото — LightGlue/LoFTR
начинают выдавать осмысленные соответствия плотнее и чище.

Точность модели здесь не критична: мы не реконструируем 3D, а лишь приводим
два вида к приблизительно одной проекции. Ошибки pitch/FoV ±10° всё ещё
оставляют матчинг значительно чище сырой перспективы.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class IPMParams:
    """Грубая калибровка GoPro для IPM.

    :param pitch_deg: Угол камеры от горизонта, °. Отрицательный = вниз.
                      Для GoPro на самолёте типично -30..-60°.
    :param fov_h_deg: Горизонтальный FoV, °. Wide/linear GoPro ~= 90..120°.
    :param out_width: Ширина warp'нутой картинки.
    :param out_height: Высота warp'нутой картинки.
    :param ground_span_m: Физический размер области земли, которую
                          хотим видеть в warp'е, метров (сторона квадрата).
                          Реальное значение не важно — это лишь выбор масштаба.
    :param altitude_m: Высота полёта, м. Используется только для подбора
                       scale'а; IPM-гомография инвариантна к общему множителю.
    """

    pitch_deg: float = -30.0
    fov_h_deg: float = 90.0
    out_width: int = 640
    out_height: int = 640
    ground_span_m: float = 400.0
    altitude_m: float = 750.0


class IPMRectifier:
    """Считает гомографию warp'а и применяет её.

    Гомография `H` переводит точки исходного кадра в warped-пространство.
    Удобно держать и обратную `H_inv` для перевода keypoint'ов из warped в
    оригинал (нужно при визуализации оверлеев на исходном кадре).
    """

    def __init__(self, params: IPMParams | None = None):
        self.params = params or IPMParams()
        self._cached_H: np.ndarray | None = None
        self._cached_shape: tuple[int, int] | None = None

    def _build_H(self, src_w: int, src_h: int) -> np.ndarray:
        p = self.params

        fov_h_rad = math.radians(p.fov_h_deg)
        fx = src_w / (2.0 * math.tan(fov_h_rad / 2.0))
        fy = fx
        cx = src_w / 2.0
        cy = src_h / 2.0
        K_cam = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        # Поворот от надира: угол камеры от вертикали = 90° - |pitch|
        # При pitch=-30 (камера смотрит на 30° вниз от горизонта)
        # угол от вертикали = 60°.
        tilt_from_nadir = math.radians(90.0 + p.pitch_deg)
        c = math.cos(tilt_from_nadir)
        s = math.sin(tilt_from_nadir)
        # Поворот вокруг оси X (горизонтальной): кадр смотрит вниз и вперёд.
        R_pitch = np.array(
            [
                [1, 0, 0],
                [0, c, -s],
                [0, s, c],
            ],
            dtype=np.float64,
        )

        # Виртуальная надирная камера: покрывает квадрат со стороной ground_span_m
        # на расстоянии altitude_m. Масштаб подобран так, чтобы эта область
        # поместилась в out_width x out_height.
        f_nadir_x = (p.out_width * p.altitude_m) / p.ground_span_m
        f_nadir_y = (p.out_height * p.altitude_m) / p.ground_span_m
        K_nadir = np.array(
            [
                [f_nadir_x, 0, p.out_width / 2.0],
                [0, f_nadir_y, p.out_height / 2.0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        # Ground-plane гомография: точка плоской земли z=0 в виртуальной
        # надирной камере <-> в физической камере через поворот R_pitch.
        # Вывод: если земля совпадает, то H_src_to_warped = K_nadir * R_pitch^T * K_cam^{-1}
        # (после деления на z).
        H = K_nadir @ R_pitch.T @ np.linalg.inv(K_cam)
        return H

    def homography(self, src_shape: tuple[int, int] | tuple[int, int, int]) -> np.ndarray:
        src_h, src_w = src_shape[:2]
        if self._cached_H is None or self._cached_shape != (src_h, src_w):
            self._cached_H = self._build_H(src_w, src_h)
            self._cached_shape = (src_h, src_w)
        return self._cached_H

    def rectify(self, frame_bgr: np.ndarray) -> np.ndarray:
        H = self.homography(frame_bgr.shape)
        warped = cv2.warpPerspective(
            frame_bgr,
            H,
            (self.params.out_width, self.params.out_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return warped

    def warp_points(self, pts_src: np.ndarray) -> np.ndarray:
        """Переводит (x, y) из исходного кадра в warped-пространство."""
        if len(pts_src) == 0:
            return pts_src.astype(np.float32)
        H = self.homography(self._cached_shape or (0, 0))  # требует предыдущего rectify
        return cv2.perspectiveTransform(pts_src.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)

    def unwarp_points(self, pts_warped: np.ndarray) -> np.ndarray:
        """Переводит (x, y) из warped обратно в оригинальный кадр."""
        if len(pts_warped) == 0:
            return pts_warped.astype(np.float32)
        H = self.homography(self._cached_shape or (0, 0))
        H_inv = np.linalg.inv(H)
        return cv2.perspectiveTransform(pts_warped.reshape(-1, 1, 2).astype(np.float32), H_inv).reshape(-1, 2)
