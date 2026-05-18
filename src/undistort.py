"""Единая точка fisheye-ректификации для стека восприятия.

Линза GoPro Hero4 требует модели fisheye — пинхол с радиальной дисторсией
недостаточен. Этот модуль оборачивает cv2.fisheye так, чтобы все последующие
компоненты (BEV §3.3, матчер §3.4, ретривер §3.5) вызывали одну точку.

Соглашения:
  * Сырой кадр    — с полной fisheye-дисторсией. Маски фюзеляжа живут в этой СК.
  * Ректифицированный — выход ``Undistorter.undistort_image``.
    Все матчинг, ретривер и BEV-варп §3.3+ работают на ректифицированных
    кадрах с ``K_rectified`` как эффективными интринсиками.

LUT-таблицы remap строятся один раз при инициализации; покадровая работа —
один вызов ``cv2.remap`` (~3 мс для 1920×1080 на CPU).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml


@dataclass
class Undistorter:
    """Fisheye-ректификатор с кэшированными LUT-таблицами remap.

    Загрузка: ``Undistorter.from_yaml("configs/camera_gopro_hx.yaml")``.
    """

    K: np.ndarray              # интринсики исходной fisheye-камеры 3×3 float64
    D: np.ndarray              # коэффициенты дисторсии fisheye 4×1 float64
    image_size: tuple[int, int]  # (w, h)
    K_rect: np.ndarray         # ректифицированные интринсики 3×3
    map1: np.ndarray           # LUT remap (16SC2)
    map2: np.ndarray           # LUT remap (16UC1)
    balance: float = 0.0       # 0.0 = обрезать до валидной области, 1.0 = сохранить все пиксели

    @classmethod
    def from_yaml(cls, path: Path | str, balance: float = 0.0) -> "Undistorter":
        cfg = yaml.safe_load(Path(path).read_text())
        if cfg.get("lens_model") != "fisheye":
            raise ValueError(
                f"{path}: expected lens_model=fisheye, got {cfg.get('lens_model')}"
            )
        K = np.array(cfg["K"], dtype=np.float64)
        D = np.array(cfg["D"], dtype=np.float64).reshape(-1, 1)
        w, h = cfg["image_size"]
        return cls.build(K, D, (int(w), int(h)), balance=balance,
                         K_rect_from_yaml=cfg.get("K_rectified"))

    @classmethod
    def build(
        cls,
        K: np.ndarray,
        D: np.ndarray,
        image_size: tuple[int, int],
        balance: float = 0.0,
        K_rect_from_yaml: Optional[list] = None,
    ) -> "Undistorter":
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        D = np.asarray(D, dtype=np.float64).reshape(-1, 1)
        if K_rect_from_yaml is not None:
            K_rect = np.array(K_rect_from_yaml, dtype=np.float64).reshape(3, 3)
        else:
            K_rect = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D, image_size, np.eye(3), balance=balance,
            )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), K_rect, image_size, cv2.CV_16SC2
        )
        return cls(K=K, D=D, image_size=image_size,
                   K_rect=K_rect, map1=map1, map2=map2, balance=balance)

    def undistort_image(self, frame: np.ndarray) -> np.ndarray:
        return cv2.remap(
            frame, self.map1, self.map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    def undistort_points(self, pts: np.ndarray) -> np.ndarray:
        """Сырой пиксель (N,2) → ректифицированный пиксель (N,2).

        Использует аргумент ``P=K_rect`` в cv2.fisheye.undistortPoints,
        чтобы выход был в координатах ректифицированного пикселя —
        той же системе, где живут keypoints матчера после §3.4.
        """
        if pts is None or len(pts) == 0:
            return np.empty((0, 2), dtype=np.float64)
        src = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.fisheye.undistortPoints(src, self.K, self.D, P=self.K_rect)
        return out.reshape(-1, 2)

    def distort_points(self, pts_rect: np.ndarray) -> np.ndarray:
        """Ректифицированный пиксель (N,2) → сырой пиксель (N,2).

        Используется для проецирования аннотаций с ректифицированного кадра
        обратно на исходное видео (HUD-оверлеи, отладочные изображения).
        """
        if pts_rect is None or len(pts_rect) == 0:
            return np.empty((0, 2), dtype=np.float64)
        K_rect_inv = np.linalg.inv(self.K_rect)
        h_pts = np.column_stack([pts_rect, np.ones(len(pts_rect))])
        n_pts = (K_rect_inv @ h_pts.T).T[:, :2]
        out = cv2.fisheye.distortPoints(
            n_pts.reshape(-1, 1, 2).astype(np.float64), self.K, self.D
        )
        return out.reshape(-1, 2)
