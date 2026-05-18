"""Визуальная одометрия на оптическом потоке (Lucas–Kanade + ray-cast).

Метрическая скорость в СК тела (vx, vy, м/с) и угловая скорость (рад/с)
из двух последовательных кадров и высоты AGL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


def camera_to_body_rotation(pitch_deg: float) -> np.ndarray:
    """Матрица 3×3: вектор из СК камеры → СК тела.

    pitch_deg=0 → оптическая ось совпадает с осью x тела (горизонт);
    pitch_deg=-90 → надир.
    """
    p = math.radians(pitch_deg)
    cos_p, sin_p = math.cos(p), math.sin(p)
    # Поворот вокруг оси x камеры; p<0 наклоняет z_c к +y_c (вниз по кадру).
    R_pitch = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_p, -sin_p],
        [0.0, sin_p, cos_p],
    ], dtype=np.float64)
    # Базис уровенной камеры → тело: x_тела = z_камеры, y_тела = x_камеры, z_тела = y_камеры.
    R_level = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    return R_level @ R_pitch


def fisheye_undistort_points(
    pts: np.ndarray, K: np.ndarray, D: np.ndarray
) -> np.ndarray:
    """Пиксель → нормированная плоскость изображения (x_n, y_n) через cv2.fisheye.

    Принимает Nx2 или Nx1x2; возвращает Nx2 float64.
    """
    if len(pts) == 0:
        return np.empty((0, 2), dtype=np.float64)
    src = pts.reshape(-1, 1, 2).astype(np.float64)
    out = cv2.fisheye.undistortPoints(src, K, D)
    return out.reshape(-1, 2)


def ray_cast_ground(
    pts_normalized: np.ndarray,
    R_c2b: np.ndarray,
    agl_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Нормированные координаты камеры → точки на земле в СК тела.

    Каждая точка трактуется как луч (x_n, y_n, 1) в СК камеры, поворачивается
    в СК тела и пересекается с плоскостью z_тело = agl_m.

    Возвращает:
      ground_pts_body : (M, 3) XYZ в метрах, СК тела, для M попаданий в землю.
      mask            : (N,) bool — какие лучи попали (r_b[2] > 0).
    """
    n = len(pts_normalized)
    if n == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=bool)

    rays_c = np.empty((n, 3), dtype=np.float64)
    rays_c[:, 0] = pts_normalized[:, 0]
    rays_c[:, 1] = pts_normalized[:, 1]
    rays_c[:, 2] = 1.0
    rays_b = (R_c2b @ rays_c.T).T
    # Лучи с r_b[2] ≤ 0 направлены выше горизонта — маскируются.
    mask = rays_b[:, 2] > 1e-6
    valid = rays_b[mask]
    t = agl_m / valid[:, 2]
    pts_b = valid * t[:, None]
    return pts_b, mask


def fit_rigid_transform_2d(
    src: np.ndarray, dst: np.ndarray
) -> tuple[float, np.ndarray, float]:
    """МНК 2D-жёсткого преобразования: dst ≈ R(theta) @ src + t.

    Возвращает (theta рад, t вектор 2D, rms м).
    """
    assert src.shape == dst.shape and src.shape[1] == 2 and len(src) >= 2
    src_mu = src.mean(axis=0)
    dst_mu = dst.mean(axis=0)
    s = src - src_mu
    d = dst - dst_mu
    H = s.T @ d
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    theta = math.atan2(R[1, 0], R[0, 0])
    t = dst_mu - R @ src_mu
    pred = (R @ src.T).T + t
    rms = float(np.sqrt(((pred - dst) ** 2).sum(axis=1).mean()))
    return theta, t, rms


def fit_rigid_transform_2d_ransac(
    src: np.ndarray,
    dst: np.ndarray,
    threshold_m: float = 3.0,
    iters: int = 80,
    rng_seed: int = 0,
) -> tuple[float, np.ndarray, float, np.ndarray]:
    """RANSAC-обёртка над ``fit_rigid_transform_2d``.

    ``threshold_m`` — порог евклидовой невязки (м) для определения инлаеров.
    При менее 2 инлаеров возвращает подгонку по всем точкам как запасной вариант.
    """
    n = len(src)
    if n < 2:
        return 0.0, np.zeros(2), float("inf"), np.zeros(n, dtype=bool)
    rng = np.random.default_rng(rng_seed)
    best_inliers = np.zeros(n, dtype=bool)
    best_count = 0
    for _ in range(iters):
        idx = rng.choice(n, size=2, replace=False)
        try:
            theta_h, t_h, _ = fit_rigid_transform_2d(src[idx], dst[idx])
        except Exception:
            continue
        R = np.array([[math.cos(theta_h), -math.sin(theta_h)],
                      [math.sin(theta_h), math.cos(theta_h)]])
        pred = (R @ src.T).T + t_h
        resid = np.linalg.norm(pred - dst, axis=1)
        inliers = resid < threshold_m
        c = int(inliers.sum())
        if c > best_count:
            best_count = c
            best_inliers = inliers
    if best_count < 2:
        theta, t, rms = fit_rigid_transform_2d(src, dst)
        return theta, t, rms, np.ones(n, dtype=bool)
    theta, t, rms = fit_rigid_transform_2d(src[best_inliers], dst[best_inliers])
    return theta, t, rms, best_inliers


@dataclass
class VoStep:
    velocity_body: np.ndarray = field(default_factory=lambda: np.zeros(3))  # м/с
    speed: float = 0.0
    yaw_rate: float = 0.0               # рад/с, >0 = нос влево
    num_features: int = 0
    num_tracked: int = 0
    num_grounded: int = 0
    num_inliers: int = 0
    rms_m: float = float("nan")
    valid: bool = False
    reject_reason: str = ""             # "" | "too_few_features" | ...


class OpticalFlowVO:
    """OF VO с учётом fisheye-дисторсии и ray-cast на землю.

    Состояние хранит предыдущий кадр и треки — вызывающий код подаёт только
    текущий кадр.
    """

    def __init__(
        self,
        K: np.ndarray,
        D: np.ndarray,
        image_size: tuple[int, int],
        pitch_deg: float = -30.0,
        max_features: int = 300,
        feature_quality: float = 0.01,
        feature_min_distance: int = 25,
        lk_window: int = 21,
        lk_max_level: int = 3,
        fwd_bwd_threshold_px: float = 1.0,
        ransac_threshold_m: float = 3.0,
        min_features_per_frame: int = 30,
        min_inliers: int = 6,
        downsample_to_width: int = 0,
    ) -> None:
        K_in = np.asarray(K, dtype=np.float64).reshape(3, 3)
        D_in = np.asarray(D, dtype=np.float64).reshape(-1, 1)
        # Опциональный downsample: на бортовом железе LK на 1920×1080 — overkill,
        # 960×540 даёт ~3× speedup при ~той же точности. Intrinsics
        # масштабируются (fx, fy, cx, cy * s; D-коэффициенты sccale-invariant).
        if 0 < downsample_to_width < image_size[0]:
            s = downsample_to_width / float(image_size[0])
            new_w = downsample_to_width
            new_h = int(round(image_size[1] * s))
            scale_mat = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1.0]], dtype=np.float64)
            self.K = scale_mat @ K_in
            self.D = D_in
            self.image_size = (new_w, new_h)
            self._downsample_scale = s
        else:
            self.K = K_in
            self.D = D_in
            self.image_size = image_size
            self._downsample_scale = 1.0
        self.pitch_deg = float(pitch_deg)
        self.R_c2b = camera_to_body_rotation(self.pitch_deg)

        self.max_features = max_features
        self.feature_quality = feature_quality
        self.feature_min_distance = feature_min_distance
        self.lk_params = dict(
            winSize=(lk_window, lk_window),
            maxLevel=lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.fwd_bwd_threshold_px = fwd_bwd_threshold_px
        self.ransac_threshold_m = ransac_threshold_m
        self.min_features_per_frame = min_features_per_frame
        self.min_inliers = min_inliers

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None
        self.last_debug: dict = {}

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_pts = None
        self.last_debug = {}

    def step(
        self,
        frame: np.ndarray,
        dt: float,
        agl_m: float,
        aircraft_mask: Optional[np.ndarray] = None,
    ) -> VoStep:
        if self._downsample_scale != 1.0:
            target_size = self.image_size  # (W, H)
            if (frame.shape[1], frame.shape[0]) != target_size:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
            if aircraft_mask is not None and (
                aircraft_mask.shape[1], aircraft_mask.shape[0]
            ) != target_size:
                aircraft_mask = cv2.resize(
                    aircraft_mask, target_size, interpolation=cv2.INTER_NEAREST,
                )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        # Маска детектора: 255 = можно трекать. Маска фюзеляжа из этапа 1.2
        # кодирует 255-на-фюзеляже, поэтому здесь инвертируется.
        det_mask = np.full(gray.shape, 255, dtype=np.uint8)
        if aircraft_mask is not None:
            det_mask[aircraft_mask > 0] = 0

        if self._prev_gray is None or self._prev_pts is None or len(self._prev_pts) < self.min_features_per_frame:
            new_pts = self._detect_features(gray, det_mask)
            self._prev_gray = gray
            self._prev_pts = new_pts
            return VoStep(
                num_features=len(new_pts) if new_pts is not None else 0,
                reject_reason="bootstrap",
            )

        curr_pts, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_pts, None, **self.lk_params
        )
        if curr_pts is None:
            self._prev_gray = gray
            self._prev_pts = self._detect_features(gray, det_mask)
            return VoStep(reject_reason="lk_failed")

        back_pts, st_bwd, _ = cv2.calcOpticalFlowPyrLK(
            gray, self._prev_gray, curr_pts, None, **self.lk_params
        )
        ok = (st_fwd.flatten() == 1) & (st_bwd.flatten() == 1)
        fb_err = np.linalg.norm(
            back_pts.reshape(-1, 2) - self._prev_pts.reshape(-1, 2), axis=1
        )
        ok &= fb_err < self.fwd_bwd_threshold_px

        if aircraft_mask is not None and ok.any():
            curr_int = curr_pts.reshape(-1, 2).astype(np.int32)
            cy = np.clip(curr_int[:, 1], 0, aircraft_mask.shape[0] - 1)
            cx = np.clip(curr_int[:, 0], 0, aircraft_mask.shape[1] - 1)
            inside_curr = aircraft_mask[cy, cx] > 0
            ok &= ~inside_curr

        prev_kept = self._prev_pts[ok].reshape(-1, 2)
        curr_kept = curr_pts[ok].reshape(-1, 2)
        n_tracked = int(ok.sum())

        prev_n = fisheye_undistort_points(prev_kept, self.K, self.D)
        curr_n = fisheye_undistort_points(curr_kept, self.K, self.D)
        prev_b, prev_mask = ray_cast_ground(prev_n, self.R_c2b, agl_m)
        curr_b, curr_mask = ray_cast_ground(curr_n, self.R_c2b, agl_m)
        both = prev_mask & curr_mask
        prev_xy_all = np.full((n_tracked, 2), np.nan)
        curr_xy_all = np.full((n_tracked, 2), np.nan)
        prev_xy_all[prev_mask] = prev_b[:, :2]
        curr_xy_all[curr_mask] = curr_b[:, :2]
        valid_track = both
        prev_xy = prev_xy_all[valid_track]
        curr_xy = curr_xy_all[valid_track]
        n_grounded = int(valid_track.sum())

        if n_grounded < self.min_inliers:
            self._refresh_for_next_step(gray, curr_pts, ok, det_mask)
            return VoStep(
                num_features=len(self._prev_pts),
                num_tracked=n_tracked, num_grounded=n_grounded,
                reject_reason="too_few_grounded",
            )

        # Жёсткое 2D-преобразование: точка земли t_prev → t_curr.
        # Движение самолёта — обратное преобразование: перемещение = -t, курс = -theta.
        theta, t, rms, inlier_mask = fit_rigid_transform_2d_ransac(
            prev_xy, curr_xy, threshold_m=self.ransac_threshold_m
        )
        n_inliers = int(inlier_mask.sum())
        if n_inliers < self.min_inliers:
            self._refresh_for_next_step(gray, curr_pts, ok, det_mask)
            return VoStep(
                num_features=len(self._prev_pts),
                num_tracked=n_tracked, num_grounded=n_grounded,
                num_inliers=n_inliers,
                rms_m=rms,
                reject_reason="too_few_inliers",
            )

        d_xy = -t
        d_yaw = -theta
        velocity = np.array([d_xy[0] / dt, d_xy[1] / dt, 0.0])
        yaw_rate = d_yaw / dt
        speed = float(np.linalg.norm(velocity))

        self._refresh_for_next_step(gray, curr_pts, ok, det_mask)

        self.last_debug = {
            "prev_kept": prev_kept,
            "curr_kept": curr_kept,
            "prev_xy_body": prev_xy,
            "curr_xy_body": curr_xy,
            "inlier_mask": inlier_mask,
        }

        return VoStep(
            velocity_body=velocity,
            speed=speed,
            yaw_rate=yaw_rate,
            num_features=len(self._prev_pts),
            num_tracked=n_tracked,
            num_grounded=n_grounded,
            num_inliers=n_inliers,
            rms_m=rms,
            valid=True,
            reject_reason="",
        )

    def _detect_features(
        self, gray: np.ndarray, mask: np.ndarray
    ) -> Optional[np.ndarray]:
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_features,
            qualityLevel=self.feature_quality,
            minDistance=self.feature_min_distance,
            mask=mask,
            blockSize=5,
        )
        return pts

    def _refresh_for_next_step(
        self,
        gray: np.ndarray,
        curr_pts: np.ndarray,
        ok: np.ndarray,
        det_mask: np.ndarray,
    ) -> None:
        kept = curr_pts[ok]
        if len(kept) >= self.min_features_per_frame:
            self._prev_gray = gray
            self._prev_pts = kept
            return
        # Добираем новые точки в незаполненных зонах.
        topup_mask = det_mask.copy()
        for x, y in kept.reshape(-1, 2).astype(np.int32):
            if 0 <= x < topup_mask.shape[1] and 0 <= y < topup_mask.shape[0]:
                cv2.circle(topup_mask, (int(x), int(y)),
                           radius=self.feature_min_distance,
                           color=0, thickness=-1)
        new_pts = self._detect_features(gray, topup_mask)
        if new_pts is None:
            self._prev_pts = kept
        else:
            self._prev_pts = np.vstack([kept, new_pts])
        self._prev_gray = gray
