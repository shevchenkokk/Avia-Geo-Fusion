"""Этап 2.6: фиксация карты как источник измерения, а не сглаживатель.

В v0 ``Geolocator`` запускал внутренний Kalman/EMA на сырых выходах гомографии
и возвращал одну сглаженную (lat, lon). Это смешивало две ответственности:
получение абсолютной фиксации из матчера и интеграцию её с VO/высотой/курсом.
Теперь сглаживание — задача EKF (этапы 2.4+2.5); матчер должен возвращать
одно наблюдение на кадр со своей неопределённостью.

Модуль предоставляет:
  * ``MapMeasurement`` — датакласс с (lat, lon, σ_xy_m, качество, гомография).
  * ``compute_map_measurement`` — чистая функция без состояния и Калмана.
  * ``apply_to_state_filter`` — передаёт принятое измерение в EKF.

Эвристика σ_xy_m:
  * геометрическая точность: ошибка репроекции, пересчитанная в метры,
  * штраф за низкий inlier_ratio,
  * настраиваемые пол/потолок.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from src.frame_bridge import FrameBridge


# Пороги Stage 1.4, продублированы для автономности модуля.
DEFAULT_MIN_MATCHES = 4
DEFAULT_MIN_INLIERS = 8
DEFAULT_MIN_INLIER_RATIO = 0.15
DEFAULT_MAX_SCALE_CHANGE = 1.5
DEFAULT_RANSAC_REPROJ_PX = 5.0


@dataclass
class MapMeasurement:
    """Одно наблюдение матчер-карта. Заменяет ``Geolocator.estimate_center_gps``."""
    accepted: bool = False
    reject_reason: str = ""

    # Геометрия (валидна только при accepted=True).
    lat: float = float("nan")
    lon: float = float("nan")
    sigma_xy_m: float = float("nan")

    # Метрики качества (заполняются всегда, когда матчер что-то вернул).
    num_matches: int = 0
    num_inliers: int = 0
    inlier_ratio: float = float("nan")
    reprojection_error_px: float = float("nan")
    homography_scale: float = float("nan")
    confidence: float = 0.0  # эвристика [0, 1]

    # Для отладки и оверлеев.
    homography: Optional[np.ndarray] = field(default=None, repr=False)
    inlier_mask: Optional[np.ndarray] = field(default=None, repr=False)


def _homography_scale(H: np.ndarray) -> float:
    a = np.asarray(H, dtype=float)[:2, :2]
    det = float(np.linalg.det(a))
    if det <= 0:
        return float("nan")
    return float(np.sqrt(det))


def _reprojection_error_px(
    pts0: np.ndarray, pts1: np.ndarray, H: np.ndarray, mask: Optional[np.ndarray]
) -> float:
    if mask is not None:
        m = mask.ravel().astype(bool)
        if m.any():
            pts0 = pts0[m]
            pts1 = pts1[m]
    if len(pts0) == 0:
        return float("nan")
    src = np.asarray(pts0, dtype=np.float64).reshape(-1, 1, 2)
    try:
        warped = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    except Exception:
        return float("nan")
    diffs = warped - np.asarray(pts1, dtype=np.float64).reshape(-1, 2)
    return float(np.sqrt((diffs ** 2).sum(axis=1)).mean())


def _meters_per_pixel(
    bbox: tuple[float, float, float, float], map_shape: tuple[int, int]
) -> float:
    """Наихудший (максимальный) м/пиксель для тайла.

    bbox = (west, south, east, north) в градусах; map_shape = (h, w) пикс.
    Возвращает max по осям — σ_xy_m получается консервативным.
    """
    west, south, east, north = bbox
    h, w = map_shape[:2]
    mid_lat = 0.5 * (south + north)
    mppx_lon = (east - west) * 111320.0 * math.cos(math.radians(mid_lat)) / max(w, 1)
    mppx_lat = (north - south) * 111320.0 / max(h, 1)
    return float(max(abs(mppx_lon), abs(mppx_lat)))


def _pixel_to_gps(
    x_px: float, y_px: float,
    bbox: tuple[float, float, float, float],
    map_shape: tuple[int, int],
) -> tuple[float, float]:
    west, south, east, north = bbox
    h, w = map_shape[:2]
    lon = west + (east - west) * (x_px / max(w, 1))
    lat = north - (north - south) * (y_px / max(h, 1))
    return float(lat), float(lon)


def _derive_sigma_xy_m(
    reproj_px: float,
    inliers: int,
    ratio: float,
    mppx: float,
    sigma_floor_m: float,
    sigma_ceiling_m: float,
) -> float:
    """Эвристика 1σ горизонтальной точности из метрик качества.

    base   = reproj_px * mppx           — геометрическая невязка в метрах
    ratio_penalty  = max(1, 0.30/ratio) — штраф за низкий inlier_ratio
    inlier_penalty = max(1, √(8/n))     — больше инлаеров → меньше шум усреднения
    """
    if not np.isfinite(reproj_px) or reproj_px < 0:
        reproj_px = 5.0
    base = reproj_px * mppx
    ratio_penalty = 1.0
    if np.isfinite(ratio) and ratio > 0 and ratio < 0.30:
        ratio_penalty = 0.30 / ratio
    inliers_eff = max(1, inliers)
    inlier_penalty = math.sqrt(max(1.0, 8.0 / inliers_eff))
    sigma = base * ratio_penalty * inlier_penalty
    return float(max(sigma_floor_m, min(sigma, sigma_ceiling_m)))


def _derive_confidence(num_inliers: int, ratio: float, reproj_px: float) -> float:
    if num_inliers <= 0 or not np.isfinite(ratio):
        return 0.0
    inl = 1.0 - math.exp(-num_inliers / 30.0)
    ratio_c = float(np.clip(ratio, 0.0, 1.0))
    reproj_term = 1.0
    if np.isfinite(reproj_px) and reproj_px > 0:
        reproj_term = float(np.clip(1.0 - (reproj_px - 2.0) / 8.0, 0.2, 1.0))
    return float(np.clip(inl * (0.4 + 0.6 * ratio_c) * reproj_term, 0.0, 1.0))


def compute_map_measurement(
    frame_shape: tuple[int, int],
    mkpts_drone: np.ndarray,
    mkpts_map: np.ndarray,
    bbox: tuple[float, float, float, float],
    map_shape: tuple[int, int],
    *,
    last_homography_scale: Optional[float] = None,
    min_matches: int = DEFAULT_MIN_MATCHES,
    min_inliers: int = DEFAULT_MIN_INLIERS,
    min_inlier_ratio: float = DEFAULT_MIN_INLIER_RATIO,
    max_scale_change: float = DEFAULT_MAX_SCALE_CHANGE,
    ransac_reproj_px: float = DEFAULT_RANSAC_REPROJ_PX,
    sigma_floor_m: float = 5.0,
    sigma_ceiling_m: float = 300.0,
) -> MapMeasurement:
    """RANSAC-гомография на выходе матчера + формирование ``MapMeasurement``.

    Чистая функция без состояния и Калмана.
    Ворота качества Stage 1.4 применяются здесь. Скоростной лимит Stage 1.4
    намеренно убран: это теперь задача ворот Махаланобиса EKF.
    """
    out = MapMeasurement(num_matches=int(len(mkpts_drone)))
    if len(mkpts_drone) < min_matches:
        out.reject_reason = "too_few_matches"
        return out

    H, mask = cv2.findHomography(mkpts_drone, mkpts_map, cv2.RANSAC, ransac_reproj_px)
    if H is None:
        out.reject_reason = "homography_failed"
        return out

    num_inliers = int(mask.sum()) if mask is not None else 0
    inlier_ratio = num_inliers / max(1, out.num_matches)
    scale = _homography_scale(H)
    reproj_px = _reprojection_error_px(mkpts_drone, mkpts_map, H, mask)

    out.num_inliers = num_inliers
    out.inlier_ratio = float(inlier_ratio)
    out.homography_scale = scale
    out.reprojection_error_px = reproj_px
    out.homography = H
    out.inlier_mask = mask

    # Ворота Stage 1.4.
    if num_inliers < min_inliers:
        out.reject_reason = "too_few_inliers"
        return out
    if not np.isfinite(inlier_ratio) or inlier_ratio < min_inlier_ratio:
        out.reject_reason = "low_inlier_ratio"
        return out
    if (
        last_homography_scale is not None
        and np.isfinite(scale) and scale > 0
        and last_homography_scale > 0
    ):
        ratio_up = scale / last_homography_scale
        ratio_dn = last_homography_scale / scale
        if max(ratio_up, ratio_dn) > max_scale_change:
            out.reject_reason = "scale_change"
            return out

    fh, fw = frame_shape[:2]
    centre_px = np.array([[[fw / 2.0, fh / 2.0]]], dtype=np.float32)
    try:
        centre_map = cv2.perspectiveTransform(centre_px, H)
    except Exception:
        out.reject_reason = "perspective_transform_failed"
        return out
    map_x, map_y = float(centre_map[0, 0, 0]), float(centre_map[0, 0, 1])
    h, w = map_shape[:2]
    if map_x < -w or map_x > 2 * w or map_y < -h or map_y > 2 * h:
        out.reject_reason = "out_of_map"
        return out
    lat, lon = _pixel_to_gps(map_x, map_y, bbox, map_shape)

    mppx = _meters_per_pixel(bbox, map_shape)
    sigma_xy_m = _derive_sigma_xy_m(
        reproj_px, num_inliers, inlier_ratio, mppx,
        sigma_floor_m, sigma_ceiling_m,
    )
    confidence = _derive_confidence(num_inliers, inlier_ratio, reproj_px)

    out.accepted = True
    out.lat = lat
    out.lon = lon
    out.sigma_xy_m = sigma_xy_m
    out.confidence = confidence
    return out


def apply_to_state_filter(
    meas: MapMeasurement,
    state_filter,                # StateFilter; импорт в рантайме во избежание циклической зависимости
    bridge: FrameBridge,
) -> Optional[object]:
    """Передаёт принятый ``MapMeasurement`` через bridge в EKF.

    Возвращает ``UpdateResult`` фильтра или None, если измерение не прошло ворота матчера.
    """
    if not meas.accepted:
        return None
    x_e, y_n, _ = bridge.wgs84_to_enu(meas.lat, meas.lon, bridge.alt0_msl)
    return state_filter.update_map_position(x_e, y_n, sigma_xy_m=meas.sigma_xy_m)
