"""Семантическое сопоставление маски BEV-кадра и спутниковой маски.

Идея:
- Единый SegFormer обрабатывает BEV-кадр и спутниковый тайл, формируя две
  карты классов в одной таксономии: вода, растительность, здания, дороги.
- Для каждого устойчивого класса считаем NCC между бинарной маской кадра
  и спутниковой маской в локальном окне поиска.
- Суммируем NCC по классам с весами → находим пик → центр кадра проецируется
  в широту/долготу, а неопределённость выводится из остроты пика.

В отличие от StructuralMatcher здесь нет зависимости от региональных
векторных слоёв Overture: обе стороны являются выходом одного SegFormer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


# Веса классов в сумме NCC. Класс 0, то есть фон, исключён.
DEFAULT_CLASS_WEIGHTS = {
    1: 1.5,   # вода: узкий, но информативный класс
    2: 0.5,   # растительность: слишком общий класс, поэтому вес ниже
    3: 1.5,   # здания: хороший структурный сигнал
    4: 1.5,   # дороги: линейные и достаточно точные структуры
}


@dataclass
class SemanticMaskFix:
    accepted: bool = False
    reject_reason: str = ""
    lat: float = float("nan")
    lon: float = float("nan")
    sigma_xy_m: float = float("nan")
    peak_score: float = 0.0
    peak_margin: float = 0.0
    n_drone_pixels: int = 0
    n_sat_pixels: int = 0
    used_classes: int = 0


def _ncc_sliding(template: np.ndarray, scene: np.ndarray) -> np.ndarray:
    """Вычислить нормированную корреляцию OpenCV для скользящего окна."""
    t = template.astype(np.float32)
    s = scene.astype(np.float32)
    if t.std() < 1e-6:
        return np.zeros((s.shape[0] - t.shape[0] + 1,
                          s.shape[1] - t.shape[1] + 1), dtype=np.float32)
    return cv2.matchTemplate(s, t, cv2.TM_CCOEFF_NORMED)


def _peak_with_margin(ncc: np.ndarray) -> tuple[int, int, float, float]:
    """Найти основной пик и запас до второго пика вне ближайшей окрестности."""
    flat = ncc.flatten()
    peak_idx = int(np.argmax(flat))
    peak = float(flat[peak_idx])
    py, px = peak_idx // ncc.shape[1], peak_idx % ncc.shape[1]

    radius = max(5, min(ncc.shape) // 20)
    masked = ncc.copy()
    y0, y1 = max(0, py - radius), min(ncc.shape[0], py + radius + 1)
    x0, x1 = max(0, px - radius), min(ncc.shape[1], px + radius + 1)
    masked[y0:y1, x0:x1] = -1.0
    second = float(masked.max())
    margin = peak - second
    return py, px, peak, margin


def mask_to_mask_match(
    drone_class_map: np.ndarray,
    sat_class_map: np.ndarray,
    drone_mppx: float,
    sat_mppx: float,
    sat_centre_latlon: tuple[float, float],
    class_weights: Optional[dict[int, float]] = None,
    min_drone_pixels_per_class: int = 50,
    min_total_drone_pixels: int = 500,
    accept_peak_threshold: float = 0.10,
    accept_margin_threshold: float = 0.02,
    sigma_floor_m: float = 15.0,
    sigma_ceiling_m: float = 250.0,
) -> SemanticMaskFix:
    """Сопоставить карту классов BEV-кадра со спутниковой картой классов."""
    if class_weights is None:
        class_weights = DEFAULT_CLASS_WEIGHTS

    # Приводим карту классов кадра к масштабу спутникового тайла.
    scale = drone_mppx / sat_mppx
    if not math.isfinite(scale) or scale <= 0:
        return SemanticMaskFix(reject_reason="bad_mppx_ratio")
    new_w = max(1, int(round(drone_class_map.shape[1] * scale)))
    new_h = max(1, int(round(drone_class_map.shape[0] * scale)))
    drone_resized = cv2.resize(drone_class_map, (new_w, new_h),
                                 interpolation=cv2.INTER_NEAREST)

    if (drone_resized.shape[0] >= sat_class_map.shape[0]
            or drone_resized.shape[1] >= sat_class_map.shape[1]):
        return SemanticMaskFix(
            reject_reason=f"drone_template_too_big_{drone_resized.shape}_in_{sat_class_map.shape}"
        )

    n_drone_total = int((drone_resized > 0).sum())
    n_sat_total = int((sat_class_map > 0).sum())
    if n_drone_total < min_total_drone_pixels:
        return SemanticMaskFix(reject_reason="too_few_drone_pixels",
                                n_drone_pixels=n_drone_total,
                                n_sat_pixels=n_sat_total)

    # Суммируем NCC по отдельным классам.
    score_sum: Optional[np.ndarray] = None
    weight_sum = 0.0
    used_classes = 0
    for cid, w in class_weights.items():
        if w <= 0:
            continue
        t = (drone_resized == cid).astype(np.uint8) * 255
        if int((t > 0).sum()) < min_drone_pixels_per_class:
            continue
        s = (sat_class_map == cid).astype(np.uint8) * 255
        if int((s > 0).sum()) < min_drone_pixels_per_class:
            continue
        ncc = _ncc_sliding(t, s)
        if score_sum is None:
            score_sum = w * ncc.astype(np.float32)
            weight_sum = w
            used_classes = 1
        else:
            if ncc.shape == score_sum.shape:
                score_sum = score_sum + w * ncc
                weight_sum += w
                used_classes += 1

    if score_sum is None or used_classes == 0 or weight_sum <= 0:
        return SemanticMaskFix(reject_reason="no_valid_channels",
                                n_drone_pixels=n_drone_total,
                                n_sat_pixels=n_sat_total)

    score_sum /= weight_sum
    py, px, peak, margin = _peak_with_margin(score_sum)

    # Пик корреляции задаёт метрический сдвиг относительно центра спутникового окна.
    centre_sat_y = py + drone_resized.shape[0] / 2.0
    centre_sat_x = px + drone_resized.shape[1] / 2.0
    sat_centre_y = sat_class_map.shape[0] / 2.0
    sat_centre_x = sat_class_map.shape[1] / 2.0
    dy_m = (centre_sat_y - sat_centre_y) * sat_mppx  # вниз по изображению — юг
    dx_m = (centre_sat_x - sat_centre_x) * sat_mppx  # вправо по изображению — восток

    lat0, lon0 = sat_centre_latlon
    cos_lat = math.cos(math.radians(lat0))
    lat = lat0 - dy_m / 111320.0
    lon = lon0 + dx_m / (111320.0 * cos_lat)

    if peak < accept_peak_threshold or margin < accept_margin_threshold:
        return SemanticMaskFix(
            reject_reason=f"low_peak={peak:.3f}_margin={margin:.3f}",
            lat=lat, lon=lon, peak_score=peak, peak_margin=margin,
            n_drone_pixels=n_drone_total, n_sat_pixels=n_sat_total,
            used_classes=used_classes,
        )

    # Чем чётче пик, тем меньше ковариация измерения.
    sigma = sigma_floor_m + (sigma_ceiling_m - sigma_floor_m) * (1.0 - min(1.0, margin / 0.2))
    return SemanticMaskFix(
        accepted=True,
        lat=lat, lon=lon, sigma_xy_m=sigma,
        peak_score=peak, peak_margin=margin,
        n_drone_pixels=n_drone_total, n_sat_pixels=n_sat_total,
        used_classes=used_classes,
    )
