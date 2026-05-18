"""Структурный матчер — многоканальная NCC семантических карт.

Сравнивает карту рёбер самолёта (Canny на BEV-кадре) или бортовую
семантическую class-map с растеризованными векторными слоями Overture.
Векторные данные инвариантны к сезону, поэтому метод работает там, где
appearance-матчер (XFeat) отказывает.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.overture_v7_rasterizer import (
    CID_BUILT_UP,
    CID_FIELD_BOUNDARY,
    CID_FOREST_EDGE,
    CID_ROADS,
    CID_WATER,
    load_region_layers,
    rasterize_bbox,
)

# Веса классов для NCC (§4 PROJECT_PLAN.md).
# Вода снижена: замёрзшие озёра неотличимы от окружающей местности в BEV.
# Дороги / граница леса / застройка несут основную долю счёта.
_DEFAULT_CLASS_WEIGHTS = {
    CID_WATER: 0.5,
    CID_FOREST_EDGE: 1.5,
    CID_ROADS: 2.0,
    CID_FIELD_BOUNDARY: 1.0,
    CID_BUILT_UP: 1.5,
}


@dataclass
class StructuralFix:
    accepted: bool = False
    reject_reason: str = ""
    lat: float = float("nan")
    lon: float = float("nan")
    sigma_xy_m: float = float("nan")
    peak_score: float = 0.0
    peak_dx_px: int = 0
    peak_dy_px: int = 0
    secondary_score: float = 0.0
    n_drone_edges: int = 0
    n_sat_features: int = 0
    semantic_mode: bool = False
    score_map: Optional[np.ndarray] = field(default=None, repr=False)


def _drone_edge_map(
    frame_bev: np.ndarray,
    lo: int = 50,
    hi: int = 150,
    bev_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if frame_bev.ndim == 3:
        gray = cv2.cvtColor(frame_bev, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame_bev
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, lo, hi)
    if bev_mask is not None:
        mask = bev_mask
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.shape != edges.shape:
            mask = cv2.resize(
                mask,
                (edges.shape[1], edges.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        mask = (mask > 0).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=1)
        edges[mask > 0] = 0
    return edges


def _per_class_channels(mask_v7: np.ndarray) -> dict[int, np.ndarray]:
    channels = {}
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for cid in (
        CID_WATER,
        CID_FOREST_EDGE,
        CID_ROADS,
        CID_FIELD_BOUNDARY,
        CID_BUILT_UP,
    ):
        filled = (mask_v7 == cid).astype(np.uint8) * 255
        if (filled > 0).sum() == 0:
            channels[cid] = filled
            continue
        edges = cv2.Canny(filled, 50, 150)
        channels[cid] = cv2.dilate(edges, kernel, iterations=1)
    return channels


# Текущие чекпойнты SegFormer для бортовой стороны используют OVERTURE_RU_5:
#   1 вода, 2 растительность, 3 здания, 4 дороги.
# Структурный растр Overture использует OVERTURE_RU_7:
#   1 вода, 2 граница леса, 3 дороги, 4 граница поля, 5 застройка.
# Растительность переводится в шаблон рёбер, поэтому ближе всего соответствует
# границе леса, а не сплошной заливке растительности.
_OVERTURE_RU_5_TO_V7 = {
    1: CID_WATER,
    2: CID_FOREST_EDGE,
    3: CID_BUILT_UP,
    4: CID_ROADS,
}


def _resize_nearest(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    if mask.shape[:2] == shape_hw:
        return mask
    h, w = shape_hw
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def _apply_exclusion_mask(
    channel: np.ndarray, bev_mask: Optional[np.ndarray]
) -> np.ndarray:
    if bev_mask is None:
        return channel
    mask = bev_mask
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = _resize_nearest(mask, channel.shape[:2])
    out = channel.copy()
    out[mask > 0] = 0
    return out


def _drone_semantic_channels(
    class_map: np.ndarray,
    bev_mask: Optional[np.ndarray] = None,
) -> dict[int, np.ndarray]:
    """Бортовая карта классов -> поклассовые шаблоны рёбер в ID Overture-v7.

    Поддерживается текущий 5-классовый чекпойнт SegFormer и возможный будущий
    чекпойнт v7. Заполненные маски перед NCC переводятся в рёбра, чтобы
    уменьшить разрыв между шумной бортовой сегментацией и чистой геометрией
    Overture.
    """
    if class_map.ndim == 3:
        class_map = class_map[:, :, 0]
    class_map = class_map.astype(np.uint8, copy=False)
    present = set(int(v) for v in np.unique(class_map) if int(v) > 0)

    # Если ID похожи на OVERTURE_RU_5, переводим их в v7. Если будущий
    # чекпойнт сразу выдаёт v7, оставляем ID как есть.
    if present and max(present) <= 4:
        mapping = _OVERTURE_RU_5_TO_V7
    else:
        mapping = {
            CID_WATER: CID_WATER,
            CID_FOREST_EDGE: CID_FOREST_EDGE,
            CID_ROADS: CID_ROADS,
            CID_FIELD_BOUNDARY: CID_FIELD_BOUNDARY,
            CID_BUILT_UP: CID_BUILT_UP,
        }

    merged: dict[int, np.ndarray] = {}
    for src_id, dst_id in mapping.items():
        if src_id not in present:
            continue
        filled = (class_map == src_id).astype(np.uint8) * 255
        filled = _apply_exclusion_mask(filled, bev_mask)
        if (filled > 0).sum() < 50:
            continue
        kernel_mid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel_mid)
        filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel_mid)
        edges = cv2.Canny(filled, 50, 150)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel_small, iterations=1)
        if dst_id in merged:
            merged[dst_id] = cv2.bitwise_or(merged[dst_id], edges)
        else:
            merged[dst_id] = edges

    return merged


def _ncc_sliding(template: np.ndarray, scene: np.ndarray) -> np.ndarray:
    """Нормированная кросс-корреляция по всем допустимым смещениям.

    template: (h, w) uint8. scene: (H, W) uint8, H≥h, W≥w.
    Возвращает (H-h+1, W-w+1) float32 со значениями в [-1, 1].
    """
    if scene.shape[0] < template.shape[0] or scene.shape[1] < template.shape[1]:
        return np.zeros((1, 1), dtype=np.float32)
    return cv2.matchTemplate(scene, template, cv2.TM_CCOEFF_NORMED)


class StructuralMatcher:
    """Многоканальный NCC-матчер: карта рёбер самолёта ↔ растеризованные векторы Overture.

    Хранит векторные слои региона в памяти (несколько сотен МБ для области 32×32 км).
    """

    def __init__(
        self,
        vectors_root: Path | str,
        region_id: str,
        bev_meters_per_pixel: float = 0.5,
        search_radius_m: float = 600.0,
        class_weights: Optional[dict[int, float]] = None,
        forest_edge_width_m: float = 12.0,
        field_boundary_width_m: float = 8.0,
    ) -> None:
        self.vectors_root = Path(vectors_root)
        self.region_id = region_id
        self.bev_mppx = float(bev_meters_per_pixel)
        self.search_radius_m = float(search_radius_m)
        self.class_weights = (
            dict(class_weights) if class_weights else _DEFAULT_CLASS_WEIGHTS
        )
        self.forest_edge_width_m = float(forest_edge_width_m)
        self.field_boundary_width_m = float(field_boundary_width_m)

        self.layer_gdfs = load_region_layers(self.vectors_root, region_id)
        loaded = [k for k, v in self.layer_gdfs.items() if v is not None and len(v) > 0]
        self._has_data = len(loaded) > 0
        if not self._has_data:
            raise FileNotFoundError(
                f"StructuralMatcher: no vector data in {self.vectors_root / region_id}"
            )

    def match(
        self,
        frame_bev: np.ndarray,
        seed_lat: float,
        seed_lon: float,
        bev_ground_span_m: float,
        bev_mask: Optional[np.ndarray] = None,
        search_radius_m: Optional[float] = None,
        drone_class_map: Optional[np.ndarray] = None,
    ) -> StructuralFix:
        """NCC-поиск вокруг начального положения.

        Аргументы:
            frame_bev: BEV-кадр самолёта (квадратный, BGR).
            seed_lat, seed_lon: грубая позиция от retriever.
            bev_ground_span_m: сторона охвата BEV в метрах (соответствует
                BevRectifier.ground_span_m).
            bev_mask: опциональная BEV-маска, исключаемая после Canny.
            search_radius_m: переопределяет ``self.search_radius_m`` для
                конкретного вызова. Например, радиус можно масштабировать
                от текущей EKF σ_pos: в TRACK искать в узком окне дешевле.
            drone_class_map: опциональная SegFormer/semantic class-map для
                настоящего поклассового режима mask-to-mask. Если не задана,
                используется старый шаблон рёбер Canny.
        """
        bev_h, bev_w = frame_bev.shape[:2]
        bev_mppx = bev_ground_span_m / bev_w  # квадратный BEV по соглашению

        semantic_mode = drone_class_map is not None
        e_drone = None
        drone_channels: dict[int, np.ndarray] = {}
        if semantic_mode:
            drone_class_map = _resize_nearest(drone_class_map, (bev_h, bev_w))
            drone_channels = _drone_semantic_channels(
                drone_class_map, bev_mask=bev_mask
            )
            n_drone_edges = int(sum((ch > 0).sum() for ch in drone_channels.values()))
            if n_drone_edges < 200:
                return StructuralFix(
                    reject_reason="too_few_drone_semantic_edges",
                    n_drone_edges=n_drone_edges,
                    semantic_mode=True,
                )
        else:
            e_drone = _drone_edge_map(frame_bev, bev_mask=bev_mask)
            n_drone_edges = int((e_drone > 0).sum())
            if n_drone_edges < 200:
                return StructuralFix(
                    reject_reason="too_few_drone_edges", n_drone_edges=n_drone_edges
                )

        eff_radius_m = (
            self.search_radius_m if search_radius_m is None else float(search_radius_m)
        )
        half_extent_m = 0.5 * bev_ground_span_m + eff_radius_m
        cos_lat = math.cos(math.radians(seed_lat))
        d_lat = half_extent_m / 111320.0
        d_lon = half_extent_m / (111320.0 * cos_lat)
        bbox = (seed_lon - d_lon, seed_lat - d_lat, seed_lon + d_lon, seed_lat + d_lat)

        # Спутниковый растр с тем же мпп, что и BEV: смещения NCC прямо в метрах.
        sat_w = int(round(2 * half_extent_m / bev_mppx))
        sat_h = sat_w
        try:
            v7_mask = rasterize_bbox(
                bbox,
                self.layer_gdfs,
                out_size=(sat_w, sat_h),
                forest_edge_width_m=self.forest_edge_width_m,
                field_boundary_width_m=self.field_boundary_width_m,
            )
        except Exception as e:
            return StructuralFix(reject_reason=f"rasterize_failed: {e}")

        n_sat_features = int((v7_mask > 0).sum())
        if n_sat_features < 1000:
            return StructuralFix(
                reject_reason="too_few_sat_features",
                n_sat_features=n_sat_features,
                n_drone_edges=n_drone_edges,
            )

        sat_channels = _per_class_channels(v7_mask)
        score_sum: Optional[np.ndarray] = None
        weight_sum = 0.0
        for cid, channel in sat_channels.items():
            if (channel > 0).sum() < 50:
                continue
            w = self.class_weights.get(cid, 1.0)
            if w <= 0:
                continue
            if semantic_mode:
                template = drone_channels.get(cid)
                if template is None or (template > 0).sum() < 50:
                    continue
            else:
                template = e_drone
            if template is None:
                continue
            ncc = _ncc_sliding(template, channel)
            if ncc.size == 0:
                continue
            if score_sum is None:
                score_sum = w * ncc
            else:
                score_sum = score_sum + w * ncc
            weight_sum += w

        if score_sum is None or weight_sum == 0:
            return StructuralFix(
                reject_reason="no_valid_channels",
                n_drone_edges=n_drone_edges,
                n_sat_features=n_sat_features,
                semantic_mode=semantic_mode,
            )
        score_map = (score_sum / weight_sum).astype(np.float32)

        _, peak_score, _, peak_loc = cv2.minMaxLoc(score_map)
        peak_x, peak_y = peak_loc

        # Вторичный пик для оценки σ: маскируем 80-метровую окрестность первичного.
        mask = score_map.copy()
        nbr = max(8, int(round(80.0 / bev_mppx)))
        x0 = max(0, peak_x - nbr)
        x1 = min(score_map.shape[1], peak_x + nbr + 1)
        y0 = max(0, peak_y - nbr)
        y1 = min(score_map.shape[0], peak_y + nbr + 1)
        mask[y0:y1, x0:x1] = -1.0
        _, secondary_score, _, _ = cv2.minMaxLoc(mask)

        # Перевод пика в (lat, lon): score_map[0,0] соответствует шаблону в пикселе (0,0)
        # спутника; пик (peak_x, peak_y) → центр BEV совпадает с пикселем
        # (peak_x + bev_w/2, peak_y + bev_h/2).
        sat_centre_px_x = peak_x + bev_w // 2
        sat_centre_px_y = peak_y + bev_h // 2
        west, south, east, north = bbox
        lat = north - (sat_centre_px_y / sat_h) * (north - south)
        lon = west + (sat_centre_px_x / sat_w) * (east - west)

        # σ из остроты пика: margin = peak - secondary.
        # Большая margin → острый пик → меньше σ.
        # Эвристика: σ_xy_m = clip(100 * (1 - margin), 15, 200) м.
        # Качественная оценка; ворота Махаланобиса EKF дополнительно фильтруют.
        margin = float(peak_score - secondary_score)
        sigma_xy_m = float(np.clip(100.0 * (1.0 - margin), 15.0, 200.0))

        accepted = peak_score > 0.05 and margin > 0.005
        reject_reason = ""
        if not accepted:
            reject_reason = "peak_too_weak" if peak_score <= 0.05 else "low_margin"

        return StructuralFix(
            accepted=accepted,
            reject_reason=reject_reason,
            lat=float(lat),
            lon=float(lon),
            sigma_xy_m=sigma_xy_m,
            peak_score=float(peak_score),
            peak_dx_px=int(peak_x),
            peak_dy_px=int(peak_y),
            secondary_score=float(secondary_score),
            n_drone_edges=n_drone_edges,
            n_sat_features=n_sat_features,
            semantic_mode=semantic_mode,
            score_map=score_map,
        )
