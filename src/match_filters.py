"""
Class-aware фильтрация keypoint-match'ей по семантическим картам.

Контекст. LoFTR/LightGlue на кросс-виде (наклонный кадр самолёта ↔ надирный спутник)
даёт много визуально-похожих, но семантически-бессмысленных соответствий:
угол здания А на кадре ↔ угол здания Б на карте, перекрёсток X ↔ перекрёсток Y,
и т.п. Бинарный «гейт по земле» такие пары пропускает, и они протаскивают
RANSAC'у фальшивые inliers — отсюда скачки гомографии и дребезг траектории.

Класс-aware фильтр: у каждой точки смотрим класс на кадре и на карте и
требуем, чтобы они были совместимыми. По умолчанию «совместимый» = «такой же»
(road↔road, building↔building, vegetation↔vegetation). Это отсекает основную
массу перекрёстных ложняков до RANSAC.
"""

from __future__ import annotations

import numpy as np
from collections import Counter
from typing import Iterable

from src.class_schemas import ClassSchema, OVERTURE_RU_5
from src.geo_segmentor import HEURISTIC_GROUND_ID


def _sample_classes(class_map: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Для каждой точки (x, y) возвращает class_id из class_map.
    Точки вне границ получают 0 (фон / не-стабильный)."""
    if len(pts) == 0:
        return np.empty(0, dtype=np.uint8)
    h, w = class_map.shape[:2]
    xs = np.clip(np.round(pts[:, 0]).astype(np.int32), 0, w - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(np.int32), 0, h - 1)
    inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    sampled = class_map[ys, xs].astype(np.uint8)
    sampled[~inside] = 0
    return sampled


def build_default_compatibility(schema: ClassSchema) -> dict[int, set[int]]:
    """Стартовая таблица совместимости: каждый стабильный класс совместим только
    сам с собой. HEURISTIC_GROUND_ID совместим со всеми стабильными классами —
    это fallback, когда одна из сторон не даёт семантики.
    """
    compat: dict[int, set[int]] = {cid: {cid} for cid in schema.stable_ids}
    # Эвристическая «земля» — wildcard: совпадает с любым стабильным классом
    # и сама с собой.
    heuristic_set = set(schema.stable_ids) | {HEURISTIC_GROUND_ID}
    compat[HEURISTIC_GROUND_ID] = heuristic_set
    for cid in schema.stable_ids:
        compat[cid].add(HEURISTIC_GROUND_ID)
    return compat


def filter_matches_by_class(
    pts_drone: np.ndarray,
    pts_map: np.ndarray,
    class_map_drone: np.ndarray,
    class_map_map: np.ndarray,
    schema: ClassSchema = OVERTURE_RU_5,
    compatibility: dict[int, set[int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Оставляет только совместимые по классу пары keypoints.

    :returns: (kept_pts_drone, kept_pts_map, stats)
      где stats содержит:
        raw                — число поданных пар
        kept               — число оставленных пар
        rejected_background— пары, где хотя бы одна сторона = 0 (не-стабильная)
        rejected_cross     — пары с несовместимыми классами (наш основной сигнал)
        by_class           — {class_id: count} распределение оставленных пар
    """
    raw = int(len(pts_drone))
    empty = np.empty((0, 2), dtype=np.float32)
    stats = {
        "raw": raw,
        "kept": 0,
        "rejected_background": 0,
        "rejected_cross": 0,
        "by_class": {},
    }
    if raw == 0:
        return empty, empty, stats

    if compatibility is None:
        compatibility = build_default_compatibility(schema)

    cls_drone = _sample_classes(class_map_drone, pts_drone)
    cls_map = _sample_classes(class_map_map, pts_map)

    bg_mask = (cls_drone == 0) | (cls_map == 0)
    stats["rejected_background"] = int(bg_mask.sum())

    keep = np.zeros(raw, dtype=bool)
    for i in range(raw):
        if bg_mask[i]:
            continue
        cd, cm = int(cls_drone[i]), int(cls_map[i])
        allowed = compatibility.get(cd, {cd})
        if cm in allowed:
            keep[i] = True

    stats["kept"] = int(keep.sum())
    stats["rejected_cross"] = raw - stats["kept"] - stats["rejected_background"]

    if stats["kept"] > 0:
        kept_classes = cls_drone[keep]
        # При совпадении классов самолёт↔карта берём сторону самолёта для сводки;
        # wildcard HEURISTIC_GROUND_ID — подменяем на класс map, если тот реальный.
        kept_pairs = list(zip(kept_classes.tolist(), cls_map[keep].tolist()))
        resolved = [
            cm if cd == HEURISTIC_GROUND_ID else cd
            for cd, cm in kept_pairs
        ]
        stats["by_class"] = dict(Counter(resolved))

    return pts_drone[keep], pts_map[keep], stats


def format_class_stats(stats: dict, schema: ClassSchema = OVERTURE_RU_5) -> str:
    """Короткая однострочная сводка для HUD/лога.
    Пример: 'kept 42/128 | ROAD=31 BLDG=9 VEG=2 | cross=74 bg=12'
    """
    by_class = stats.get("by_class", {})
    by_name: list[str] = []
    name_by_id = schema.names
    role_abbrev = {"road": "ROAD", "building": "BLDG", "vegetation": "VEG",
                   "water": "WATER", "barren": "BARE", "background": "BG"}
    # Упорядочим по ролям, чтобы строка была стабильной между кадрами
    order = ["road", "building", "vegetation", "barren", "water"]
    role_of = {e.id: e.role for e in schema.entries}
    # Добавим heuristic ground как GND
    if HEURISTIC_GROUND_ID in by_class:
        by_name.append(f"GND={by_class[HEURISTIC_GROUND_ID]}")
    for role in order:
        ids_in_role = [cid for cid, r in role_of.items() if r == role and cid in by_class]
        for cid in ids_in_role:
            abbr = role_abbrev.get(role, name_by_id.get(cid, str(cid)).upper()[:4])
            by_name.append(f"{abbr}={by_class[cid]}")
    classes_str = " ".join(by_name) if by_name else "-"
    return (
        f"kept {stats.get('kept', 0)}/{stats.get('raw', 0)} | {classes_str} "
        f"| cross={stats.get('rejected_cross', 0)} bg={stats.get('rejected_background', 0)}"
    )
