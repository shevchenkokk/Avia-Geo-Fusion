"""Синтетическая межсезонная проверка структурного матчера.

Тесту не нужно полётное видео. Он растеризует векторы Overture для известного
региона, рендерит снегоподобный BEV-кадр из тех же структурных классов,
смещает начальное положение и проверяет, что StructuralMatcher восстанавливает
известный центр в пределах метрического допуска.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.frame_bridge import FrameBridge
from src.overture_v7_rasterizer import (
    CID_BUILT_UP,
    CID_FIELD_BOUNDARY,
    CID_FOREST_EDGE,
    CID_ROADS,
    CID_WATER,
    load_region_layers,
    rasterize_bbox,
)
from src.structural_matcher import StructuralMatcher


def _bbox_around(lat: float, lon: float, span_m: float) -> tuple[float, float, float, float]:
    cos_lat = math.cos(math.radians(lat))
    d_lat = 0.5 * span_m / 111320.0
    d_lon = 0.5 * span_m / (111320.0 * cos_lat)
    return (lon - d_lon, lat - d_lat, lon + d_lon, lat + d_lat)


def _centre_from_bbox(bbox: str) -> tuple[float, float]:
    west, south, east, north = map(float, bbox.split(","))
    return 0.5 * (south + north), 0.5 * (west + east)


def _select_structural_centre(
    meta_tiles: Path,
    layer_gdfs: dict,
    region_id: str,
    ground_span_m: float,
    out_size_px: int,
    sample_step: int,
) -> tuple[float, float, str, int]:
    rows = [r for r in csv.DictReader(meta_tiles.open()) if r["region_id"] == region_id]
    if not rows:
        raise ValueError(f"region_id={region_id!r} not found in {meta_tiles}")

    best: tuple[int, float, float, str] | None = None
    for row in rows[::max(1, sample_step)]:
        lat, lon = _centre_from_bbox(row["bbox_wgs84"])
        mask = rasterize_bbox(
            _bbox_around(lat, lon, ground_span_m),
            layer_gdfs,
            out_size=(out_size_px, out_size_px),
        )
        feature_px = int((mask > 0).sum())
        if best is None or feature_px > best[0]:
            best = (feature_px, lat, lon, row["tile_id"])

    assert best is not None
    feature_px, lat, lon, tile_id = best
    return lat, lon, tile_id, feature_px


def _render_snow_like_bev(mask: np.ndarray, seed: int) -> np.ndarray:
    gray = np.full(mask.shape, 225, dtype=np.uint8)
    class_tones = {
        CID_WATER: 170,
        CID_FOREST_EDGE: 35,
        CID_ROADS: 20,
        CID_FIELD_BOUNDARY: 55,
        CID_BUILT_UP: 80,
    }
    for cid, value in class_tones.items():
        gray[mask == cid] = value

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 10.0, gray.shape).astype(np.int16)
    gray = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=Path("data/overture_ru_dataset_starter"))
    p.add_argument("--region-id", type=str, default="moscow_city_small")
    p.add_argument("--output", type=Path, default=Path("results/stage5_structural"))
    p.add_argument("--ground-span-m", type=float, default=320.0)
    p.add_argument("--out-size-px", type=int, default=320)
    p.add_argument("--search-radius-m", type=float, default=220.0)
    p.add_argument("--seed-offset-east-m", type=float, default=-120.0)
    p.add_argument("--seed-offset-north-m", type=float, default=80.0)
    p.add_argument("--max-position-error-m", type=float, default=30.0)
    p.add_argument("--sample-step", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    vectors_root = args.dataset_root / "vectors"
    meta_tiles = args.dataset_root / "meta_tiles.csv"
    layer_gdfs = load_region_layers(vectors_root, args.region_id)
    if not layer_gdfs:
        raise RuntimeError(f"No vector layers loaded from {vectors_root / args.region_id}")

    lat, lon, tile_id, feature_px = _select_structural_centre(
        meta_tiles=meta_tiles,
        layer_gdfs=layer_gdfs,
        region_id=args.region_id,
        ground_span_m=args.ground_span_m,
        out_size_px=args.out_size_px,
        sample_step=args.sample_step,
    )
    target_mask = rasterize_bbox(
        _bbox_around(lat, lon, args.ground_span_m),
        layer_gdfs,
        out_size=(args.out_size_px, args.out_size_px),
    )
    frame_bev = _render_snow_like_bev(target_mask, seed=args.seed)

    bridge = FrameBridge(lat0=lat, lon0=lon, alt0_msl=0.0)
    seed_lat, seed_lon, _ = bridge.enu_to_wgs84(
        args.seed_offset_east_m,
        args.seed_offset_north_m,
        0.0,
    )

    matcher = StructuralMatcher(
        vectors_root=vectors_root,
        region_id=args.region_id,
        search_radius_m=args.search_radius_m,
    )
    fix = matcher.match(
        frame_bev=frame_bev,
        seed_lat=seed_lat,
        seed_lon=seed_lon,
        bev_ground_span_m=args.ground_span_m,
    )

    err_e, err_n, _ = bridge.wgs84_to_enu(fix.lat, fix.lon, 0.0)
    pos_err_m = float(math.hypot(err_e, err_n))
    passed = bool(fix.accepted and pos_err_m <= args.max_position_error_m)

    cv2.imwrite(str(args.output / "synthetic_bev.png"), frame_bev)
    score_path = args.output / "score_map.png"
    if fix.score_map is not None:
        score_norm = cv2.normalize(fix.score_map, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(str(score_path), score_norm.astype(np.uint8))

    summary = {
        "passed": passed,
        "region_id": args.region_id,
        "tile_id": tile_id,
        "target_lat": lat,
        "target_lon": lon,
        "seed_lat": seed_lat,
        "seed_lon": seed_lon,
        "accepted": fix.accepted,
        "reject_reason": fix.reject_reason,
        "position_error_m": pos_err_m,
        "peak_score": fix.peak_score,
        "secondary_score": fix.secondary_score,
        "sigma_xy_m": fix.sigma_xy_m,
        "feature_px_selected": feature_px,
        "n_drone_edges": fix.n_drone_edges,
        "n_sat_features": fix.n_sat_features,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[verify] synthetic structural matching")
    print(f"  region/tile : {args.region_id} / {tile_id}")
    print(f"  seed offset : E={args.seed_offset_east_m:.1f} N={args.seed_offset_north_m:.1f} m")
    print(f"  accepted    : {fix.accepted} {fix.reject_reason}")
    print(f"  pos err     : {pos_err_m:.2f} m (target <= {args.max_position_error_m:.1f})")
    print(f"  score/margin: {fix.peak_score:.3f} / {fix.peak_score - fix.secondary_score:.3f}")
    print(f"  sigma_xy    : {fix.sigma_xy_m:.1f} m")
    print(f"  output      : {args.output}")
    print()
    if passed:
        print("[verify] CRITERION PASSED")
        return
    print("[verify] CRITERION NOT MET")
    sys.exit(2)


if __name__ == "__main__":
    main()
