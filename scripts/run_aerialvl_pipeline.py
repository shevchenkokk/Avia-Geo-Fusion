"""Полный пайплайн валидации на последовательностях AerialVL VAL.

AerialVL устроен не как обычный `.mp4`: кадры лежат отдельными PNG, а эталонные
широта/долгота записаны прямо в имени файла. Поэтому для него нужен отдельный
скрипт запуска, но внутри он использует те же ключевые блоки проекта:

  кадры AerialVL → VO по оптическому потоку → EKF → абсолютное измерение по карте → метрики по эталону.

Это не заменяет `scripts/run_full_pipeline.py` для GP010269.MP4, а даёт внешний
реальный тестовый набор, где есть траектория и координаты каждого кадра.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ekf import StateFilter
from src.frame_bridge import FrameBridge
from src.geo_segmentor import GeoSegmentor
from src.map_measurement import compute_map_measurement
from src.neural_matching import NeuralMatcher
from src.optical_flow_vo import OpticalFlowVO
from src.semantic_mask_matcher import mask_to_mask_match


@dataclass(frozen=True)
class ValFrame:
    index: int
    split: str
    sequence: str
    path: Path
    rel_path: str
    utc_timestamp: str
    lat: float
    lon: float


@dataclass(frozen=True)
class GeoMap:
    name: str
    path: Path
    rel_path: str
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float
    center_lat: float
    center_lon: float


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_image_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")
    return img


def _timestamp_to_seconds(value: str, fallback: float) -> float:
    try:
        raw = float(value)
    except ValueError:
        return fallback
    # В AerialVL имена кадров обычно содержат Unix timestamp в миллисекундах.
    # Для EKF нужен dt в секундах, иначе ковариация раздувается в тысячу раз.
    if raw > 1e10:
        return raw / 1000.0
    return raw


def _dist_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat_mid = math.radians(0.5 * (lat1 + lat2))
    dy = (lat2 - lat1) * 111_320.0
    dx = (lon2 - lon1) * 111_320.0 * math.cos(lat_mid)
    return float(math.hypot(dx, dy))


def _load_val_frames(
    dataset_root: Path,
    manifest_dir: Path,
    sequence: str | None,
    split: str | None,
) -> list[ValFrame]:
    rows = _read_csv(manifest_dir / "val_frames.csv")
    frames: list[ValFrame] = []
    for row in rows:
        if sequence is not None and row["sequence"] != sequence:
            continue
        if split is not None and row["split"] != split:
            continue
        rel_path = row["path"]
        frames.append(
            ValFrame(
                index=len(frames),
                split=row["split"],
                sequence=row["sequence"],
                path=dataset_root / rel_path,
                rel_path=rel_path,
                utc_timestamp=row["utc_timestamp"],
                lat=float(row["lat"]),
                lon=float(row["lon"]),
            )
        )
    frames.sort(
        key=lambda item: (
            _timestamp_to_seconds(item.utc_timestamp, item.index),
            item.rel_path,
        )
    )
    return frames


def _load_maps(dataset_root: Path, manifest_dir: Path) -> list[GeoMap]:
    rows = _read_csv(manifest_dir / "val_maps.csv")
    maps: list[GeoMap] = []
    for row in rows:
        rel_path = row["path"]
        maps.append(
            GeoMap(
                name=row["name"],
                path=dataset_root / rel_path,
                rel_path=rel_path,
                lon_min=float(row["lon_min"]),
                lat_min=float(row["lat_min"]),
                lon_max=float(row["lon_max"]),
                lat_max=float(row["lat_max"]),
                center_lat=float(row["center_lat"]),
                center_lon=float(row["center_lon"]),
            )
        )
    return maps


def _map_contains(map_row: GeoMap, lat: float, lon: float) -> bool:
    return (
        map_row.lat_min <= lat <= map_row.lat_max
        and map_row.lon_min <= lon <= map_row.lon_max
    )


def _select_map(
    maps: list[GeoMap], frames: list[ValFrame], requested_name: str | None
) -> GeoMap:
    if requested_name is not None:
        for item in maps:
            if (
                item.name == requested_name
                or Path(item.rel_path).name == requested_name
            ):
                return item
        raise ValueError(f"Карта с именем {requested_name!r} не найдена в val_maps.csv")

    if not maps:
        raise ValueError("В manifest-файле нет ни одной VAL-карты")

    best_map = maps[0]
    best_count = -1
    for item in maps:
        count = sum(1 for frame in frames if _map_contains(item, frame.lat, frame.lon))
        if count > best_count:
            best_map = item
            best_count = count
    return best_map


def _latlon_to_pixel(
    lat: float, lon: float, map_row: GeoMap, shape: tuple[int, int]
) -> tuple[float, float]:
    h, w = shape[:2]
    x = (lon - map_row.lon_min) / max(map_row.lon_max - map_row.lon_min, 1e-12) * w
    y = (map_row.lat_max - lat) / max(map_row.lat_max - map_row.lat_min, 1e-12) * h
    return float(x), float(y)


def _crop_map_patch(
    map_img: np.ndarray,
    map_row: GeoMap,
    center_lat: float,
    center_lon: float,
    span_m: float,
    out_size: int,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    h, w = map_img.shape[:2]
    x, y = _latlon_to_pixel(center_lat, center_lon, map_row, map_img.shape)

    lat_mid = math.radians(center_lat)
    half_lat_deg = 0.5 * span_m / 111_320.0
    half_lon_deg = 0.5 * span_m / max(111_320.0 * math.cos(lat_mid), 1e-9)
    patch_bbox = (
        center_lon - half_lon_deg,
        center_lat - half_lat_deg,
        center_lon + half_lon_deg,
        center_lat + half_lat_deg,
    )

    px_per_m_x = w / max(
        (map_row.lon_max - map_row.lon_min) * 111_320.0 * math.cos(lat_mid), 1e-9
    )
    px_per_m_y = h / max((map_row.lat_max - map_row.lat_min) * 111_320.0, 1e-9)
    crop_w = max(4, int(round(span_m * px_per_m_x)))
    crop_h = max(4, int(round(span_m * px_per_m_y)))

    x0 = int(round(x - crop_w / 2))
    y0 = int(round(y - crop_h / 2))
    x1 = x0 + crop_w
    y1 = y0 + crop_h

    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(w, x1)
    src_y1 = min(h, y1)

    patch = np.full((crop_h, crop_w, 3), 128, dtype=np.uint8)
    if src_x1 > src_x0 and src_y1 > src_y0:
        dst_x0 = src_x0 - x0
        dst_y0 = src_y0 - y0
        patch[
            dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)
        ] = map_img[src_y0:src_y1, src_x0:src_x1]
    patch = cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return patch, patch_bbox


def _rotate_frame_for_map(frame: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return frame
    if mode == "cw90":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if mode == "ccw90":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if mode == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    raise ValueError(f"Неизвестный режим поворота кадра: {mode}")


def _make_camera(
    frame_shape: tuple[int, int], focal_scale: float
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    h, w = frame_shape[:2]
    f = focal_scale * max(w, h)
    k = np.array(
        [
            [f, 0.0, w / 2.0],
            [0.0, f, h / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    d = np.zeros((4, 1), dtype=np.float64)
    return k, d, (w, h)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_trajectory(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    if not rows:
        return
    gt_e = np.array([row["gt_x_e"] for row in rows], dtype=np.float64)
    gt_n = np.array([row["gt_y_n"] for row in rows], dtype=np.float64)
    est_e = np.array([row["x_e"] for row in rows], dtype=np.float64)
    est_n = np.array([row["y_n"] for row in rows], dtype=np.float64)
    accepted_e = np.array(
        [row["x_e"] for row in rows if row["map_accepted"]], dtype=np.float64
    )
    accepted_n = np.array(
        [row["y_n"] for row in rows if row["map_accepted"]], dtype=np.float64
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(gt_e, gt_n, label="эталон AerialVL", linewidth=2)
    ax.plot(est_e, est_n, label="оценка EKF", linewidth=1.5)
    if len(accepted_e) > 0:
        ax.scatter(accepted_e, accepted_n, s=14, label="принятые map-fix")
    ax.set_title(title)
    ax.set_xlabel("Восток, м")
    ax.set_ylabel("Север, м")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# AerialVL: валидация полного пайплайна",
        "",
        f"Статус: {'PASS' if payload['passed'] else 'FAIL'}",
        "",
        "## Эксперимент",
        "",
        f"- split: `{payload['split']}`",
        f"- sequence: `{payload['sequence']}`",
        f"- map: `{payload['map']}`",
        f"- frames: `{payload['frames']}`",
        f"- backend: `{payload['backend']}`",
        f"- oracle_map_crop: `{payload.get('oracle_map_crop', False)}`",
        f"- agl_m: `{payload.get('agl_m', 'N/A')}`",
        f"- map_crop_span_m: `{payload.get('map_crop_span_m', 'N/A')}`",
        "",
        "## Метрики траектории",
        "",
        f"- median_error_m: `{payload['median_error_m']:.2f}`",
        f"- p95_error_m: `{payload['p95_error_m']:.2f}`",
        f"- mean_error_m: `{payload['mean_error_m']:.2f}`",
        f"- final_error_m: `{payload['final_error_m']:.2f}`",
        f"- track_pct: `{payload['track_pct']:.2f}`",
        "",
        "## Каналы",
        "",
        f"- vo_valid_pct: `{payload['vo_valid_pct']:.2f}`",
        f"- map_attempts: `{payload['map_attempts']}`",
        f"- map_accepted: `{payload['map_accepted']}`",
        f"- map_accept_pct: `{payload['map_accept_pct']:.2f}`",
        "",
        "## Файлы",
        "",
        f"- state_csv: `{payload['state_csv']}`",
        f"- trajectory_png: `{payload['trajectory_png']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="путь к распакованному каталогу AerialVL",
    )
    parser.add_argument(
        "--manifest-dir", type=Path, default=Path("data/aerialvl/manifests")
    )
    parser.add_argument(
        "--split", default=None, help="опционально: long_trajtr или short_trajtr"
    )
    parser.add_argument(
        "--sequence",
        default=None,
        help="имя последовательности; если не задано, берётся первая подходящая",
    )
    parser.add_argument(
        "--map-name",
        default=None,
        help="имя карты из val_maps.csv; если не задано, выбирается карта с максимальным покрытием",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/aerialvl_full_pipeline")
    )
    parser.add_argument(
        "--backend", default="orb", choices=["orb", "xfeat", "loftr", "lightglue"]
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="0 означает обработать всю выбранную последовательность",
    )
    parser.add_argument(
        "--fps-fallback",
        type=float,
        default=1.0,
        help="частота кадров, если timestamp в имени файла не числовой",
    )
    parser.add_argument("--map-period-frames", type=int, default=5)
    parser.add_argument("--map-crop-span-m", type=float, default=500.0)
    parser.add_argument("--map-crop-size", type=int, default=900)
    parser.add_argument(
        "--oracle-map-crop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "диагностический режим: брать crop карты вокруг GT-точки кадра, "
            "а не вокруг прогноза EKF; помогает отделить ошибку VO от ошибки матчера"
        ),
    )
    parser.add_argument(
        "--rotate-frame",
        default="cw90",
        choices=["none", "cw90", "ccw90", "180"],
        help="AerialVL VAL-кадры ориентированы на восток; cw90 обычно совмещает их с картой, где север сверху",
    )
    parser.add_argument("--agl-m", type=float, default=300.0)
    parser.add_argument("--pitch-deg", type=float, default=-90.0)
    parser.add_argument("--camera-focal-scale", type=float, default=0.9)
    parser.add_argument("--sigma-map-floor-m", type=float, default=10.0)
    parser.add_argument("--sigma-map-ceiling-m", type=float, default=250.0)
    parser.add_argument("--track-threshold-m", type=float, default=100.0)
    parser.add_argument(
        "--pass-p95-threshold-m",
        type=float,
        default=100.0,
        help="критерий PASS: p95 ошибки должен быть не выше этого порога",
    )
    parser.add_argument(
        "--pass-track-pct-threshold",
        type=float,
        default=80.0,
        help="критерий PASS: доля кадров внутри --track-threshold-m должна быть не ниже этого процента",
    )
    parser.add_argument("--sigma-v-mps", type=float, default=4.0)
    parser.add_argument(
        "--semantic-mask",
        action="store_true",
        default=False,
        help="фильтровать appearance keypoints по стабильному классу спутниковой маски (SegFormer)",
    )
    parser.add_argument(
        "--semantic-structural-match",
        action="store_true",
        default=False,
        help="включить независимый mask-to-mask NCC канал (SegFormer на обеих сторонах)",
    )
    parser.add_argument(
        "--struct-search-radius-m",
        type=float,
        default=200.0,
        help="радиус поиска NCC для structural канала (используется через padding sat-маски)",
    )
    parser.add_argument(
        "--seg-config",
        default="results/segformer_overture_b0_phase_c_osm_manualcw/segformer_overture_quick_cfg.py",
        help="mmseg config для GeoSegmentor",
    )
    parser.add_argument(
        "--seg-checkpoint",
        default="results/segformer_overture_b0_phase_c_osm_manualcw/best_mIoU_iter_1165.pth",
        help="mmseg checkpoint для GeoSegmentor",
    )
    args = parser.parse_args()

    if args.frame_stride <= 0:
        parser.error("--frame-stride должен быть положительным")
    if args.map_period_frames <= 0:
        parser.error("--map-period-frames должен быть положительным")

    dataset_root = args.dataset_root.resolve()
    manifest_dir = args.manifest_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    frames = _load_val_frames(dataset_root, manifest_dir, args.sequence, args.split)
    if not frames:
        parser.error(
            "не найдено кадров VAL; сначала запустите scripts/prepare_aerialvl.py"
        )
    if args.sequence is None:
        first_seq = frames[0].sequence
        frames = [frame for frame in frames if frame.sequence == first_seq]
    frames = frames[:: args.frame_stride]
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    if len(frames) < 2:
        parser.error("для пайплайна нужно минимум два кадра")

    maps = _load_maps(dataset_root, manifest_dir)
    selected_map = _select_map(maps, frames, args.map_name)
    map_img = _read_image_bgr(selected_map.path)

    first_frame_raw = _read_image_bgr(frames[0].path)
    first_frame = _rotate_frame_for_map(first_frame_raw, args.rotate_frame)
    k, d, image_size = _make_camera(first_frame.shape, args.camera_focal_scale)

    bridge = FrameBridge(lat0=frames[0].lat, lon0=frames[0].lon, alt0_msl=0.0)
    filt = StateFilter(bridge)
    filt.initialize_from_wgs84(
        frames[0].lat,
        frames[0].lon,
        0.0,
        yaw=0.0,
        sigma_pos_m=20.0,
        sigma_yaw_rad=math.pi,
    )
    filt.P[StateFilter.IDX_VX_E, StateFilter.IDX_VX_E] = 30.0**2
    filt.P[StateFilter.IDX_VY_N, StateFilter.IDX_VY_N] = 30.0**2

    vo = OpticalFlowVO(
        K=k,
        D=d,
        image_size=image_size,
        pitch_deg=args.pitch_deg,
        min_features_per_frame=20,
        min_inliers=5,
        ransac_threshold_m=8.0,
    )
    matcher = NeuralMatcher(backend=args.backend)

    seg = None
    need_seg = args.semantic_mask or args.semantic_structural_match
    if need_seg:
        import torch as _torch
        seg_device = "mps" if _torch.backends.mps.is_available() else "cpu"
        seg = GeoSegmentor(
            backend="mmseg",
            mmseg_config=str(args.seg_config),
            mmseg_checkpoint=str(args.seg_checkpoint),
            mmseg_device=seg_device,
            stable_class_ids={1, 2, 3, 4},
            top_crop_ratio=0.0,
        )
        if seg.backend != "mmseg":
            print("[aerialvl-full] WARNING: GeoSegmentor failed to load — semantic features disabled")
            seg = None
        else:
            print(f"[aerialvl-full] semantic_mask    : {'ON' if args.semantic_mask else 'off'}")
            print(f"[aerialvl-full] structural_match : {'ON' if args.semantic_structural_match else 'off'}")
            print(f"[aerialvl-full] seg_ckpt         : {args.seg_checkpoint}")

    rows: list[dict[str, Any]] = []
    map_attempts = 0
    map_accepted = 0
    struct_attempts = 0
    struct_accepted = 0
    vo_steps = 0
    vo_valid = 0
    last_t: float | None = None

    print(f"[aerialvl-full] sequence : {frames[0].split}/{frames[0].sequence}")
    print(f"[aerialvl-full] frames   : {len(frames)}")
    print(f"[aerialvl-full] map      : {selected_map.rel_path}")
    print(f"[aerialvl-full] backend  : {matcher.backend} (запрошен {args.backend})")
    print()

    for i, frame_info in enumerate(frames):
        t_raw = _timestamp_to_seconds(frame_info.utc_timestamp, i / args.fps_fallback)
        if last_t is None:
            dt = 1.0 / args.fps_fallback
        else:
            dt = max(1e-3, t_raw - last_t)
        last_t = t_raw

        raw_frame = _read_image_bgr(frame_info.path)
        frame = _rotate_frame_for_map(raw_frame, args.rotate_frame)

        filt.predict(dt)

        step = vo.step(frame, dt=dt, agl_m=args.agl_m, aircraft_mask=None)
        vo_steps += 1
        if step.valid:
            vo_valid += 1
            filt.update_of_velocity(
                np.array(step.velocity_body[:2]),
                step.yaw_rate,
                dt=dt,
                sigma_v_mps=args.sigma_v_mps,
                sigma_yaw_rate_radps=math.radians(2.0),
            )
        filt.update_altitude(0.0, sigma_h_m=30.0)

        map_accepted_this_frame = False
        map_reject_reason = ""
        meas_lat = float("nan")
        meas_lon = float("nan")
        meas_sigma = float("nan")
        meas_inliers = 0
        n_before_sem = 0
        n_after_sem = 0
        sat_class_share: dict[int, float] = {}
        struct_accepted_this_frame = False
        struct_reject_reason = ""
        struct_lat = float("nan")
        struct_lon = float("nan")
        struct_sigma = float("nan")
        struct_peak = float("nan")
        struct_margin = float("nan")

        if i % args.map_period_frames == 0:
            map_attempts += 1
            pred_lat, pred_lon, _ = filt.position_wgs84()
            crop_lat = frame_info.lat if args.oracle_map_crop else pred_lat
            crop_lon = frame_info.lon if args.oracle_map_crop else pred_lon
            map_patch, patch_bbox = _crop_map_patch(
                map_img,
                selected_map,
                crop_lat,
                crop_lon,
                args.map_crop_span_m,
                args.map_crop_size,
            )
            match_result = matcher.match(frame, map_patch, aircraft_mask=None)
            mk_d = match_result["mkpts0"]
            mk_m = match_result["mkpts1"]
            n_before_sem = int(len(mk_d))
            n_after_sem = n_before_sem
            sat_class_share: dict[int, float] = {}
            if seg is not None and args.semantic_mask and len(mk_d) > 0:
                sat_cmap = seg.stable_class_map(map_patch)
                # доли классов в маске для диагностики (видим что модель «видит»)
                vals, counts = np.unique(sat_cmap, return_counts=True)
                total = int(sat_cmap.size)
                sat_class_share = {int(v): float(c) / total for v, c in zip(vals, counts)}
                s_xy = mk_m.astype(np.int32)
                s_xy[:, 0] = np.clip(s_xy[:, 0], 0, sat_cmap.shape[1] - 1)
                s_xy[:, 1] = np.clip(s_xy[:, 1], 0, sat_cmap.shape[0] - 1)
                sat_keep = sat_cmap[s_xy[:, 1], s_xy[:, 0]] > 0
                mk_d = mk_d[sat_keep]
                mk_m = mk_m[sat_keep]
                n_after_sem = int(len(mk_d))
            meas = compute_map_measurement(
                frame_shape=frame.shape,
                mkpts_drone=mk_d,
                mkpts_map=mk_m,
                bbox=patch_bbox,
                map_shape=map_patch.shape,
                sigma_floor_m=args.sigma_map_floor_m,
                sigma_ceiling_m=args.sigma_map_ceiling_m,
            )
            map_reject_reason = meas.reject_reason
            meas_inliers = meas.num_inliers
            # Pending appearance measurement (commit в EKF после cross-channel gate)
            app_pending: tuple[float, float, float] | None = None
            if meas.accepted:
                meas_lat = meas.lat
                meas_lon = meas.lon
                meas_sigma = meas.sigma_xy_m
                app_pending = (meas.lat, meas.lon, meas.sigma_xy_m)

            # --- Structural channel: SegFormer на обеих сторонах + mask-to-mask NCC ---
            struct_pending: tuple[float, float, float] | None = None
            if args.semantic_structural_match and seg is not None:
                struct_attempts += 1
                drone_class_map = seg.stable_class_map(frame)
                if "sat_cmap" not in locals() or sat_cmap is None:
                    sat_cmap = seg.stable_class_map(map_patch)
                fh, fw = frame.shape[:2]
                f_px = k[0, 0]
                drone_ground_w = args.agl_m * fw / f_px
                drone_mppx = drone_ground_w / fw
                sat_mppx = args.map_crop_span_m / args.map_crop_size
                crop_lat_centre = (patch_bbox[1] + patch_bbox[3]) / 2.0
                crop_lon_centre = (patch_bbox[0] + patch_bbox[2]) / 2.0
                sfix = mask_to_mask_match(
                    drone_class_map=drone_class_map,
                    sat_class_map=sat_cmap,
                    drone_mppx=drone_mppx,
                    sat_mppx=sat_mppx,
                    sat_centre_latlon=(crop_lat_centre, crop_lon_centre),
                    sigma_floor_m=args.sigma_map_floor_m,
                    sigma_ceiling_m=args.sigma_map_ceiling_m,
                )
                struct_reject_reason = sfix.reject_reason
                struct_peak = sfix.peak_score
                struct_margin = sfix.peak_margin
                if sfix.accepted:
                    struct_lat = sfix.lat
                    struct_lon = sfix.lon
                    struct_sigma = sfix.sigma_xy_m
                    struct_pending = (sfix.lat, sfix.lon, sfix.sigma_xy_m)

            # --- Cross-channel consistency gate (§2.4): если оба канала accept
            # но pose расходятся >3σ joint covariance → оба отбрасываются ---
            if app_pending is not None and struct_pending is not None:
                d_lat_m = (app_pending[0] - struct_pending[0]) * 111320.0
                cos_lat = math.cos(math.radians(app_pending[0]))
                d_lon_m = (app_pending[1] - struct_pending[1]) * 111320.0 * cos_lat
                disagree_m = math.hypot(d_lat_m, d_lon_m)
                joint_sigma_m = math.hypot(app_pending[2], struct_pending[2])
                if disagree_m > 3.0 * joint_sigma_m:
                    map_reject_reason = f"consistency_reject_{disagree_m:.0f}m"
                    struct_reject_reason = f"consistency_reject_{disagree_m:.0f}m"
                    app_pending = None
                    struct_pending = None

            # --- Commit в EKF (с независимым Mahalanobis gate каждый канал) ---
            if app_pending is not None:
                a_lat, a_lon, a_sigma = app_pending
                upd = filt.update_map_position_wgs84(a_lat, a_lon, sigma_xy_m=a_sigma)
                if upd.accepted:
                    map_accepted += 1
                    map_accepted_this_frame = True
                else:
                    map_reject_reason = f"ekf_gate_d2={upd.mahalanobis2:.1f}"
            if struct_pending is not None:
                s_lat, s_lon, s_sigma = struct_pending
                upd_s = filt.update_map_position_wgs84(s_lat, s_lon, sigma_xy_m=s_sigma)
                if upd_s.accepted:
                    struct_accepted += 1
                    struct_accepted_this_frame = True
                else:
                    struct_reject_reason = f"ekf_gate_d2={upd_s.mahalanobis2:.1f}"

        state_lat, state_lon, _ = filt.position_wgs84()
        x_e, y_n, _ = filt.position_enu()
        gt_x_e, gt_y_n, _ = bridge.wgs84_to_enu(frame_info.lat, frame_info.lon, 0.0)
        err_m = _dist_m(state_lat, state_lon, frame_info.lat, frame_info.lon)

        rows.append(
            {
                "idx": i,
                "rel_path": frame_info.rel_path,
                "t_raw": t_raw,
                "dt": dt,
                "lat": state_lat,
                "lon": state_lon,
                "gt_lat": frame_info.lat,
                "gt_lon": frame_info.lon,
                "x_e": x_e,
                "y_n": y_n,
                "gt_x_e": gt_x_e,
                "gt_y_n": gt_y_n,
                "error_m": err_m,
                "sigma_pos_m": filt.position_sigma_m(),
                "speed_mps": filt.speed(),
                "vo_valid": int(step.valid),
                "vo_inliers": step.num_inliers,
                "vo_reject_reason": step.reject_reason,
                "map_accepted": int(map_accepted_this_frame),
                "map_reject_reason": map_reject_reason,
                "map_meas_lat": meas_lat,
                "map_meas_lon": meas_lon,
                "map_meas_sigma_m": meas_sigma,
                "map_meas_inliers": meas_inliers,
                "map_crop_source": "gt" if args.oracle_map_crop else "ekf",
                "sem_matches_before": n_before_sem if seg is not None else None,
                "sem_matches_after": n_after_sem if seg is not None else None,
                "sat_class_share_water": sat_class_share.get(1, 0.0) if seg is not None else None,
                "sat_class_share_veg": sat_class_share.get(2, 0.0) if seg is not None else None,
                "sat_class_share_bld": sat_class_share.get(3, 0.0) if seg is not None else None,
                "sat_class_share_road": sat_class_share.get(4, 0.0) if seg is not None else None,
                "sat_class_share_bg": sat_class_share.get(0, 0.0) if seg is not None else None,
                "struct_accepted": int(struct_accepted_this_frame),
                "struct_reject_reason": struct_reject_reason,
                "struct_lat": struct_lat,
                "struct_lon": struct_lon,
                "struct_sigma_m": struct_sigma,
                "struct_peak": struct_peak,
                "struct_margin": struct_margin,
            }
        )

        if (i + 1) % 25 == 0 or i == len(frames) - 1:
            print(
                f"[aerialvl-full] {i + 1}/{len(frames)}  "
                f"err={err_m:.1f}м σ={filt.position_sigma_m():.1f}м "
                f"map={map_accepted}/{map_attempts}"
            )

    errors = np.array([row["error_m"] for row in rows], dtype=np.float64)
    state_csv = output / "state.csv"
    summary_json = output / "summary.json"
    summary_md = output / "summary.md"
    trajectory_png = output / "trajectory.png"
    _write_csv(state_csv, rows)
    _plot_trajectory(trajectory_png, rows, f"AerialVL {frames[0].sequence}")

    median_error_m = float(np.median(errors))
    p95_error_m = float(np.percentile(errors, 95))
    mean_error_m = float(np.mean(errors))
    final_error_m = float(errors[-1])
    track_pct = 100.0 * float(np.mean(errors <= args.track_threshold_m))
    passed = (
        p95_error_m <= args.pass_p95_threshold_m
        and track_pct >= args.pass_track_pct_threshold
    )

    payload: dict[str, Any] = {
        "passed": passed,
        "dataset_root": str(dataset_root),
        "manifest_dir": str(manifest_dir),
        "split": frames[0].split,
        "sequence": frames[0].sequence,
        "map": selected_map.rel_path,
        "frames": len(frames),
        "backend": matcher.backend,
        "backend_requested": args.backend,
        "oracle_map_crop": bool(args.oracle_map_crop),
        "agl_m": float(args.agl_m),
        "map_crop_span_m": float(args.map_crop_span_m),
        "median_error_m": median_error_m,
        "p95_error_m": p95_error_m,
        "mean_error_m": mean_error_m,
        "final_error_m": final_error_m,
        "track_pct": track_pct,
        "pass_p95_threshold_m": args.pass_p95_threshold_m,
        "pass_track_pct_threshold": args.pass_track_pct_threshold,
        "vo_valid_pct": 100.0 * vo_valid / max(vo_steps, 1),
        "map_attempts": map_attempts,
        "map_accepted": map_accepted,
        "map_accept_pct": 100.0 * map_accepted / max(map_attempts, 1),
        "struct_attempts": struct_attempts,
        "struct_accepted": struct_accepted,
        "struct_accept_pct": 100.0 * struct_accepted / max(struct_attempts, 1),
        "state_csv": str(state_csv),
        "summary_md": str(summary_md),
        "trajectory_png": str(trajectory_png),
        "notes": (
            "AerialVL-скрипт использует GT первого кадра как стартовую априорную точку. "
            "Камера задаётся приближённой моделью камеры-обскуры, потому что публичный VAL "
            "манифест не содержит внутренних параметров камеры."
        ),
    }
    summary_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_summary(summary_md, payload)
    print(f"[aerialvl-full] summary -> {summary_md}")
    print(f"[aerialvl-full] json    -> {summary_json}")


if __name__ == "__main__":
    main()
