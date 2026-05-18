"""Оркестратор этапа 1.1: покадровый диагностический CSV и сводные графики.

Запускает пайплайн локализации (matcher → geolocator) по видео, пишет одну
строку CSV на каждый обработанный кадр и строит два диагностических графика,
по которым быстро видно доминирующий режим отказа из шести, перечисленных в
PROJECT_PLAN.md §1.

Важно: это *диагностический* runner, а не рабочий пайплайн. Он намеренно
прогоняет каждый кадр через matcher без LK-ускорения, чтобы кривые
inlier/ratio/reproj отражали поведение матчера, а не протяжки по optical flow.
Поэтому он медленнее ``run_hud_video.py``; для быстрых итераций используйте
``--max-seconds`` и ``--fps``.

Выходы пишутся в ``results/diag/<run_name>/``:
    frames_<run>.csv             — полный покадровый trace
    summary_<run>.png            — 4-panel inliers/ratio/reproj/scale plot
    state_timeline_<run>.png     — TRACK/WEAK/RELOC + reject reasons over time
    summary_<run>.txt            — human-readable rollup
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aircraft_mask import AircraftMaskTracker, load_aircraft_mask_tracker_for_video
from src.diagnostic_logger import (
    DiagnosticLogger,
    FrameDiagnostic,
    classify_state,
    confidence_from_inliers,
)
from src.geolocator import Geolocator
from src.map_manager import MapManager
from src.neural_matching import NeuralMatcher
from src.obstruction_detector import ObstructionDetector
from src.video_processor import VideoProcessor
from src.video_discovery import pick_default_video


def _tile_id(map_manager: MapManager) -> str:
    if map_manager.center_tx is None or map_manager.center_ty is None:
        return ""
    return f"{map_manager.zoom}/{map_manager.center_tx}/{map_manager.center_ty}"


def _count_keypoints(img: np.ndarray, max_kp: int = 4096) -> int:
    """Дешёвая прокси-метрика плотности для ``num_keypoints_*``: число ORB.

    Сам matcher не отдаёт отдельное число keypoints кадра, сравнимое между
    backend'ами LightGlue/LoFTR/ORB, поэтому используем лёгкий ORB-детектор.
    Он даёт для диагностического графика сигнал *плотности текстуры* — это
    помогает понять, низкое число inlier-точек вызвано нехваткой текстуры или
    сбоем матчера.
    """
    if img is None:
        return -1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    orb = cv2.ORB_create(nfeatures=max_kp, fastThreshold=15)
    kp = orb.detect(gray, None)
    return len(kp)


def run(args: argparse.Namespace) -> None:
    video_path = Path(args.video) if args.video else pick_default_video()
    if video_path is None or not Path(video_path).exists():
        raise SystemExit(f"video not found: {video_path}")

    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[diag] video      = {video_path}")
    print(f"[diag] output_dir = {out_dir}")

    processor = VideoProcessor(
        str(video_path), default_geo=(args.start_lat, args.start_lon, args.start_alt)
    )
    src_fps = processor.info.fps
    total_frames_video = processor.info.frame_count

    pos = processor.telemetry.get_position_at_frame(0.0)
    start_lat = float(getattr(pos, "latitude", args.start_lat))
    start_lon = float(getattr(pos, "longitude", args.start_lon))
    print(f"[diag] start_pos  = ({start_lat:.6f}, {start_lon:.6f})")

    map_manager = MapManager(zoom=args.zoom, window_radius=args.window_radius, closer_threshold=1)
    map_cv2, bbox = map_manager.initialize(start_lat, start_lon)

    matcher = NeuralMatcher(backend=args.backend)
    geolocator = Geolocator(bbox=bbox, map_shape=map_cv2.shape)
    obstruction_detector: ObstructionDetector | None = None
    if args.obstruction_detect:
        obstruction_detector = ObstructionDetector()
        print(f"[diag] obstruction detector: enabled (std<{obstruction_detector.std_threshold} "
              f"H<{obstruction_detector.entropy_threshold} "
              f"edge<{obstruction_detector.edge_density_threshold} "
              f"pf>{obstruction_detector.low_var_patch_frac_threshold})")

    mask_tracker: AircraftMaskTracker | None = None
    if args.anchors_dir is not None:
        anchors_dir = Path(args.anchors_dir)
        if (anchors_dir / "index.json").exists():
            mask_tracker = load_aircraft_mask_tracker_for_video(anchors_dir, video_path)
            if mask_tracker is not None:
                print(f"[diag] aircraft mask tracker: {mask_tracker.num_anchors()} anchors from {anchors_dir}")
        else:
            print(f"[diag] WARNING: --anchors-dir set but {anchors_dir/'index.json'} missing; "
                  f"running WITHOUT aircraft mask")

    sample_step = max(1, int(round(src_fps / args.fps)))
    start_frame = int(round(args.start_seconds * src_fps))
    if args.max_seconds is not None and args.max_seconds > 0:
        end_frame = min(total_frames_video, start_frame + int(args.max_seconds * src_fps))
    else:
        end_frame = total_frames_video
    iter_count = max(0, (end_frame - start_frame) // sample_step)
    print(
        f"[diag] src_fps={src_fps:.2f}  diag_fps={args.fps}  "
        f"sample_step={sample_step}  frames={start_frame}-{end_frame}  iters={iter_count}"
    )

    logger_csv = DiagnosticLogger(output_dir=out_dir, run_name=run_name)

    debug_frames_dir = out_dir / "debug_frames"
    if args.save_debug_every > 0:
        debug_frames_dir.mkdir(exist_ok=True)

    cur_tile_id = _tile_id(map_manager)
    iters_done = 0

    pbar = tqdm(total=iter_count, desc="diagnostic")
    cur = start_frame
    while cur < end_frame:
        frame = processor.extract_frame(cur)
        if frame is None:
            break
        ts = cur / max(src_fps, 1e-6)

        t_kp0 = time.perf_counter()
        n_kp_frame = _count_keypoints(frame)
        n_kp_tile = _count_keypoints(map_cv2)
        kp_ms = (time.perf_counter() - t_kp0) * 1000.0

        aircraft_mask = None
        if mask_tracker is not None:
            aircraft_mask = mask_tracker.mask_for_frame(cur, frame.shape[:2], frame=frame)

        # Ворота этапа 1.5: детектируем облако / блик / размытие до матчера.
        # Если кадр перекрыт, полностью пропускаем matcher: на почти однородном
        # кадре он придумывает совпадения, которые портят состояние Калмана.
        # bbox продолжает дрейфовать через predict_only().
        obstr_result = None
        if obstruction_detector is not None:
            obstr_result = obstruction_detector.detect(frame, aircraft_mask=aircraft_mask)

        skipped_for_obstruction = obstr_result is not None and obstr_result.is_obstructed

        if skipped_for_obstruction:
            match_result = {"mkpts0": np.empty((0, 2), dtype=np.float32),
                            "mkpts1": np.empty((0, 2), dtype=np.float32),
                            "raw_matches": 0, "backend": "skipped:obstruction"}
            match_ms = 0.0
        else:
            t_match0 = time.perf_counter()
            match_result = matcher.match(frame, map_cv2, aircraft_mask=aircraft_mask)
            match_ms = (time.perf_counter() - t_match0) * 1000.0

        mkpts_drone = match_result["mkpts0"]
        mkpts_map = match_result["mkpts1"]
        raw_matches = int(match_result.get("raw_matches", len(mkpts_drone)))
        backend_name = str(match_result.get("backend", ""))

        if skipped_for_obstruction:
            gps = None
            geo_ms = 0.0
            geolocator.last_diagnostics = {"reject_reason": "obstruction"}
        else:
            t_geo0 = time.perf_counter()
            gps = geolocator.estimate_center_gps(
                frame.shape, mkpts_drone, mkpts_map, is_keyframe=False
            )
            geo_ms = (time.perf_counter() - t_geo0) * 1000.0

        d = dict(geolocator.last_diagnostics)
        n_inliers = int(d.get("num_inliers", 0))
        ratio = float(d.get("inlier_ratio", float("nan")))
        reproj = float(d.get("reproj_error_px", float("nan")))
        scale = float(d.get("homography_scale", float("nan")))
        reject = str(d.get("reject_reason", ""))
        state = classify_state(n_inliers, ratio if np.isfinite(ratio) else 0.0, reject)
        confidence = confidence_from_inliers(n_inliers, ratio if np.isfinite(ratio) else 0.0, reproj)

        new_map_cv2, new_bbox = (map_cv2, bbox)
        tile_changed = False
        # Этап 1.3: каждый кадр отдаём MapManager лучшую доступную позицию.
        # Если текущий кадр дал валидный lock, используем его. Иначе берём
        # прогноз Калмана по счислению пути, чтобы bbox дрейфовал вдоль ожидаемой
        # траектории в окнах потери трека, а не замирал на последнем lock.
        if gps is not None:
            update_lat, update_lon = gps[0], gps[1]
        else:
            pred = geolocator.predict_only()
            if pred is not None:
                update_lat, update_lon = pred
            else:
                update_lat = update_lon = None
        if update_lat is not None and update_lon is not None:
            new_map_cv2, new_bbox = map_manager.update(update_lat, update_lon)
            if new_bbox != bbox:
                map_cv2 = new_map_cv2
                bbox = new_bbox
                geolocator.update_bbox(new_bbox, map_cv2.shape)
                cur_tile_id = _tile_id(map_manager)
                tile_changed = True

        if obstr_result is not None:
            obstr_metrics = obstr_result.metrics
            obstr_fields = dict(
                obstructed=bool(obstr_result.is_obstructed),
                obstr_std=float(obstr_metrics.std),
                obstr_entropy=float(obstr_metrics.entropy),
                obstr_edge_density=float(obstr_metrics.edge_density),
                obstr_low_var_patch_frac=float(obstr_metrics.low_var_patch_frac),
                obstr_votes=int(obstr_metrics.votes),
            )
        else:
            obstr_fields = {}

        diag = FrameDiagnostic(
            frame_idx=cur,
            timestamp_seconds=ts,
            wallclock_ms=kp_ms + match_ms + geo_ms,
            backend=backend_name,
            num_keypoints_frame=n_kp_frame,
            num_keypoints_tile=n_kp_tile,
            num_matches=raw_matches,
            num_inliers=n_inliers,
            inlier_ratio=ratio,
            reprojection_error_px=reproj,
            homography_scale=scale,
            confidence=confidence,
            state=state,
            reject_reason=reject,
            raw_lat=float(d.get("raw_lat", float("nan"))),
            raw_lon=float(d.get("raw_lon", float("nan"))),
            kalman_lat=float(d.get("kalman_lat", float("nan"))),
            kalman_lon=float(d.get("kalman_lon", float("nan"))),
            predicted_lat=float(d.get("predicted_lat", float("nan"))),
            predicted_lon=float(d.get("predicted_lon", float("nan"))),
            tile_id=cur_tile_id,
            tile_changed=tile_changed,
            is_keyframe=False,
            matcher_ms=match_ms,
            geolocator_ms=geo_ms,
            **obstr_fields,
        )
        logger_csv.log_frame(diag)

        if args.save_debug_every > 0 and (iters_done % args.save_debug_every == 0):
            debug_img = match_result.get("debug_img")
            if debug_img is not None:
                cv2.putText(
                    debug_img,
                    f"f={cur} t={ts:6.1f}s state={state} inl={n_inliers}/{raw_matches} "
                    f"ratio={ratio:.2f} reproj={reproj:.1f}px reject={reject or '-'}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA
                )
                cv2.putText(
                    debug_img,
                    f"f={cur} t={ts:6.1f}s state={state} inl={n_inliers}/{raw_matches} "
                    f"ratio={ratio:.2f} reproj={reproj:.1f}px reject={reject or '-'}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA
                )
                cv2.imwrite(str(debug_frames_dir / f"f{cur:08d}.jpg"), debug_img)

        iters_done += 1
        cur += sample_step
        pbar.update(1)
    pbar.close()

    summary = logger_csv.close()
    summary["video"] = str(video_path)
    summary["start_lat"] = start_lat
    summary["start_lon"] = start_lon
    summary["sample_step"] = sample_step
    summary["src_fps"] = src_fps
    summary["diag_fps"] = args.fps

    summary_txt = out_dir / f"summary_{run_name}.txt"
    lines = [
        f"run            : {run_name}",
        f"video          : {video_path}",
        f"start_pos      : ({start_lat:.6f}, {start_lon:.6f})",
        f"src_fps        : {src_fps:.3f}",
        f"diag_fps       : {args.fps}",
        f"frames logged  : {summary.get('frames', 0)}",
        "",
        f"TRACK pct      : {summary.get('track_pct', 0):.1f}%",
        f"WEAK pct       : {summary.get('weak_pct', 0):.1f}%",
        f"RELOCALIZE pct : {summary.get('relocalize_pct', 0):.1f}%",
        f"median inliers : {summary.get('median_inliers', 0):.1f}",
        f"median ratio   : {summary.get('median_inlier_ratio', 0):.3f}",
        f"median reproj  : {summary.get('median_reproj_px', 0):.2f} px",
        f"tile changes   : {summary.get('tile_changes', 0)}",
        "",
        "reject_reason counts:",
    ]
    for r, c in sorted(summary.get("reject_counts", {}).items(), key=lambda kv: -kv[1]):
        lines.append(f"  {r:30s} {c}")
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / f"summary_{run_name}.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print()
    print("\n".join(lines))
    print(f"\n[diag] CSV  -> {summary.get('csv')}")
    print(f"[diag] plot -> {summary.get('plot')}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, default=None,
                   help="path to video; default = video_discovery.pick_default_video()")
    p.add_argument("--output-dir", type=Path, default=Path("results/diag"))
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--start-seconds", type=float, default=0.0)
    p.add_argument("--max-seconds", type=float, default=None,
                   help="optional cap on diagnostic duration; None = whole video")
    p.add_argument("--fps", type=float, default=5.0,
                   help="diagnostic sampling rate (Hz). 5 Hz over 11 min ≈ 3300 rows")
    p.add_argument("--zoom", type=int, default=17)
    p.add_argument("--window-radius", type=int, default=3)
    p.add_argument("--backend", type=str, default="auto")
    p.add_argument("--start-lat", type=float, default=55.086025)
    p.add_argument("--start-lon", type=float, default=38.149033)
    p.add_argument("--start-alt", type=float, default=750.0)
    p.add_argument("--save-debug-every", type=int, default=50,
                   help="save matcher debug image every N samples; 0 disables")
    p.add_argument("--anchors-dir", type=Path, default=Path("data/masks/anchors"),
                   help="seed-mask anchors directory (Stage 0b output). "
                        "Set to '' to disable aircraft mask filtering.")
    p.add_argument("--obstruction-detect", action=argparse.BooleanOptionalAction, default=True,
                   help="Stage 1.5: gate the matcher on cloud/blur/glare detection. "
                        "Disable with --no-obstruction-detect.")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
