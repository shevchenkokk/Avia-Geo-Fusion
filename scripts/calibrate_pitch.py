"""Этап 3.2: калибровка pitch через перебор сетки.

Для каждого кандидата pitch запускается **тот же покадровый пайплайн, что и
в ``end_to_end_smoke``**: retriever выбирает top-1 тайл z=14, MapManager
открывает вокруг него небольшое окно z=17, а матчер сопоставляет
BEV-выпрямленный кадр с этим окном. По всем выбранным кадрам агрегируются
метрики качества матчера; побеждает pitch с максимальной медианой
``num_inliers``.

Почему через retriever: исходный перебор на мозаике 5×5 км без retriever давал
0/N принятых измерений при любом pitch. Это исправлено в §3.5: retriever решает
bootstrap-проблему, которую прежняя калибровка пыталась обойти широким окном.

Выход:
    results/stage3_2/pitch_grid.csv   метрики по каждой паре (pitch, frame)
    results/stage3_2/pitch_summary.txt
    results/stage3_2/best_pitch.png   визуально: BEV при лучшем pitch рядом
                                       со спутниковым тайлом среднего кадра
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import certifi  # type: ignore
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except Exception:
    pass

from src.aircraft_mask import load_aircraft_mask_tracker_for_video
from src.bev_rectifier import BevRectifier
from src.map_manager import MapManager
from src.map_measurement import compute_map_measurement
from src.neural_matching import NeuralMatcher
from src.retriever import Retriever
from src.undistort import Undistorter
from src.video_processor import VideoProcessor


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, default=Path("data/videos/GP010269.MP4"))
    p.add_argument("--camera-config", type=Path,
                   default=Path("configs/camera_gopro_hx.yaml"))
    p.add_argument("--anchors-dir", type=Path, default=Path("data/masks/anchors"))
    p.add_argument("--retriever-db", type=Path, default=Path("data/retrieval/db_z14"))
    p.add_argument("--output", type=Path, default=Path("results/stage3_2"))
    p.add_argument("--zoom", type=int, default=17)
    p.add_argument("--window-radius", type=int, default=2,
                   help="±radius z=17 tiles around retriever fix (matches "
                        "end_to_end_smoke working config)")
    p.add_argument("--cruise-start-s", type=float, default=45.0,
                   help="start of clean-cruise window (after frost clears)")
    p.add_argument("--cruise-end-s", type=float, default=89.0,
                   help="end of clean-cruise window (memory says 21.7% acceptance "
                        "for the working config in 45-89s)")
    p.add_argument("--sample-period-s", type=float, default=2.0)
    p.add_argument("--pitches-deg", type=float, nargs="+",
                   default=[-50.0, -40.0, -30.0, -20.0, -10.0],
                   help="grid (negative = camera below horizontal)")
    p.add_argument("--agl-m", type=float, default=620.0,
                   help="cruise alt 750 - typical Kolomna ground 130")
    p.add_argument("--ground-span-m", type=float, default=400.0)
    p.add_argument("--bev-out-size", type=int, default=800)
    p.add_argument("--backend", type=str, default="xfeat")
    p.add_argument("--apply-clahe", action=argparse.BooleanOptionalAction, default=False)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if not args.video.exists():
        raise SystemExit(f"video not found: {args.video}")
    if not args.retriever_db.with_suffix(".npz").exists():
        raise SystemExit(
            f"retriever DB not found at {args.retriever_db}.npz — "
            f"run scripts/build_retrieval_db.py first"
        )

    print(f"[calib] video         : {args.video}")
    print(f"[calib] camera config : {args.camera_config}")
    print(f"[calib] cruise window : t=[{args.cruise_start_s}, {args.cruise_end_s}] s "
          f"sample={args.sample_period_s}s")
    print(f"[calib] pitch grid    : {args.pitches_deg}")
    print(f"[calib] window_radius : {args.window_radius} (retriever-based)")

    # Компоненты пайплайна.
    undistorter = Undistorter.from_yaml(args.camera_config)
    print(f"[calib] K_rect[0,0]   : {undistorter.K_rect[0,0]:.2f}")

    mask_tracker = load_aircraft_mask_tracker_for_video(args.anchors_dir, args.video)
    if mask_tracker is not None:
        print(f"[calib] anchors       : {mask_tracker.num_anchors()}")

    matcher = NeuralMatcher(backend=args.backend, apply_clahe=args.apply_clahe)
    retr = Retriever(); retr.load_database(args.retriever_db)
    print(f"[calib] retriever     : device={retr.device}  N={len(retr.database)}")

    proc = VideoProcessor(str(args.video),
                          default_geo=(55.086025, 38.149033, 750.0))
    src_fps = proc.info.fps

    # Кэш MapManager по retriever tile_id: тайл z=14 -> окно z=17.
    mm_cache: dict[str, tuple] = {}

    def _ensure_window(tile_id: str, lat: float, lon: float):
        if tile_id in mm_cache:
            return mm_cache[tile_id]
        mm = MapManager(zoom=args.zoom, window_radius=args.window_radius,
                        closer_threshold=1)
        m_cv2, bbox = mm.initialize(lat, lon)
        mm_cache[tile_id] = (m_cv2, bbox)
        return mm_cache[tile_id]

    sample_frames: list[int] = []
    cur_t = args.cruise_start_s
    while cur_t <= args.cruise_end_s:
        sample_frames.append(int(round(cur_t * src_fps)))
        cur_t += args.sample_period_s

    print(f"[calib] sample frames : {len(sample_frames)}  "
          f"({sample_frames[0]}..{sample_frames[-1]})")
    print(f"[calib] total matcher calls: {len(args.pitches_deg) * len(sample_frames)}")
    print()

    rows: list[dict] = []
    per_pitch: dict[float, list[int]] = {p: [] for p in args.pitches_deg}

    for pitch in args.pitches_deg:
        bev = BevRectifier.build(
            K_rect=undistorter.K_rect,
            image_size=undistorter.image_size,
            pitch_deg=pitch,
            agl_m=args.agl_m,
            ground_span_m=args.ground_span_m,
            out_size=(args.bev_out_size, args.bev_out_size),
        )
        for fi in sample_frames:
            frame_raw = proc.extract_frame(fi)
            if frame_raw is None:
                continue

            # Retriever выбирает тайл.
            top = retr.query_image(frame_raw, top_k=1)[0]
            map_cv2, bbox = _ensure_window(top["tile_id"], top["lat"], top["lon"])

            # Стандартный покадровый пайплайн, как в end_to_end_smoke.
            mask_raw = (
                mask_tracker.mask_for_frame(fi, frame_raw.shape[:2], frame=frame_raw)
                if mask_tracker is not None else None
            )
            frame_rect = undistorter.undistort_image(frame_raw)
            mask_rect = (
                undistorter.undistort_image(mask_raw)
                if mask_raw is not None else None
            )
            frame_bev = bev.warp(frame_rect)
            mask_bev = bev.warp_mask(mask_rect) if mask_rect is not None else None

            res = matcher.match(frame_bev, map_cv2, aircraft_mask=mask_bev)
            mkpts0 = res["mkpts0"]
            mkpts1 = res["mkpts1"]
            raw_matches = int(res.get("raw_matches", len(mkpts0)))

            meas = compute_map_measurement(
                frame_shape=frame_bev.shape,
                mkpts_drone=mkpts0,
                mkpts_map=mkpts1,
                bbox=bbox,
                map_shape=map_cv2.shape,
                last_homography_scale=None,
            )
            rows.append({
                "pitch_deg": pitch,
                "frame_idx": fi,
                "t_sec": fi / src_fps,
                "ret_score": top["score"],
                "ret_tile": top["tile_id"],
                "raw_matches": raw_matches,
                "num_inliers": meas.num_inliers,
                "inlier_ratio": meas.inlier_ratio,
                "reproj_px": meas.reprojection_error_px,
                "accepted": int(meas.accepted),
                "reject_reason": meas.reject_reason,
            })
            per_pitch[pitch].append(meas.num_inliers)
            print(f"  pitch={pitch:+5.1f}°  f={fi:>6}  t={fi/src_fps:5.1f}s  "
                  f"retr={top['score']:.3f} ({top['tile_id']})  "
                  f"raw={raw_matches:<4} inl={meas.num_inliers:<3} "
                  f"ratio={meas.inlier_ratio:.2f} reproj={meas.reprojection_error_px:.2f} "
                  f"acc={int(meas.accepted)}")
        med = float(np.median(per_pitch[pitch])) if per_pitch[pitch] else 0.0
        mean = float(np.mean(per_pitch[pitch])) if per_pitch[pitch] else 0.0
        accept_n = sum(1 for r in rows if r["pitch_deg"] == pitch and r["accepted"])
        print(f"  pitch={pitch:+5.1f}°  -> median inliers {med:.1f}  "
              f"mean {mean:.1f}  accepted {accept_n}/{len(per_pitch[pitch])}")
        print()

    csv_path = args.output / "pitch_grid.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[calib] CSV -> {csv_path}")

    summary_lines = ["pitch_deg, median_inliers, mean_inliers, accept_rate"]
    pitch_scores = []
    for pitch in args.pitches_deg:
        med = float(np.median(per_pitch[pitch])) if per_pitch[pitch] else 0.0
        mean = float(np.mean(per_pitch[pitch])) if per_pitch[pitch] else 0.0
        accept_n = sum(1 for r in rows if r["pitch_deg"] == pitch and r["accepted"])
        accept_rate = accept_n / max(1, len(per_pitch[pitch]))
        pitch_scores.append((pitch, med, mean, accept_rate))
        summary_lines.append(f"{pitch:+.1f}, {med:.1f}, {mean:.1f}, {accept_rate:.2f}")

    # Оценка = (accept_rate, median_inliers): доля принятия важнее, потому что
    # число inlier-точек ниже порога §1.4 бесполезно в работе, даже если оно
    # максимальное в сетке.
    best_pitch, best_med, best_mean, best_acc = max(
        pitch_scores, key=lambda x: (x[3], x[1])
    )
    summary_lines.append("")
    summary_lines.append(f"recommended pitch_deg = {best_pitch:+.1f}  "
                         f"(median inliers {best_med:.1f}, accept_rate {best_acc:.2f})")
    summary_lines.append(f"map windows used     = {len(mm_cache)} "
                         f"(retriever tiles encountered)")
    summary_txt = "\n".join(summary_lines)
    (args.output / "pitch_summary.txt").write_text(summary_txt + "\n", encoding="utf-8")
    print()
    print(summary_txt)

    # Визуализация: BEV при лучшем pitch + выбранный retriever спутниковый тайл.
    mid_idx = sample_frames[len(sample_frames) // 2]
    frame_raw = proc.extract_frame(mid_idx)
    if frame_raw is not None:
        bev = BevRectifier.build(
            K_rect=undistorter.K_rect, image_size=undistorter.image_size,
            pitch_deg=best_pitch, agl_m=args.agl_m,
            ground_span_m=args.ground_span_m,
            out_size=(args.bev_out_size, args.bev_out_size),
        )
        rect = undistorter.undistort_image(frame_raw)
        bev_img = bev.warp(rect)
        top = retr.query_image(frame_raw, top_k=1)[0]
        map_cv2, _ = _ensure_window(top["tile_id"], top["lat"], top["lon"])
        sat_small = cv2.resize(map_cv2, (args.bev_out_size, args.bev_out_size),
                               interpolation=cv2.INTER_AREA)

        def _label(img, txt):
            o = img.copy()
            cv2.putText(o, txt, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(o, txt, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 1, cv2.LINE_AA)
            return o

        sheet = np.hstack([
            _label(bev_img, f"BEV pitch={best_pitch:+.1f}  f={mid_idx}"),
            _label(sat_small, f"satellite (retriever top-1, {top['tile_id']})"),
        ])
        cv2.imwrite(str(args.output / "best_pitch.png"), sheet)
        print(f"\n[calib] visual -> {args.output / 'best_pitch.png'}")


if __name__ == "__main__":
    main()
