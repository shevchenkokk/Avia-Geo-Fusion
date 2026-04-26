"""End-to-end smoke test: prove the full perception stack acquires
locks on real GP010269.MP4 cruise frames once the retriever solves
bootstrap.

Pipeline per frame:

    raw frame
       ├─> retriever (DINOv2-B) -> top-1 z=14 tile -> approx (lat, lon)
       │
       └─> undistort -> aircraft mask -> BEV(pitch=-30) -> matcher
                                                              vs
                                                z=17 MapManager window
                                                centred on retriever fix
                                                              │
                                                              v
                                                  compute_map_measurement

We do NOT chain successive frames yet (no temporal trust, no EKF) —
that's what comes after this proves matching is feasible. We're
asking the simplest possible question: **does the matcher accept any
frame at all once the right tile is in the window?** If yes, every
following milestone (real §3.2 calibration, EKF on real video,
trajectory drift measurement) is unblocked. If no, we need §3.4
(XFeat/MatchAnything) before anything else.

Metrics logged per sample:
  * retriever top-1 cosine, lat/lon, tile_id
  * matcher: raw_matches, num_inliers, inlier_ratio, reproj_px, accepted
  * residual between retriever fix and matcher fix (km)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
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

from src.aircraft_mask import AircraftMaskTracker
from src.bev_rectifier import BevRectifier
from src.geo_segmentor import GeoSegmentor
from src.map_manager import MapManager
from src.map_measurement import compute_map_measurement
from src.neural_matching import NeuralMatcher
from src.retriever import Retriever
from src.snow_mask import combine_with_aircraft_mask, detect_snow_mask
from src.undistort import Undistorter
from src.video_processor import VideoProcessor


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6_371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, default=Path("data/videos/GP010269.MP4"))
    p.add_argument("--camera-config", type=Path, default=Path("configs/camera_gopro_hx.yaml"))
    p.add_argument("--anchors-dir", type=Path, default=Path("data/masks/anchors"))
    p.add_argument("--retriever-db", type=Path, default=Path("data/retrieval/db_z14"))
    p.add_argument("--output", type=Path, default=Path("results/end_to_end"))
    p.add_argument("--cruise-start-s", type=float, default=45.0)
    p.add_argument("--cruise-end-s", type=float, default=120.0)
    p.add_argument("--sample-period-s", type=float, default=2.0)
    p.add_argument("--zoom", type=int, default=17)
    p.add_argument("--window-radius", type=int, default=5,
                   help="±radius z=17 tiles around retriever fix (covers ~1.3 km)")
    p.add_argument("--pitch-deg", type=float, default=-30.0)
    p.add_argument("--agl-m", type=float, default=620.0)
    p.add_argument("--ground-span-m", type=float, default=400.0)
    p.add_argument("--bev-out-size", type=int, default=800)
    p.add_argument("--backend", type=str, default="auto")
    p.add_argument("--snow-mask", action=argparse.BooleanOptionalAction, default=False,
                   help="Stage 4-lite: mask out snow regions on BEV before matcher")
    p.add_argument("--snow-v-threshold", type=float, default=0.80)
    p.add_argument("--snow-s-threshold", type=float, default=0.15)
    p.add_argument("--retriever-top-k", type=int, default=1,
                   help=">1: try top-K retriever tiles per frame, accept the "
                        "best-scoring matcher result (handles retriever noise)")
    p.add_argument("--apply-clahe", action=argparse.BooleanOptionalAction, default=False,
                   help="Stage 3.6: CLAHE on grayscale before matcher. Empirical "
                        "result on cross-seasonal GP010269: -3 accepted frames "
                        "(5/23 -> 2/23), so OFF by default. Toggle for ablation.")
    p.add_argument("--semantic-mask", action=argparse.BooleanOptionalAction, default=False,
                   help="Stage 4: filter matches by semantic class — keep only "
                        "keypoints on buildings+roads (seasonally stable)")
    p.add_argument("--seg-config", type=Path,
                   default=Path("results/segformer_overture_b0_ade1400_nocw_best/segformer_overture_quick_cfg.py"))
    p.add_argument("--seg-checkpoint", type=Path,
                   default=Path("results/segformer_overture_b0_ade1400_nocw_best/best_mIoU_iter_932.pth"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if not args.video.exists():
        raise SystemExit(f"video not found: {args.video}")

    print(f"[run] video         : {args.video}")
    print(f"[run] retriever DB  : {args.retriever_db}")
    print(f"[run] cruise window : t=[{args.cruise_start_s}, {args.cruise_end_s}] s "
          f"sample={args.sample_period_s}s")
    print()

    # Components.
    retr = Retriever()
    retr.load_database(args.retriever_db)
    print(f"[run] retriever device : {retr.device}  N={len(retr.database)}")
    undistorter = Undistorter.from_yaml(args.camera_config)
    print(f"[run] K_rect[0,0]      : {undistorter.K_rect[0,0]:.2f}")
    mask_tracker = (
        AircraftMaskTracker.from_index(args.anchors_dir)
        if (args.anchors_dir / "index.json").exists() else None
    )
    if mask_tracker:
        print(f"[run] aircraft mask    : {mask_tracker.num_anchors()} anchors")
    bev = BevRectifier.build(
        K_rect=undistorter.K_rect, image_size=undistorter.image_size,
        pitch_deg=args.pitch_deg, agl_m=args.agl_m,
        ground_span_m=args.ground_span_m,
        out_size=(args.bev_out_size, args.bev_out_size),
    )
    print(f"[run] BEV pitch        : {args.pitch_deg}°")

    matcher = NeuralMatcher(backend=args.backend, apply_clahe=args.apply_clahe)
    proc = VideoProcessor(str(args.video),
                          default_geo=(55.086025, 38.149033, 750.0))

    # Stage 4 semantic mask: SegFormer-B0 fine-tuned on Overture-RU.
    # Stable IDs hand-restricted to {3,4} (buildings, roads) — vegetation
    # is in the schema's `stable` set but isn't seasonally stable for
    # our winter-vs-summer gap; water can't host keypoints anyway.
    seg = None
    if args.semantic_mask:
        import torch
        seg_device = "mps" if torch.backends.mps.is_available() else "cpu"
        # SegFormer-B0 fine-tuned on Overture-RU is reliable on satellite
        # imagery (its training domain). On winter BEV-warped drone frames
        # it suffers severe domain shift (predicts ~97 % background), so
        # we run the heuristic texture-based segmenter on the drone side
        # and the ML one on the satellite side. The semantic mask filter
        # then keeps only matches whose drone endpoint is in a textured
        # ground region AND whose sat endpoint is on a stable class.
        seg = GeoSegmentor(
            backend="mmseg",
            mmseg_config=str(args.seg_config),
            mmseg_checkpoint=str(args.seg_checkpoint),
            mmseg_device=seg_device,
            # Include water (1) — over agricultural/forest tiles the
            # model misclassifies dark forest as water, but those are
            # still seasonally stable surfaces; including water keeps
            # ~50 % of pixels eligible. Exclude bg (0) — that's the
            # "no-class" output and we want to drop those matches.
            stable_class_ids={1, 2, 3, 4},
            top_crop_ratio=0.0,
        )
        seg_drone = GeoSegmentor(backend="heuristic", top_crop_ratio=0.0)
        if seg.backend != "mmseg":
            print(f"[run] WARNING: semantic mask requested but mmseg "
                  f"failed to init; running WITHOUT class filter")
            seg = None
        else:
            print(f"[run] semantic mask  : ML(sat) + heuristic(drone)  "
                  f"sat stable={1, 2, 3, 4}")
    src_fps = proc.info.fps

    # MapManager cache keyed by retriever tile_id (z=14).
    mm_cache: dict[str, tuple] = {}

    def _ensure_window(tile_id: str, lat: float, lon: float):
        if tile_id in mm_cache:
            return mm_cache[tile_id]
        mm = MapManager(zoom=args.zoom, window_radius=args.window_radius,
                        closer_threshold=1)
        map_cv2, bbox = mm.initialize(lat, lon)
        mm_cache[tile_id] = (map_cv2, bbox)
        return mm_cache[tile_id]

    sample_frames = []
    t = args.cruise_start_s
    while t <= args.cruise_end_s:
        sample_frames.append(int(round(t * src_fps)))
        t += args.sample_period_s
    print(f"[run] frames to test   : {len(sample_frames)}")
    print()

    rows: list[dict] = []
    accepted_count = 0
    last_inliers_log: list[int] = []
    t_start = time.time()
    for idx, fi in enumerate(sample_frames):
        frame = proc.extract_frame(fi)
        if frame is None:
            continue

        # Retriever bootstrap (top-K with fusion if requested).
        tops = retr.query_image(frame, top_k=args.retriever_top_k)

        # Frame pipeline (one-shot, shared across all retriever candidates):
        # undistort -> mask -> BEV.
        rect = undistorter.undistort_image(frame)
        if mask_tracker is not None:
            mask_raw = mask_tracker.mask_for_frame(fi, frame.shape[:2])
            mask_rect = undistorter.undistort_image(mask_raw)
            mask_bev = bev.warp_mask(mask_rect)
        else:
            mask_bev = None
        frame_bev = bev.warp(rect)
        if args.snow_mask:
            snow = detect_snow_mask(
                frame_bev,
                v_threshold=args.snow_v_threshold,
                s_threshold=args.snow_s_threshold,
            )
            mask_bev = combine_with_aircraft_mask(snow, mask_bev)

        # Stage 4 final shape: ML segmentation only on satellite side.
        # The drone BEV in winter fools the heuristic SkySegmenter (snow
        # mistaken for sky → returns 0 % texture) and the ML segmenter
        # (97 % background, severe domain shift). Sat-side filtering
        # alone, combined with the existing aircraft+snow masks on the
        # drone side, gives the same defense story without the drone-
        # side false-zeros problem.
        drone_cmap = None  # explicitly disabled

        # Try each top-K candidate; keep the result with the highest
        # inlier count. If multiple tie, prefer the higher retriever
        # score (top-1 first).
        best_meas = None
        best_top = tops[0]
        best_match_ms = 0.0
        best_raw = 0
        for top in tops:
            map_cv2, bbox = _ensure_window(top["tile_id"], top["lat"], top["lon"])
            mt0 = time.time()
            r = matcher.match(frame_bev, map_cv2, aircraft_mask=mask_bev)
            m_ms = (time.time() - mt0) * 1000.0
            mkpts0_arr = r["mkpts0"]; mkpts1_arr = r["mkpts1"]

            # Stage 4 semantic filter (sat side only). Drop matches whose
            # satellite endpoint lies on the "no-class" background (open
            # field, parking, untextured terrain) — those are typically
            # cross-seasonal RANSAC noise. Drone side is left unfiltered
            # because the ML segmenter has severe domain shift on winter
            # BEV, and the aircraft+snow masks already do their job.
            if seg is not None and len(mkpts0_arr) > 0:
                sat_cmap = seg.stable_class_map(map_cv2)
                s_xy = mkpts1_arr.astype(np.int32)
                s_xy[:, 0] = np.clip(s_xy[:, 0], 0, sat_cmap.shape[1] - 1)
                s_xy[:, 1] = np.clip(s_xy[:, 1], 0, sat_cmap.shape[0] - 1)
                sat_keep = sat_cmap[s_xy[:, 1], s_xy[:, 0]] > 0
                mkpts0_arr = mkpts0_arr[sat_keep]
                mkpts1_arr = mkpts1_arr[sat_keep]

            n_kept = len(mkpts0_arr)
            n_raw = int(r.get("raw_matches", n_kept))
            m = compute_map_measurement(
                frame_shape=frame_bev.shape,
                mkpts_drone=mkpts0_arr, mkpts_map=mkpts1_arr,
                bbox=bbox, map_shape=map_cv2.shape,
                last_homography_scale=None,
            )
            if best_meas is None or m.num_inliers > best_meas.num_inliers:
                best_meas = m; best_top = top
                best_match_ms = m_ms; best_raw = n_raw
        meas = best_meas
        ret_lat = best_top["lat"]; ret_lon = best_top["lon"]
        ret_score = best_top["score"]; tile_id = best_top["tile_id"]
        match_ms = best_match_ms; raw_matches = best_raw
        # bbox/map_cv2 of the *winning* candidate (for logging only).
        map_cv2, bbox = _ensure_window(tile_id, ret_lat, ret_lon)
        mkpts0 = np.empty((0, 2)); mkpts1 = np.empty((0, 2))  # not needed downstream

        residual_km = float("nan")
        if meas.accepted:
            accepted_count += 1
            residual_km = _haversine_km(ret_lat, ret_lon, meas.lat, meas.lon)
            last_inliers_log.append(meas.num_inliers)
        rows.append({
            "frame_idx": fi,
            "t_sec": fi / src_fps,
            "ret_score": ret_score,
            "ret_lat": ret_lat,
            "ret_lon": ret_lon,
            "ret_tile": tile_id,
            "raw_matches": raw_matches,
            "num_inliers": meas.num_inliers,
            "inlier_ratio": meas.inlier_ratio,
            "reproj_px": meas.reprojection_error_px,
            "sigma_xy_m": meas.sigma_xy_m,
            "accepted": int(meas.accepted),
            "reject_reason": meas.reject_reason,
            "match_ms": round(match_ms, 1),
            "fix_lat": meas.lat,
            "fix_lon": meas.lon,
            "residual_ret_km": residual_km,
        })

        marker = "*" if meas.accepted else " "
        print(f"  {marker} f={fi:>6}  t={fi/src_fps:5.1f}s  retr={ret_score:.3f} "
              f"({ret_lat:.4f},{ret_lon:.4f})  raw={raw_matches:<3} "
              f"inl={meas.num_inliers:<3} ratio={meas.inlier_ratio:.2f} "
              f"reproj={meas.reprojection_error_px:.2f}  acc={int(meas.accepted)} "
              f"reject={meas.reject_reason or '-':<20s} match={match_ms:5.0f}ms")

    elapsed = time.time() - t_start
    print()
    print(f"[run] total runtime   : {elapsed:.1f}s")
    print(f"[run] mm windows used : {len(mm_cache)} (cached by retriever tile_id)")
    print(f"[run] frames tested   : {len(rows)}")
    print(f"[run] frames accepted : {accepted_count}  ({100*accepted_count/max(1,len(rows)):.1f}%)")
    if last_inliers_log:
        arr = np.array(last_inliers_log)
        print(f"[run] inliers (acc'd) : median {float(np.median(arr)):.0f}  "
              f"max {int(arr.max())}")

    # Reject-reason histogram.
    rej_counts: dict[str, int] = {}
    for r in rows:
        if not r["accepted"]:
            rej_counts[r["reject_reason"]] = rej_counts.get(r["reject_reason"], 0) + 1
    if rej_counts:
        print("[run] rejects:")
        for k, v in sorted(rej_counts.items(), key=lambda kv: -kv[1]):
            print(f"    {k:30s} {v}")

    # CSV + summary.
    csv_path = args.output / "smoke.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[run] CSV -> {csv_path}")

    summary = [
        f"frames tested     : {len(rows)}",
        f"frames accepted   : {accepted_count}",
        f"acceptance rate   : {100*accepted_count/max(1,len(rows)):.1f}%",
        f"map windows used  : {len(mm_cache)}",
        f"runtime           : {elapsed:.1f}s",
    ]
    (args.output / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
