"""End-to-end integration runner: EKF + OF VO + retriever + matcher.

Wires together every Stage 1-3 component into a single per-frame loop
that produces an actual *trajectory* on real GP010269.MP4 footage —
the demonstration that goes beyond per-frame fixes.

Per video frame:

    1. Compute dt since previous frame.
    2. ``predict``: EKF propagates state by dt.
    3. **OF VO**: Lucas-Kanade on the current frame vs previous; gives
       body-frame velocity + yaw rate. Fed to EKF as a velocity update
       (R inflated by scale_bias uncertainty per Stage 2.5).
    4. **Altitude prior**: soft anchor on z_u (no barometer / telemetry).
    5. **Heading prior**: only fires when ``|yaw_rate| < threshold``.

Every ``map_period`` frames we also run the full matcher pipeline:

    6. Retriever (DINOv2-B) -> top-1 z=14 tile centre.
    7. MapManager z=17 window (cached by tile_id).
    8. Undistort -> aircraft mask -> BEV at default pitch.
    9. NeuralMatcher (XFeat).
   10. ``compute_map_measurement`` -> ``MapMeasurement``.
   11. If accepted: ``StateFilter.update_map_position_wgs84``.

Outputs:
    results/full_pipeline/state.csv        per-frame filter state
    results/full_pipeline/trajectory.png   lat/lon plot with map fixes
    results/full_pipeline/scale_history.png  scale_bias convergence
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
from src.ekf import StateFilter
from src.frame_bridge import FrameBridge
from src.geo_segmentor import GeoSegmentor
from src.map_manager import MapManager
from src.map_measurement import compute_map_measurement
from src.neural_matching import NeuralMatcher
from src.obstruction_detector import ObstructionDetector
from src.optical_flow_vo import OpticalFlowVO
from src.retriever import Retriever
from src.snow_mask import combine_with_aircraft_mask, detect_snow_mask
from src.undistort import Undistorter
from src.video_processor import VideoProcessor


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, default=Path("data/videos/GP010269.MP4"))
    p.add_argument("--camera-config", type=Path, default=Path("configs/camera_gopro_hx.yaml"))
    p.add_argument("--anchors-dir", type=Path, default=Path("data/masks/anchors"))
    p.add_argument("--retriever-db", type=Path, default=Path("data/retrieval/db_z14"))
    p.add_argument("--output", type=Path, default=Path("results/full_pipeline"))
    p.add_argument("--start-s", type=float, default=45.0)
    p.add_argument("--end-s", type=float, default=90.0)
    p.add_argument("--start-lat", type=float, default=55.086025)
    p.add_argument("--start-lon", type=float, default=38.149033)
    p.add_argument("--start-alt-msl", type=float, default=750.0)
    p.add_argument("--ground-msl", type=float, default=130.0,
                   help="approx ground elevation at mission start (Kolomna)")
    p.add_argument("--dem", type=Path, default=Path("data/dem/srtm_n55_e038.tif"),
                   help="DEM GeoTIFF for per-frame AGL; falls back to --agl-m if unavailable")
    p.add_argument("--map-period-s", type=float, default=2.0)
    p.add_argument("--zoom", type=int, default=17)
    p.add_argument("--window-radius", type=int, default=2)
    p.add_argument("--pitch-deg", type=float, default=-30.0)
    p.add_argument("--agl-m", type=float, default=620.0)
    p.add_argument("--ground-span-m", type=float, default=400.0)
    p.add_argument("--bev-out-size", type=int, default=800)
    p.add_argument("--backend", type=str, default="xfeat")
    p.add_argument("--apply-clahe", action=argparse.BooleanOptionalAction, default=False,
                   help="Stage 3.6: CLAHE preprocessing in matcher. Off by default "
                        "(measured -3 accepted frames on cross-seasonal GP010269).")
    p.add_argument("--obstruction-detect", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--snow-mask", action=argparse.BooleanOptionalAction, default=True,
                   help="mask bright low-saturation snow in the drone BEV before matching")
    p.add_argument("--snow-v-threshold", type=float, default=0.80)
    p.add_argument("--snow-s-threshold", type=float, default=0.15)
    p.add_argument("--semantic-mask", action=argparse.BooleanOptionalAction, default=False,
                   help="filter matches by sat-side stable semantic classes")
    p.add_argument("--seg-config", type=Path,
                   default=Path("results/segformer_overture_b0_ade1400_nocw_best/segformer_overture_quick_cfg.py"))
    p.add_argument("--seg-checkpoint", type=Path,
                   default=Path("results/segformer_overture_b0_ade1400_nocw_best/best_mIoU_iter_932.pth"))
    p.add_argument("--vo-stride", type=int, default=2,
                   help="run OF VO every N video frames (cuts total runtime)")
    p.add_argument("--log-stride", type=int, default=2,
                   help="log filter state to CSV every N video frames")
    p.add_argument("--bootstrap-buffer", type=int, default=5,
                   help="collect N accepted measurements before EKF init "
                        "(handles cross-seasonal false-positive locks)")
    p.add_argument("--bootstrap-cluster-radius-m", type=float, default=2000.0,
                   help="measurements within this radius of each other "
                        "form one cluster; largest cluster's centroid "
                        "is used as the EKF init position")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"[run] video         : {args.video}")
    print(f"[run] window        : t=[{args.start_s}, {args.end_s}] s")
    print(f"[run] map period    : every {args.map_period_s} s")
    print()

    # ---- bring up components ---------------------------------------------
    proc = VideoProcessor(str(args.video),
                          default_geo=(args.start_lat, args.start_lon, args.start_alt_msl))
    src_fps = proc.info.fps

    bridge = FrameBridge(
        lat0=args.start_lat, lon0=args.start_lon, alt0_msl=args.ground_msl,
    )
    filt = StateFilter(bridge)
    filt.initialize_from_wgs84(
        args.start_lat, args.start_lon, args.start_alt_msl,
        yaw=0.0, sigma_pos_m=5_000.0, sigma_yaw_rad=math.pi,
    )
    # Velocity initial: don't seed at zero with tight σ. Start unknown
    # (σ=50 m/s) so the first OF measurement is taken seriously.
    filt.P[StateFilter.IDX_VX_E, StateFilter.IDX_VX_E] = 50.0 ** 2
    filt.P[StateFilter.IDX_VY_N, StateFilter.IDX_VY_N] = 50.0 ** 2

    undistorter = Undistorter.from_yaml(args.camera_config)
    mask_tracker = (
        AircraftMaskTracker.from_index(args.anchors_dir)
        if (args.anchors_dir / "index.json").exists() else None
    )
    bev = BevRectifier.build(
        K_rect=undistorter.K_rect, image_size=undistorter.image_size,
        pitch_deg=args.pitch_deg, agl_m=args.agl_m,
        ground_span_m=args.ground_span_m,
        out_size=(args.bev_out_size, args.bev_out_size),
    )
    matcher = NeuralMatcher(backend=args.backend, apply_clahe=args.apply_clahe)
    retr = Retriever(); retr.load_database(args.retriever_db)
    obstruction_detector = ObstructionDetector() if args.obstruction_detect else None
    dem_lookup = None
    dem_error = None
    if args.dem is not None and args.dem.exists():
        try:
            from src.dem_lookup import DemLookup
            dem_lookup = DemLookup(args.dem)
        except Exception as e:
            dem_error = e
    seg = None
    if args.semantic_mask:
        import torch
        seg_device = "mps" if torch.backends.mps.is_available() else "cpu"
        seg = GeoSegmentor(
            backend="mmseg",
            mmseg_config=str(args.seg_config),
            mmseg_checkpoint=str(args.seg_checkpoint),
            mmseg_device=seg_device,
            stable_class_ids={1, 2, 3, 4},
            top_crop_ratio=0.0,
        )
        if seg.backend != "mmseg":
            seg = None

    vo = OpticalFlowVO(
        K=undistorter.K, D=undistorter.D, image_size=undistorter.image_size,
        pitch_deg=args.pitch_deg,
    )

    print(f"[run] EKF initial pos σ : {filt.position_sigma_m():.0f} m")
    print(f"[run] retriever device  : {retr.device}")
    print(f"[run] matcher backend   : {matcher.backend}")
    print(f"[run] obstruction gate  : {'on' if obstruction_detector is not None else 'off'}")
    print(f"[run] snow mask         : {'on' if args.snow_mask else 'off'}")
    print(f"[run] semantic mask     : {'on' if seg is not None else 'off'}")
    print(f"[run] DEM AGL source    : {args.dem if dem_lookup is not None else 'constant --agl-m'}")
    if dem_error is not None:
        print(f"[run] DEM disabled      : {dem_error}")
    print()

    # ---- per-frame loop ---------------------------------------------------
    f0 = int(round(args.start_s * src_fps))
    f1 = int(round(args.end_s * src_fps))
    map_stride = max(1, int(round(args.map_period_s * src_fps)))

    mm_cache: dict[str, tuple] = {}
    sat_sem_cache: dict[str, np.ndarray] = {}

    def _ensure_window(tile_id: str, lat: float, lon: float):
        if tile_id in mm_cache:
            return mm_cache[tile_id]
        mm = MapManager(zoom=args.zoom, window_radius=args.window_radius,
                        closer_threshold=1)
        m_cv2, m_bbox = mm.initialize(lat, lon)
        mm_cache[tile_id] = (m_cv2, m_bbox)
        return mm_cache[tile_id]

    rows: list[dict] = []
    map_fixes: list[tuple[float, float, float]] = []   # (lat, lon, sigma_xy_m) accepted
    n_map_attempt = 0
    n_map_accept = 0
    n_vo_step = 0
    n_vo_valid = 0
    n_obstructed = 0
    n_map_skipped_obstruction = 0
    n_dem_valid = 0
    n_dem_fallback = 0

    # Cross-seasonal bootstrap buffer: until we've accumulated
    # ``bootstrap_buffer`` matcher-accepted measurements, we don't push
    # any of them into the EKF; instead we cluster them by spatial
    # proximity and use the largest cluster's centroid to initialise.
    # This filters the ~25 % retriever noise rate that was producing
    # false-positive locks at the western edge of the DB.
    bootstrap_buf: list[tuple[float, float, float]] = []
    bootstrap_done = False

    def _cluster_centroid(points: list[tuple[float, float, float]],
                          radius_m: float) -> tuple[float, float, list[int]]:
        """Find largest spatial cluster of (lat, lon, sigma) points.
        Returns (centroid_lat, centroid_lon, member_indices)."""
        n = len(points)
        best_count, best_indices = 0, []
        for i in range(n):
            members = []
            for j in range(n):
                d_lat_m = (points[j][0] - points[i][0]) * 111320.0
                d_lon_m = (points[j][1] - points[i][1]) * 111320.0 * \
                          math.cos(math.radians(points[i][0]))
                if math.hypot(d_lat_m, d_lon_m) <= radius_m:
                    members.append(j)
            if len(members) > best_count:
                best_count = len(members)
                best_indices = members
        if best_count == 0:
            return points[0][0], points[0][1], [0]
        cl = [points[i] for i in best_indices]
        return (
            float(np.mean([c[0] for c in cl])),
            float(np.mean([c[1] for c in cl])),
            best_indices,
        )

    last_t = None
    t_start = time.time()
    print(f"[run] processing frames {f0}..{f1} ({f1-f0} total, "
          f"VO stride {args.vo_stride}, map every {map_stride})")
    print()

    for fi in range(f0, f1 + 1):
        ts = fi / src_fps
        if last_t is None:
            dt = 1.0 / src_fps
        else:
            dt = ts - last_t

        frame = proc.extract_frame(fi)
        if frame is None:
            continue

        mask_raw = (
            mask_tracker.mask_for_frame(fi, frame.shape[:2])
            if mask_tracker else None
        )
        obstr_result = (
            obstruction_detector.detect(frame, aircraft_mask=mask_raw)
            if obstruction_detector is not None else None
        )
        obstructed = bool(obstr_result.is_obstructed) if obstr_result is not None else False
        raw_obstructed = bool(obstr_result.raw_obstructed) if obstr_result is not None else False
        if obstructed:
            n_obstructed += 1
        if obstr_result is not None:
            obstr_metrics = obstr_result.metrics
            obstr_std = float(obstr_metrics.std)
            obstr_entropy = float(obstr_metrics.entropy)
            obstr_edge_density = float(obstr_metrics.edge_density)
            obstr_low_var_patch_frac = float(obstr_metrics.low_var_patch_frac)
            obstr_votes = int(obstr_metrics.votes)
        else:
            obstr_std = float("nan")
            obstr_entropy = float("nan")
            obstr_edge_density = float("nan")
            obstr_low_var_patch_frac = float("nan")
            obstr_votes = 0

        # 2. EKF predict.
        filt.predict(dt)

        state_lat_for_agl, state_lon_for_agl, _ = filt.position_wgs84()
        agl_m = args.agl_m
        dem_agl = None
        if dem_lookup is not None:
            dem_agl = dem_lookup.height_agl(
                state_lat_for_agl, state_lon_for_agl, args.start_alt_msl,
            )
            if dem_agl is not None and math.isfinite(dem_agl) and dem_agl > 1.0:
                agl_m = float(dem_agl)
                n_dem_valid += 1
            else:
                n_dem_fallback += 1

        # 3. OF VO (every vo_stride frames).
        if (fi - f0) % args.vo_stride == 0:
            step = vo.step(frame, dt=dt * args.vo_stride,
                           agl_m=agl_m, aircraft_mask=mask_raw)
            n_vo_step += 1
            if step.valid:
                n_vo_valid += 1
                filt.update_of_velocity(
                    np.array(step.velocity_body[:2]), step.yaw_rate,
                    dt=dt * args.vo_stride,
                    sigma_v_mps=2.0,
                    sigma_yaw_rate_radps=math.radians(1.0),
                )

        # 4. Altitude prior.
        filt.update_altitude(args.start_alt_msl, sigma_h_m=80.0)

        # 5. Heading prior (gated on |yaw_rate| internally).
        filt.maybe_update_heading_prior(sigma_heading_rad=math.radians(20.0))

        # 6-11. Map measurement (every map_stride frames).
        accepted_this_frame = False
        if (fi - f0) % map_stride == 0:
            n_map_attempt += 1
            if obstructed:
                n_map_skipped_obstruction += 1
                print(f"  f={fi:>5} t={ts:5.1f}s  OBSTRUCTION skip map "
                      f"votes={obstr_votes} std={obstr_std:.1f} H={obstr_entropy:.1f}")
            else:
                top = retr.query_image(frame, top_k=1)[0]
                map_cv2, mbbox = _ensure_window(top["tile_id"], top["lat"], top["lon"])

                rect = undistorter.undistort_image(frame)
                bev_frame = bev
                if abs(agl_m - args.agl_m) > 1e-6:
                    bev_frame = BevRectifier.build(
                        K_rect=undistorter.K_rect,
                        image_size=undistorter.image_size,
                        pitch_deg=args.pitch_deg,
                        agl_m=agl_m,
                        ground_span_m=args.ground_span_m,
                        out_size=(args.bev_out_size, args.bev_out_size),
                    )
                if mask_tracker is not None:
                    mask_rect = undistorter.undistort_image(mask_raw)
                    mask_bev = bev_frame.warp_mask(mask_rect)
                else:
                    mask_bev = None
                frame_bev = bev_frame.warp(rect)
                if args.snow_mask:
                    snow = detect_snow_mask(
                        frame_bev,
                        v_threshold=args.snow_v_threshold,
                        s_threshold=args.snow_s_threshold,
                    )
                    mask_bev = combine_with_aircraft_mask(snow, mask_bev)
                res = matcher.match(frame_bev, map_cv2, aircraft_mask=mask_bev)
                mkpts_drone = res["mkpts0"]
                mkpts_map = res["mkpts1"]
                if seg is not None and len(mkpts_drone) > 0:
                    if top["tile_id"] not in sat_sem_cache:
                        sat_sem_cache[top["tile_id"]] = seg.stable_class_map(map_cv2)
                    sat_cmap = sat_sem_cache[top["tile_id"]]
                    s_xy = mkpts_map.astype(np.int32)
                    s_xy[:, 0] = np.clip(s_xy[:, 0], 0, sat_cmap.shape[1] - 1)
                    s_xy[:, 1] = np.clip(s_xy[:, 1], 0, sat_cmap.shape[0] - 1)
                    sat_keep = sat_cmap[s_xy[:, 1], s_xy[:, 0]] > 0
                    mkpts_drone = mkpts_drone[sat_keep]
                    mkpts_map = mkpts_map[sat_keep]
                meas = compute_map_measurement(
                    frame_shape=frame_bev.shape,
                    mkpts_drone=mkpts_drone, mkpts_map=mkpts_map,
                    bbox=mbbox, map_shape=map_cv2.shape,
                    last_homography_scale=None,
                )
                if meas.accepted:
                    if not bootstrap_done:
                        # Buffer until we have enough to cluster.
                        bootstrap_buf.append((meas.lat, meas.lon, meas.sigma_xy_m))
                        print(f"  f={fi:>5} t={ts:5.1f}s  bootstrap buffer "
                              f"[{len(bootstrap_buf)}/{args.bootstrap_buffer}]: "
                              f"({meas.lat:.4f},{meas.lon:.4f}) inl={meas.num_inliers}")
                        if len(bootstrap_buf) >= args.bootstrap_buffer:
                            c_lat, c_lon, members = _cluster_centroid(
                                bootstrap_buf, args.bootstrap_cluster_radius_m,
                            )
                            print(f"  [bootstrap] largest cluster "
                                  f"{len(members)}/{len(bootstrap_buf)} measurements; "
                                  f"centroid=({c_lat:.4f}, {c_lon:.4f}) — initialising EKF")
                            # Re-init EKF at cluster centroid with tight σ.
                            filt.initialize_from_wgs84(
                                c_lat, c_lon, args.start_alt_msl,
                                yaw=filt.x[StateFilter.IDX_YAW],
                                sigma_pos_m=200.0, sigma_yaw_rad=math.pi,
                            )
                            # Keep the velocity / scale_bias state we already
                            # learned during the bootstrap window.
                            # Now feed the cluster members into the EKF as a
                            # batch of measurements (each gets gated normally).
                            for (la, lo, sg) in [bootstrap_buf[i] for i in members]:
                                filt.update_map_position_wgs84(la, lo, sigma_xy_m=max(sg, 20.0))
                                map_fixes.append((la, lo, sg))
                                n_map_accept += 1
                            bootstrap_done = True
                    else:
                        upd = filt.update_map_position_wgs84(
                            meas.lat, meas.lon, sigma_xy_m=max(meas.sigma_xy_m, 10.0),
                        )
                        state_lat, state_lon, _ = filt.position_wgs84()
                        print(f"  f={fi:>5} t={ts:5.1f}s  meas=({meas.lat:.4f},{meas.lon:.4f}) "
                              f"σ={meas.sigma_xy_m:.0f}m inl={meas.num_inliers}  "
                              f"state→({state_lat:.4f},{state_lon:.4f}) σ_pos={filt.position_sigma_m():.0f}m  "
                              f"upd_acc={upd.accepted} d²={upd.mahalanobis2:.1f}")
                        if upd.accepted:
                            n_map_accept += 1
                            map_fixes.append((meas.lat, meas.lon, meas.sigma_xy_m))
                            accepted_this_frame = True

        # Log every log_stride frames.
        if (fi - f0) % args.log_stride == 0:
            lat, lon, alt = filt.position_wgs84()
            x_e, y_n, _ = filt.position_enu()
            rows.append({
                "frame_idx": fi, "t_sec": ts,
                "lat": lat, "lon": lon, "alt_msl": alt,
                "x_e": x_e, "y_n": y_n,
                "sigma_pos_m": filt.position_sigma_m(),
                "speed_mps": filt.speed(),
                "agl_m": agl_m,
                "dem_agl_m": dem_agl if dem_agl is not None else float("nan"),
                "heading_deg": filt.heading_deg(),
                "scale_bias": filt.scale_bias(),
                "scale_sigma": filt.scale_bias_sigma(),
                "map_accepted": int(accepted_this_frame),
                "obstructed": int(obstructed),
                "raw_obstructed": int(raw_obstructed),
                "obstr_votes": obstr_votes,
                "obstr_std": obstr_std,
                "obstr_entropy": obstr_entropy,
                "obstr_edge_density": obstr_edge_density,
                "obstr_low_var_patch_frac": obstr_low_var_patch_frac,
            })

        last_t = ts

    elapsed = time.time() - t_start
    print(f"[run] elapsed         : {elapsed:.1f}s")
    print(f"[run] frames logged   : {len(rows)}")
    print(f"[run] OF VO steps     : {n_vo_step}  valid {n_vo_valid}")
    print(f"[run] map attempts    : {n_map_attempt}  accepted {n_map_accept}  "
          f"({100*n_map_accept/max(1,n_map_attempt):.1f}%)")
    print(f"[run] obstruction     : frames {n_obstructed}  map skips {n_map_skipped_obstruction}")
    print(f"[run] DEM AGL         : valid {n_dem_valid}  fallback {n_dem_fallback}")
    print(f"[run] map windows     : {len(mm_cache)}")
    print(f"[run] final scale_bias: {filt.scale_bias():.3f} ± {filt.scale_bias_sigma():.3f}")
    print(f"[run] final pos σ     : {filt.position_sigma_m():.1f} m")

    # ---- CSV --------------------------------------------------------------
    csv_path = args.output / "state.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[run] CSV -> {csv_path}")

    # ---- plots ------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lats = np.array([r["lat"] for r in rows])
    lons = np.array([r["lon"] for r in rows])
    sigs = np.array([r["sigma_pos_m"] for r in rows])
    times = np.array([r["t_sec"] for r in rows])
    scales = np.array([r["scale_bias"] for r in rows])
    scale_sigs = np.array([r["scale_sigma"] for r in rows])

    fix_lats = np.array([f[0] for f in map_fixes]) if map_fixes else np.array([])
    fix_lons = np.array([f[1] for f in map_fixes]) if map_fixes else np.array([])

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.plot(lons, lats, color="tab:blue", lw=1.0, label="EKF trajectory", alpha=0.8)
    if len(fix_lats):
        ax.scatter(fix_lons, fix_lats, s=80, color="tab:red", marker="x",
                   label=f"map fixes ({len(map_fixes)})", zorder=5)
    ax.scatter(lons[0], lats[0], s=120, color="tab:green", marker="o",
               label="start", zorder=6)
    ax.scatter(lons[-1], lats[-1], s=120, color="tab:purple", marker="s",
               label="end", zorder=6)
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    ax.set_title(f"Trajectory  ({args.start_s:.0f}-{args.end_s:.0f}s, "
                 f"{n_map_accept} accepted fixes)")
    ax.legend(loc="best", fontsize=10); ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(args.output / "trajectory.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(times, sigs, color="tab:blue")
    axes[0].set_ylabel("σ_pos (m)"); axes[0].set_yscale("log")
    axes[0].grid(alpha=0.3)
    axes[0].set_title("Filter state evolution")

    axes[1].plot(times, scales, color="tab:purple", label="scale_bias")
    axes[1].fill_between(times, scales - scale_sigs, scales + scale_sigs,
                         alpha=0.2, color="tab:purple")
    axes[1].axhline(1.0, color="grey", ls="--", lw=0.6)
    axes[1].set_ylabel("scale_bias"); axes[1].grid(alpha=0.3)
    axes[1].legend()

    speeds = np.array([r["speed_mps"] for r in rows])
    axes[2].plot(times, speeds, color="tab:green")
    axes[2].set_ylabel("speed (m/s)"); axes[2].set_xlabel("time (s)")
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output / "scale_history.png", dpi=120)
    plt.close(fig)

    print(f"[run] plots -> {args.output}/")


if __name__ == "__main__":
    main()
