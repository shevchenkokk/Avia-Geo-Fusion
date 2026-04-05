"""Stage 2.2 verification: prove the optical-flow VO recovers metric
velocity correctly under known motion, and produces sane numbers on a
real cruise segment of GP010269.MP4.

Three stages:

  1. **Geometry round-trip** — bypass LK entirely. Synthesize ground
     points in body frame, project to pixels via the inverse ray-cast,
     undistort + re-ray-cast, confirm the body coordinates round-trip
     to within ~mm. Then fit a rigid transform on a synthetic motion
     and confirm the recovered (Δt, Δyaw) matches truth.

  2. **Synthetic LK pair** — generate a pair of frames under known
     body motion using the actual fisheye projection, run the full
     VO step (LK + ray-cast + RANSAC fit), confirm the recovered
     velocity matches truth within 2-5 % (the §2.2 criterion).

  3. **Real cruise sanity** — pick a clean cruise segment in
     GP010269.MP4 (after frost clears, before maneuvers), run VO at
     the source frame rate, report median speed and yaw-rate stability.
     Without telemetry we can't enforce the "≤2 % drift" half of the
     criterion; this stage demonstrates that the magnitudes are in
     the expected fixed-wing range (50-100 m/s, |yaw rate| < 5°/s).

The verification PASSES when stages 1 and 2 meet their numeric tols.
Stage 3 is reported but advisory.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.optical_flow_vo import (
    OpticalFlowVO,
    camera_to_body_rotation,
    fisheye_undistort_points,
    fit_rigid_transform_2d,
    ray_cast_ground,
)


# ---------------------------------------------------------------------------
# Helpers — forward camera projection (body XYZ -> pixel) for tests
# ---------------------------------------------------------------------------

def project_body_to_pixel(
    pts_body: np.ndarray,
    R_c2b: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    """Inverse of ray_cast_ground + fisheye_undistort_points.

    pts_body : (N, 3) float64. Returns (N, 2) pixel coords.
    """
    R_b2c = R_c2b.T
    pts_cam = (R_b2c @ pts_body.T).T  # (N, 3) in camera frame
    # Drop points that are behind the camera (z<=0) — they have no
    # valid projection.
    valid = pts_cam[:, 2] > 1e-6
    pts_cam = pts_cam[valid]
    if len(pts_cam) == 0:
        return np.empty((0, 2), dtype=np.float64)
    pts_n = (pts_cam[:, :2] / pts_cam[:, 2:3]).reshape(-1, 1, 2)
    pts_px = cv2.fisheye.distortPoints(pts_n, K, D)
    return pts_px.reshape(-1, 2)


def load_camera(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    cfg = yaml.safe_load(path.read_text())
    K = np.array(cfg["K"], dtype=np.float64)
    D = np.array(cfg["D"], dtype=np.float64).reshape(-1, 1)
    w, h = cfg["image_size"]
    return K, D, (int(w), int(h))


# ---------------------------------------------------------------------------
# Stage 1 — geometry round-trip
# ---------------------------------------------------------------------------

def stage1_geometry(K, D, image_size, pitch_deg, agl, vx_truth, vy_truth, dt) -> dict:
    R_c2b = camera_to_body_rotation(pitch_deg)
    # Sample a grid of body-frame ground points within the camera footprint.
    # For pitch ~ -30, AGL ~ 500m, the camera sees ~hundred metres ahead and
    # tens-of-metres laterally. Pick a 10x6 grid in body XY at z=AGL.
    xs = np.linspace(50, 600, 10)            # forward distance, metres
    ys = np.linspace(-150, 150, 6)           # lateral, metres
    X, Y = np.meshgrid(xs, ys)
    pts_b_prev = np.column_stack([X.ravel(), Y.ravel(),
                                  np.full(X.size, agl)])
    # After dt seconds of motion (velocity (vx, vy)), the same physical
    # ground points are now at body-frame XY = prev - v*dt.
    pts_b_curr = pts_b_prev.copy()
    pts_b_curr[:, 0] -= vx_truth * dt
    pts_b_curr[:, 1] -= vy_truth * dt

    # Forward-project both to pixels.
    px_prev = project_body_to_pixel(pts_b_prev, R_c2b, K, D)
    px_curr = project_body_to_pixel(pts_b_curr, R_c2b, K, D)
    n = min(len(px_prev), len(px_curr))
    px_prev = px_prev[:n]
    px_curr = px_curr[:n]
    pts_b_prev = pts_b_prev[:n]
    pts_b_curr = pts_b_curr[:n]

    # Round-trip: pixels -> normalized -> body. Should match prev/curr.
    n_prev = fisheye_undistort_points(px_prev, K, D)
    n_curr = fisheye_undistort_points(px_curr, K, D)
    rec_prev_b, mask_p = ray_cast_ground(n_prev, R_c2b, agl)
    rec_curr_b, mask_c = ray_cast_ground(n_curr, R_c2b, agl)

    # Roundtrip error per-point (in metres, body XY).
    err_prev = np.linalg.norm(rec_prev_b[:, :2] - pts_b_prev[mask_p, :2], axis=1)
    err_curr = np.linalg.norm(rec_curr_b[:, :2] - pts_b_curr[mask_c, :2], axis=1)

    # Rigid fit on the fully-paired subset.
    both = mask_p & mask_c
    n_pair = int(both.sum())
    src = pts_b_prev[both, :2]
    dst = pts_b_curr[both, :2]
    theta, t, rms = fit_rigid_transform_2d(src, dst)
    # Aircraft motion = -theta yaw, -t translation in body XY.
    rec_v = -t / dt
    rec_yaw_rate = -theta / dt

    return {
        "n_pts": n,
        "n_paired": n_pair,
        "roundtrip_err_max_m": float(max(err_prev.max() if err_prev.size else 0,
                                         err_curr.max() if err_curr.size else 0)),
        "fit_rms_m": rms,
        "v_truth": (vx_truth, vy_truth),
        "v_recovered": (float(rec_v[0]), float(rec_v[1])),
        "yaw_rate_recovered_radps": float(rec_yaw_rate),
    }


# ---------------------------------------------------------------------------
# Stage 2 — synthetic LK pair
# ---------------------------------------------------------------------------

def stage2_synthetic_lk(K, D, image_size, pitch_deg, agl, vx_truth, vy_truth, dt) -> dict:
    """Render two synthetic frames under known body motion and run full VO.

    Texture: random uniform noise, smoothed, so cv2.goodFeaturesToTrack
    has plenty of corners to lock onto. We sample the SAME texture
    field at the projected positions of each ground point, so the pair
    is geometrically self-consistent.
    """
    R_c2b = camera_to_body_rotation(pitch_deg)
    w, h = image_size

    # Build a random ground-texture image: 1000x1000 m square in body XY,
    # at z = AGL. We then project it into the camera at both prev and
    # curr aircraft poses by computing per-pixel ground intersections.
    # That's heavy but exact. For verification we don't need real-time.
    rng = np.random.default_rng(seed=42)
    tex_size = 800   # m, square side, centred at (forward=300, lateral=0)
    tex_res = 1.0    # m per texture pixel
    tex_n = int(tex_size / tex_res)
    texture = rng.integers(0, 256, size=(tex_n, tex_n), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (5, 5), 0)
    tex_origin_x = 300 - tex_size / 2  # forward-min in body
    tex_origin_y = -tex_size / 2       # lateral-min

    def sample_ground(pos_aircraft_body):
        """Render the camera image when the aircraft is at pos_aircraft_body.

        Each output pixel: undistort -> body ray (with aircraft origin
        at pos_aircraft_body) -> intersect z=AGL plane -> sample texture.
        """
        # Build a uv grid of pixel centres.
        u = np.arange(w) + 0.5
        v = np.arange(h) + 0.5
        U, V = np.meshgrid(u, v)
        pix = np.column_stack([U.ravel(), V.ravel()]).astype(np.float64)
        n = fisheye_undistort_points(pix, K, D)
        rays_c = np.column_stack([n, np.ones(len(n))])
        rays_b = (R_c2b @ rays_c.T).T  # (N, 3)
        valid = rays_b[:, 2] > 1e-6
        t_param = (agl - 0.0) / np.where(valid, rays_b[:, 2], 1)
        # Aircraft is at body origin in its own frame; the ground-point
        # in WORLD frame coords (with aircraft as origin shifted by
        # pos_aircraft_body) — but for this test we operate purely in
        # body frame, so the aircraft offset just translates the texture.
        ground_xy = rays_b[:, :2] * t_param[:, None] + pos_aircraft_body[None, :2]
        tx = ((ground_xy[:, 0] - tex_origin_x) / tex_res).astype(np.int32)
        ty = ((ground_xy[:, 1] - tex_origin_y) / tex_res).astype(np.int32)
        ok = valid & (tx >= 0) & (tx < tex_n) & (ty >= 0) & (ty < tex_n)
        out = np.zeros(len(pix), dtype=np.uint8)
        out[ok] = texture[ty[ok], tx[ok]]
        img = out.reshape(h, w)
        return img

    pos_prev = np.array([0.0, 0.0])
    pos_curr = np.array([vx_truth * dt, vy_truth * dt])  # aircraft moved by v*dt

    img_prev = sample_ground(pos_prev)
    img_curr = sample_ground(pos_curr)

    # Run two VO steps: bootstrap then a real measurement.
    vo = OpticalFlowVO(K, D, image_size, pitch_deg=pitch_deg)
    vo.step(cv2.cvtColor(img_prev, cv2.COLOR_GRAY2BGR), dt=dt, agl_m=agl)
    step = vo.step(cv2.cvtColor(img_curr, cv2.COLOR_GRAY2BGR), dt=dt, agl_m=agl)

    if not step.valid:
        return {"valid": False, "reject": step.reject_reason,
                "num_features": step.num_features,
                "num_tracked": step.num_tracked,
                "num_inliers": step.num_inliers}

    err_x = step.velocity_body[0] - vx_truth
    err_y = step.velocity_body[1] - vy_truth
    truth_speed = math.hypot(vx_truth, vy_truth)
    err_rel = math.hypot(err_x, err_y) / max(truth_speed, 1e-6)

    return {
        "valid": True,
        "v_truth": (vx_truth, vy_truth),
        "v_recovered": tuple(step.velocity_body[:2]),
        "speed_truth": truth_speed,
        "speed_recovered": step.speed,
        "rel_err": err_rel,
        "yaw_rate_radps": step.yaw_rate,
        "num_inliers": step.num_inliers,
        "num_tracked": step.num_tracked,
        "rms_m": step.rms_m,
    }


# ---------------------------------------------------------------------------
# Stage 3 — real cruise sanity
# ---------------------------------------------------------------------------

def stage3_real_cruise(K, D, image_size, pitch_deg, agl_m, video_path, t0, t1) -> dict:
    from src.video_processor import VideoProcessor
    from src.aircraft_mask import AircraftMaskTracker

    proc = VideoProcessor(str(video_path), default_geo=(55.086025, 38.149033, 750.0))
    fps = proc.info.fps
    f0 = int(round(t0 * fps))
    f1 = int(round(t1 * fps))

    tracker = None
    anchors = Path("data/masks/anchors")
    if (anchors / "index.json").exists():
        tracker = AircraftMaskTracker.from_index(anchors)

    vo = OpticalFlowVO(K, D, image_size, pitch_deg=pitch_deg)
    speeds: list[float] = []
    yaws: list[float] = []
    rejects: dict[str, int] = {}

    prev_idx = None
    for fi in range(f0, f1):
        frame = proc.extract_frame(fi)
        if frame is None:
            continue
        mask = tracker.mask_for_frame(fi, frame.shape[:2]) if tracker else None
        dt = 1.0 / fps if prev_idx is None else (fi - prev_idx) / fps
        prev_idx = fi
        step = vo.step(frame, dt=dt, agl_m=agl_m, aircraft_mask=mask)
        if step.valid:
            speeds.append(step.speed)
            yaws.append(step.yaw_rate)
        else:
            rejects[step.reject_reason] = rejects.get(step.reject_reason, 0) + 1

    if not speeds:
        return {"valid_frames": 0, "rejects": rejects}

    speeds_arr = np.array(speeds)
    yaws_arr = np.array(yaws)
    return {
        "valid_frames": len(speeds),
        "rejects": rejects,
        "speed_median": float(np.median(speeds_arr)),
        "speed_p10": float(np.percentile(speeds_arr, 10)),
        "speed_p90": float(np.percentile(speeds_arr, 90)),
        "yaw_rate_median_degps": float(np.degrees(np.median(yaws_arr))),
        "yaw_rate_p90_degps": float(np.degrees(np.percentile(np.abs(yaws_arr), 90))),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera-config", type=Path, default=Path("configs/camera_gopro_hx.yaml"))
    p.add_argument("--video", type=Path, default=Path("data/videos/GP010269.MP4"))
    p.add_argument("--pitch-deg", type=float, default=-30.0,
                   help="default until Stage 3.2 calibrates the real value")
    p.add_argument("--agl-m", type=float, default=620.0,
                   help="cruise AGL = alt_msl 750 - typical Kolomna ground 130 = 620")
    p.add_argument("--vx", type=float, default=70.0,
                   help="synthetic forward velocity (m/s)")
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--dt", type=float, default=1.0 / 30.0)
    p.add_argument("--cruise-start", type=float, default=60.0,
                   help="real-cruise stage start (s) — after the frost clears")
    p.add_argument("--cruise-end", type=float, default=80.0)
    p.add_argument("--rel-tol", type=float, default=0.05,
                   help="stage-2 relative velocity error tolerance (5 %% by default)")
    args = p.parse_args()

    K, D, image_size = load_camera(args.camera_config)
    print(f"[verify] camera : K[0,0]={K[0,0]:.1f}  size={image_size}")
    print(f"[verify] pitch  : {args.pitch_deg}°   AGL={args.agl_m} m")
    print(f"[verify] truth  : v=({args.vx:.1f}, {args.vy:.1f}) m/s  dt={args.dt:.4f} s")
    print()

    # --- Stage 1 ---
    s1 = stage1_geometry(K, D, image_size, args.pitch_deg, args.agl_m,
                         args.vx, args.vy, args.dt)
    print("[stage1] geometry round-trip + analytic rigid fit")
    print(f"  pts paired         : {s1['n_paired']}/{s1['n_pts']}")
    print(f"  roundtrip max err  : {s1['roundtrip_err_max_m']:.4e} m")
    print(f"  rigid fit RMS      : {s1['fit_rms_m']:.4e} m")
    print(f"  velocity truth     : {s1['v_truth']}")
    print(f"  velocity recovered : ({s1['v_recovered'][0]:.4f}, {s1['v_recovered'][1]:.4f})")
    print(f"  yaw rate (rad/s)   : {s1['yaw_rate_recovered_radps']:.4e}")
    s1_ok = (
        s1["roundtrip_err_max_m"] < 1e-3
        and s1["fit_rms_m"] < 1e-6
        and abs(s1["v_recovered"][0] - args.vx) < 1e-3
        and abs(s1["v_recovered"][1] - args.vy) < 1e-3
    )
    print(f"  stage1 -> {'OK' if s1_ok else 'FAIL'}")
    print()

    # --- Stage 2 ---
    s2 = stage2_synthetic_lk(K, D, image_size, args.pitch_deg, args.agl_m,
                             args.vx, args.vy, args.dt)
    print("[stage2] synthetic LK pair through full VO step")
    if not s2["valid"]:
        print(f"  VO rejected: {s2.get('reject', '')}  "
              f"feat={s2.get('num_features', 0)} track={s2.get('num_tracked', 0)} "
              f"inl={s2.get('num_inliers', 0)}")
        s2_ok = False
    else:
        print(f"  v truth     : {s2['v_truth']}")
        print(f"  v recovered : ({s2['v_recovered'][0]:.3f}, {s2['v_recovered'][1]:.3f})  "
              f"|v|={s2['speed_recovered']:.3f}")
        print(f"  rel err     : {100 * s2['rel_err']:.2f} %% (tol {100 * args.rel_tol}%%)")
        print(f"  inliers     : {s2['num_inliers']}/{s2['num_tracked']}  RMS={s2['rms_m']:.3f} m")
        s2_ok = s2["rel_err"] <= args.rel_tol
    print(f"  stage2 -> {'OK' if s2_ok else 'FAIL'}")
    print()

    # --- Stage 3 ---
    if args.video.exists():
        s3 = stage3_real_cruise(K, D, image_size, args.pitch_deg, args.agl_m,
                                args.video, args.cruise_start, args.cruise_end)
        print("[stage3] real cruise sanity (advisory)")
        print(f"  segment        : t=[{args.cruise_start:.1f}, {args.cruise_end:.1f}] s")
        print(f"  valid frames   : {s3['valid_frames']}")
        print(f"  rejects        : {s3['rejects']}")
        if s3["valid_frames"]:
            print(f"  speed (m/s)    : median {s3['speed_median']:.1f}  "
                  f"p10 {s3['speed_p10']:.1f}  p90 {s3['speed_p90']:.1f}")
            print(f"  |yaw rate| p90 : {s3['yaw_rate_p90_degps']:.2f} °/s   "
                  f"(median {s3['yaw_rate_median_degps']:.2f})")
            in_range = 30.0 <= s3["speed_median"] <= 150.0
            print(f"  median speed in [30, 150] m/s -> {'plausible' if in_range else 'SUSPECT'}")
    else:
        print(f"[stage3] video {args.video} not found — skipping real cruise check")

    print()
    if s1_ok and s2_ok:
        print("[verify] CRITERION PASSED (stages 1 + 2)")
        return
    print("[verify] CRITERION NOT MET")
    sys.exit(2)


if __name__ == "__main__":
    main()
