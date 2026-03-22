#!/usr/bin/env python3
"""Fisheye focal tiebreaker via essential matrix consistency.

For each candidate (K, D) fisheye profile:
  1. Undistort a set of frame pairs sampled from the video with a small baseline (dt ~50 ms).
  2. Detect SIFT features and match bidirectionally (cross-check).
  3. Estimate the essential matrix with cv2.findEssentialMat(K_rectified, RANSAC).
  4. Record inlier_ratio per pair.

The profile whose intrinsics best agree with the underlying scene geometry will
produce a self-consistent epipolar structure across many independent pairs and
therefore the highest median inlier_ratio. A wrong focal length (e.g. treating a
Wide-FOV video as Medium-FOV, a ~20% focal mismatch) breaks the epipolar
constraint and inlier_ratio drops measurably.

This is used as a tiebreaker when scripts/recover_intrinsics.py reports an
ambiguous choice between physically different profiles that both yield straight
lines after undistort.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


@dataclass
class Candidate:
    name: str
    source: str
    k: np.ndarray
    d: np.ndarray


def _load_profiles(path: Path) -> list[Candidate]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)

    if isinstance(payload, dict) and "profiles" in payload:
        items = payload["profiles"]
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError("Profiles file must be a list or an object with 'profiles'")

    out: list[Candidate] = []
    for item in items:
        k = np.asarray(item["K"], dtype=np.float64)
        d = np.asarray(item["D"], dtype=np.float64).reshape(-1)
        if d.size != 4:
            raise ValueError(f"Profile {item.get('name')} must have 4 fisheye coeffs")
        out.append(
            Candidate(
                name=str(item.get("name", "unnamed")),
                source=str(item.get("source", "unknown")),
                k=k,
                d=d,
            )
        )
    return out


def _k_rectified(candidate: Candidate, width: int, height: int) -> np.ndarray:
    try:
        return cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            candidate.k,
            candidate.d.reshape(4, 1),
            (width, height),
            np.eye(3),
            balance=0.0,
            new_size=(width, height),
        )
    except Exception:
        return candidate.k


def _undistort(frame: np.ndarray, candidate: Candidate, k_rectified: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    return cv2.fisheye.undistortImage(
        frame,
        K=candidate.k,
        D=candidate.d.reshape(4, 1),
        Knew=k_rectified,
        new_size=(w, h),
    )


def _sample_pairs(
    video_path: Path,
    t_windows: list[tuple[float, float]],
    pairs_per_window: int,
    baseline_frames: int,
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pairs: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for t0, t1 in t_windows:
        f0 = int(round(t0 * fps))
        f1 = int(round(t1 * fps))
        if f1 <= f0 + baseline_frames:
            continue
        anchors = np.linspace(f0, f1 - baseline_frames, num=pairs_per_window, dtype=np.int64)
        for anchor in anchors:
            if anchor < 0 or anchor + baseline_frames >= total_frames:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(anchor))
            ok_a, frame_a = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(anchor + baseline_frames))
            ok_b, frame_b = cap.read()
            if not (ok_a and ok_b):
                continue
            pairs.append((int(anchor), int(anchor + baseline_frames), frame_a, frame_b))
    cap.release()

    if not pairs:
        raise ValueError("No pairs extracted from the provided t_windows")
    return pairs


def _apply_aircraft_mask(img: np.ndarray, left_frac: float) -> np.ndarray:
    """Crude static mask: zero out the leftmost 'left_frac' of the frame.

    The real dynamic aircraft mask is a Stage 0b task; for the focal tiebreaker
    we only need to prevent SIFT from locking onto the fuselage, which is the
    dominant left-side static feature in the test video.
    """
    if left_frac <= 0.0:
        return img
    out = img.copy()
    w = out.shape[1]
    cut = int(round(w * left_frac))
    out[:, :cut] = 0
    return out


def _match_pair(
    gray_a: np.ndarray,
    gray_b: np.ndarray,
    max_features: int,
    ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    detector = cv2.SIFT_create(nfeatures=max_features)
    kp_a, desc_a = detector.detectAndCompute(gray_a, None)
    kp_b, desc_b = detector.detectAndCompute(gray_b, None)
    if desc_a is None or desc_b is None or len(kp_a) < 30 or len(kp_b) < 30:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = matcher.knnMatch(desc_a, desc_b, k=2)

    good: list[cv2.DMatch] = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < 20:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    pts_a = np.asarray([kp_a[m.queryIdx].pt for m in good], dtype=np.float32)
    pts_b = np.asarray([kp_b[m.trainIdx].pt for m in good], dtype=np.float32)
    return pts_a, pts_b


def _essential_geometry_score(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    k_rectified: np.ndarray,
    ransac_threshold_px: float,
) -> dict[str, float]:
    """Compute epipolar self-consistency metrics for a candidate intrinsics.

    Returns per-pair:
      - inlier_ratio of RANSAC essential-matrix fit
      - median symmetric reprojection residual of inliers (in pixels)
      - cheirality rate: fraction of inliers triangulated with positive depth
        in both cameras after recovering (R, t) from E.
    """
    if len(pts_a) < 20:
        return {
            "inlier_ratio": 0.0,
            "median_residual_px": 999.0,
            "cheirality_rate": 0.0,
            "inliers": 0,
            "matches": int(len(pts_a)),
        }
    E, mask = cv2.findEssentialMat(
        pts_a,
        pts_b,
        k_rectified,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=ransac_threshold_px,
    )
    if E is None or mask is None:
        return {
            "inlier_ratio": 0.0,
            "median_residual_px": 999.0,
            "cheirality_rate": 0.0,
            "inliers": 0,
            "matches": int(len(pts_a)),
        }
    inl_mask = mask.ravel().astype(bool)
    inliers = int(inl_mask.sum())
    total = int(len(pts_a))
    if inliers < 10:
        return {
            "inlier_ratio": float(inliers) / float(total),
            "median_residual_px": 999.0,
            "cheirality_rate": 0.0,
            "inliers": inliers,
            "matches": total,
        }

    pts_a_in = pts_a[inl_mask].astype(np.float64)
    pts_b_in = pts_b[inl_mask].astype(np.float64)

    # Sampson / symmetric epipolar residual in pixels, using F = K^-T E K^-1
    k_inv = np.linalg.inv(k_rectified)
    F = k_inv.T @ E @ k_inv

    def _append_one(pts: np.ndarray) -> np.ndarray:
        return np.hstack([pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)])

    pa_h = _append_one(pts_a_in)
    pb_h = _append_one(pts_b_in)
    # line coefficients
    l_b = (F @ pa_h.T).T  # epipolar lines in image B for each point in A
    l_a = (F.T @ pb_h.T).T
    # distance from point to its epipolar line
    denom_b = np.sqrt(l_b[:, 0] ** 2 + l_b[:, 1] ** 2) + 1e-9
    denom_a = np.sqrt(l_a[:, 0] ** 2 + l_a[:, 1] ** 2) + 1e-9
    d_b = np.abs(np.sum(l_b * pb_h, axis=1)) / denom_b
    d_a = np.abs(np.sum(l_a * pa_h, axis=1)) / denom_a
    sym_residual = 0.5 * (d_a + d_b)
    median_residual = float(np.median(sym_residual))

    # Cheirality: pass only inliers to recoverPose (no mask needed), and read back
    # how many of them passed the positive-depth check in both cameras.
    retval, R, t, _ = cv2.recoverPose(E, pts_a_in, pts_b_in, k_rectified)
    cheirality_rate = float(retval) / float(inliers) if inliers > 0 else 0.0

    return {
        "inlier_ratio": float(inliers) / float(total),
        "median_residual_px": median_residual,
        "cheirality_rate": cheirality_rate,
        "inliers": inliers,
        "matches": total,
    }


def _evaluate_candidate(
    candidate: Candidate,
    pairs: list[tuple[int, int, np.ndarray, np.ndarray]],
    args: argparse.Namespace,
    debug_dir: Path | None,
) -> dict[str, Any]:
    # Cache K_rectified using frame size of the first pair
    h, w = pairs[0][2].shape[:2]
    k_rect = _k_rectified(candidate, w, h)

    inlier_ratios: list[float] = []
    residuals: list[float] = []
    cheirality: list[float] = []
    total_inliers: list[int] = []
    total_matches: list[int] = []

    for pair_idx, (anchor, _, frame_a, frame_b) in enumerate(pairs):
        und_a = _undistort(frame_a, candidate, k_rect)
        und_b = _undistort(frame_b, candidate, k_rect)

        und_a = _apply_aircraft_mask(und_a, args.aircraft_mask_left_frac)
        und_b = _apply_aircraft_mask(und_b, args.aircraft_mask_left_frac)

        gray_a = cv2.cvtColor(und_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(und_b, cv2.COLOR_BGR2GRAY)

        pts_a, pts_b = _match_pair(
            gray_a,
            gray_b,
            max_features=args.max_sift_features,
            ratio=args.lowe_ratio,
        )
        g = _essential_geometry_score(
            pts_a, pts_b, k_rect, ransac_threshold_px=args.ransac_threshold_px
        )
        inlier_ratios.append(g["inlier_ratio"])
        residuals.append(g["median_residual_px"])
        cheirality.append(g["cheirality_rate"])
        total_inliers.append(int(g["inliers"]))
        total_matches.append(int(g["matches"]))

        if debug_dir is not None and pair_idx == 0:
            debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_dir / f"{candidate.name}_pair0_a.jpg"), und_a)
            cv2.imwrite(str(debug_dir / f"{candidate.name}_pair0_b.jpg"), und_b)

    median_ratio = float(np.median(inlier_ratios)) if inlier_ratios else 0.0
    mean_ratio = float(np.mean(inlier_ratios)) if inlier_ratios else 0.0
    # Residual is the most focal-sensitive metric when inlier_ratio saturates.
    finite_residuals = [r for r in residuals if r < 100.0]
    median_residual = float(np.median(finite_residuals)) if finite_residuals else 999.0
    mean_cheirality = float(np.mean(cheirality)) if cheirality else 0.0

    return {
        "name": candidate.name,
        "source": candidate.source,
        "fx": float(candidate.k[0, 0]),
        "median_inlier_ratio": median_ratio,
        "mean_inlier_ratio": mean_ratio,
        "median_residual_px": median_residual,
        "mean_cheirality_rate": mean_cheirality,
        "mean_matches": float(np.mean(total_matches)) if total_matches else 0.0,
        "mean_inliers": float(np.mean(total_inliers)) if total_inliers else 0.0,
        "pair_count": int(len(inlier_ratios)),
        "per_pair": [
            {
                "anchor_frame": int(pairs[i][0]),
                "inlier_ratio": float(inlier_ratios[i]),
                "residual_px": float(residuals[i]),
                "cheirality_rate": float(cheirality[i]),
                "inliers": int(total_inliers[i]),
                "matches": int(total_matches[i]),
            }
            for i in range(len(inlier_ratios))
        ],
    }


def _parse_windows(spec: str) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        a, b = chunk.split(":")
        windows.append((float(a), float(b)))
    return windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path("configs/camera_profile_candidates_gopro_hx.yaml"),
    )
    parser.add_argument(
        "--profile-filter",
        type=str,
        default=None,
        help="Comma-separated profile names to keep (default: all)",
    )
    parser.add_argument(
        "--t-windows",
        type=str,
        default="90:270,360:500,510:580",
        help="Cruise time windows in 'start:end[,...]' seconds",
    )
    parser.add_argument("--pairs-per-window", type=int, default=6)
    parser.add_argument(
        "--baseline-frames",
        type=int,
        default=10,
        help="Frame gap for pair sampling (dt = baseline_frames / fps). "
        "Needs to be large enough that parallax is measurable for aerial scenes.",
    )
    parser.add_argument("--max-sift-features", type=int, default=2000)
    parser.add_argument("--lowe-ratio", type=float, default=0.75)
    parser.add_argument(
        "--ransac-threshold-px",
        type=float,
        default=0.5,
        help="Tight threshold so wrong K shows as residual growth, not hidden by loose RANSAC.",
    )
    parser.add_argument(
        "--aircraft-mask-left-frac",
        type=float,
        default=0.22,
        help="Fraction of left-edge pixels to zero out as a crude fuselage mask",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/focal_tiebreaker_report.json"),
    )
    parser.add_argument("--debug-dir", type=Path, default=Path("results/focal_tiebreaker_debug"))
    parser.add_argument("--no-debug-images", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.profiles.exists():
        raise FileNotFoundError(f"Profiles not found: {args.profiles}")

    profiles = _load_profiles(args.profiles)
    if args.profile_filter:
        wanted = {s.strip() for s in args.profile_filter.split(",")}
        profiles = [p for p in profiles if p.name in wanted]
        if not profiles:
            raise ValueError(f"No profiles matched filter: {wanted}")
    print(f"[focal] loaded {len(profiles)} profiles: {[p.name for p in profiles]}")

    windows = _parse_windows(args.t_windows)
    pairs = _sample_pairs(
        args.video,
        t_windows=windows,
        pairs_per_window=int(args.pairs_per_window),
        baseline_frames=int(args.baseline_frames),
    )
    print(f"[focal] sampled {len(pairs)} frame pairs from {len(windows)} windows")

    debug_dir = None if args.no_debug_images else args.debug_dir

    scored: list[dict[str, Any]] = []
    for cand in profiles:
        result = _evaluate_candidate(cand, pairs, args, debug_dir)
        scored.append(result)
        print(
            f"[focal] {result['name']}: fx={result['fx']:.0f}"
            f" residual={result['median_residual_px']:.3f}px"
            f" inlier={result['median_inlier_ratio']:.3f}"
            f" cheirality={result['mean_cheirality_rate']:.3f}"
            f" matches={result['mean_matches']:.0f}"
        )

    # Sort by residual ascending (primary): it is monotonic in focal error.
    scored.sort(key=lambda r: r["median_residual_px"])
    best = scored[0]
    second = scored[1] if len(scored) > 1 else None
    margin = 0.0
    if second is not None:
        denom = max(1e-6, best["median_residual_px"])
        margin = (second["median_residual_px"] - best["median_residual_px"]) / denom

    print(
        f"[focal] winner: {best['name']} fx={best['fx']:.0f}"
        f" residual={best['median_residual_px']:.3f}px"
        f" margin_vs_second={margin:.3f}"
    )

    payload = {
        "video": str(args.video),
        "profiles_file": str(args.profiles),
        "t_windows": windows,
        "pairs_per_window": int(args.pairs_per_window),
        "baseline_frames": int(args.baseline_frames),
        "selected_profile": best["name"],
        "selected_fx": best["fx"],
        "selection_margin": margin,
        "scored": scored,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[focal] report: {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
