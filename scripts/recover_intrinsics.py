#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


@dataclass
class CandidateProfile:
    name: str
    source: str
    k: np.ndarray
    d: np.ndarray


@dataclass
class FrameMetric:
    score: float
    mean_rmse: float
    line_count: int
    support_points: int
    fallback: bool = False


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _point_line_rmse(xs: np.ndarray, ys: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> float:
    dx = x2 - x1
    dy = y2 - y1
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return 1e9
    dist = np.abs(dy * xs - dx * ys + x2 * y1 - y2 * x1) / norm
    if dist.size > 20:
        keep = max(8, int(round(dist.size * 0.7)))
        dist = np.partition(dist, keep - 1)[:keep]
    return float(np.sqrt(np.mean(np.square(dist))))


def _prepare_gray(frame_bgr: np.ndarray, max_frame_size: int) -> np.ndarray:
    if frame_bgr.ndim == 3:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame_bgr

    h, w = gray.shape[:2]
    if max(h, w) > max_frame_size:
        scale = max_frame_size / float(max(h, w))
        gray = cv2.resize(
            gray,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )

    valid = gray > 3
    if np.any(valid):
        ys, xs = np.where(valid)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        roi_w = x1 - x0 + 1
        roi_h = y1 - y0 + 1
        if roi_w * roi_h > int(0.35 * gray.size):
            pad_x = max(2, int(round(0.02 * roi_w)))
            pad_y = max(2, int(round(0.02 * roi_h)))
            x0 = max(0, x0 - pad_x)
            x1 = min(gray.shape[1] - 1, x1 + pad_x)
            y0 = max(0, y0 - pad_y)
            y1 = min(gray.shape[0] - 1, y1 + pad_y)
            gray = gray[y0 : y1 + 1, x0 : x1 + 1]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _estimate_straightness(frame_bgr: np.ndarray, max_frame_size: int) -> FrameMetric:
    gray = _prepare_gray(frame_bgr, max_frame_size=max_frame_size)
    edges = cv2.Canny(gray, 60, 160, apertureSize=3, L2gradient=True)

    h, w = edges.shape[:2]
    min_len = max(40, int(round(0.08 * max(h, w))))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=60,
        minLineLength=min_len,
        maxLineGap=10,
    )
    if lines is None or len(lines) == 0:
        return FrameMetric(score=0.0, mean_rmse=999.0, line_count=0, support_points=0, fallback=True)

    ranked: list[tuple[float, int, int, int, int]] = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = (int(line[0]), int(line[1]), int(line[2]), int(line[3]))
        length = float(math.hypot(x2 - x1, y2 - y1))
        ranked.append((length, x1, y1, x2, y2))

    ranked.sort(key=lambda item: item[0], reverse=True)
    ranked = ranked[:200]

    total_score = 0.0
    total_support = 0
    rmses: list[float] = []
    used_lines = 0

    edge_mask = edges > 0
    for length, x1, y1, x2, y2 in ranked:
        stripe = np.zeros_like(edges)
        cv2.line(stripe, (x1, y1), (x2, y2), 255, thickness=5)
        ys, xs = np.where(edge_mask & (stripe > 0))
        if xs.size < 12:
            continue

        rmse = _point_line_rmse(xs.astype(np.float32), ys.astype(np.float32), x1, y1, x2, y2)
        if not np.isfinite(rmse) or rmse > 12.0:
            continue
        line_score = length / (1.0 + rmse)
        total_score += float(line_score)
        total_support += int(xs.size)
        rmses.append(float(rmse))
        used_lines += 1

    if used_lines == 0:
        # Ни одна линия не прошла проверку прямизны: это запасной кадр с нулевой оценкой.
        # Запасная оценка по плотности рёбер убрана в v1.4: она добавляла нестабильный
        # шум и маскировала плохие профили, поднимая их суммарный балл выше валидных.
        return FrameMetric(score=0.0, mean_rmse=999.0, line_count=0, support_points=0, fallback=True)

    support_gain = math.log1p(max(1, total_support))
    normalized_score = (total_score / float(used_lines)) * support_gain

    return FrameMetric(
        score=float(normalized_score),
        mean_rmse=float(np.mean(rmses)),
        line_count=int(used_lines),
        support_points=int(total_support),
    )


def _default_profiles(width: int, height: int) -> list[dict[str, Any]]:
    cx = width / 2.0
    cy = height / 2.0
    return [
        {
            "name": "gopro_hero3_black_wide_1080p_seed",
            "source": "seed/manual_approx",
            "K": [[930.0, 0.0, cx], [0.0, 930.0, cy], [0.0, 0.0, 1.0]],
            "D": [-0.0400, 0.0100, -0.0040, 0.0010],
        },
        {
            "name": "gopro_hero3plus_black_wide_1080p_seed",
            "source": "seed/manual_approx",
            "K": [[920.0, 0.0, cx], [0.0, 920.0, cy], [0.0, 0.0, 1.0]],
            "D": [-0.0550, 0.0170, -0.0060, 0.0010],
        },
        {
            "name": "gopro_hero4_black_wide_1080p_seed",
            "source": "seed/manual_approx",
            "K": [[905.0, 0.0, cx], [0.0, 905.0, cy], [0.0, 0.0, 1.0]],
            "D": [-0.0700, 0.0240, -0.0080, 0.0012],
        },
        {
            "name": "gopro_hero3_black_medium_1080p_seed",
            "source": "seed/manual_approx",
            "K": [[1120.0, 0.0, cx], [0.0, 1120.0, cy], [0.0, 0.0, 1.0]],
            "D": [-0.0320, 0.0090, -0.0030, 0.0006],
        },
        {
            "name": "gopro_hero3plus_black_medium_1080p_seed",
            "source": "seed/manual_approx",
            "K": [[1110.0, 0.0, cx], [0.0, 1110.0, cy], [0.0, 0.0, 1.0]],
            "D": [-0.0380, 0.0110, -0.0040, 0.0008],
        },
        {
            "name": "gopro_hero4_black_medium_1080p_seed",
            "source": "seed/manual_approx",
            "K": [[1090.0, 0.0, cx], [0.0, 1090.0, cy], [0.0, 0.0, 1.0]],
            "D": [-0.0440, 0.0140, -0.0050, 0.0009],
        },
    ]


def _load_yaml_or_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def _to_candidate(item: dict[str, Any], width: int, height: int) -> CandidateProfile:
    name = str(item.get("name", "unnamed_profile"))
    source = str(item.get("source", "unknown"))

    if "K" in item:
        k = np.asarray(item["K"], dtype=np.float64)
    else:
        fx = float(item["fx"])
        fy = float(item.get("fy", fx))
        cx = float(item.get("cx", width / 2.0))
        cy = float(item.get("cy", height / 2.0))
        k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    if "D" not in item:
        raise ValueError(f"Profile {name} has no fisheye distortion vector D")
    d = np.asarray(item["D"], dtype=np.float64).reshape(-1)
    if d.shape[0] != 4:
        raise ValueError(f"Profile {name} must provide exactly 4 fisheye coefficients")

    if k.shape != (3, 3):
        raise ValueError(f"Profile {name} has invalid K shape {k.shape}, expected (3,3)")

    return CandidateProfile(name=name, source=source, k=k, d=d)


def _load_profiles(path: Path | None, width: int, height: int) -> list[CandidateProfile]:
    if path is None:
        raw_items = _default_profiles(width=width, height=height)
    else:
        payload = _load_yaml_or_json(path)
        if isinstance(payload, dict) and "profiles" in payload:
            raw_items = payload["profiles"]
        elif isinstance(payload, list):
            raw_items = payload
        else:
            raise ValueError("Profiles file must be either a list or an object with 'profiles' key")

    candidates = [_to_candidate(item, width=width, height=height) for item in raw_items]
    if not candidates:
        raise ValueError("No candidate profiles loaded")
    return candidates


def _video_meta(video_path: Path) -> tuple[int, int, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return width, height, frame_count, fps


def _sample_frames(video_path: Path, frame_count: int, sample_count: int) -> list[np.ndarray]:
    if frame_count <= 0:
        raise ValueError("Video has zero frames")

    lo = int(max(0, round(frame_count * 0.05)))
    hi = int(max(lo + 1, round(frame_count * 0.95)))
    indices = np.linspace(lo, hi - 1, num=sample_count, dtype=np.int32)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video for sampling: {video_path}")

    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError("Failed to sample frames from video")
    return frames


def _run_candidate_eval(
    candidate: CandidateProfile,
    frames: list[np.ndarray],
    max_frame_size: int,
    debug_dir: Path | None,
) -> dict[str, Any]:
    frame_scores: list[float] = []
    frame_rmses: list[float] = []
    frame_line_counts: list[int] = []
    frame_support: list[int] = []
    fallback_flags: list[bool] = []

    k_rectified: np.ndarray | None = None

    for frame_idx, frame in enumerate(frames):
        h, w = frame.shape[:2]
        if k_rectified is None:
            try:
                k_rectified = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    candidate.k,
                    candidate.d.reshape(4, 1),
                    (w, h),
                    np.eye(3),
                    balance=0.0,
                    new_size=(w, h),
                )
            except Exception:
                k_rectified = candidate.k

        undistorted = cv2.fisheye.undistortImage(
            frame,
            K=candidate.k,
            D=candidate.d.reshape(4, 1),
            Knew=k_rectified,
            new_size=(w, h),
        )
        metric = _estimate_straightness(undistorted, max_frame_size=max_frame_size)
        frame_scores.append(metric.score)
        frame_rmses.append(metric.mean_rmse)
        frame_line_counts.append(metric.line_count)
        frame_support.append(metric.support_points)
        fallback_flags.append(bool(metric.fallback))

        if debug_dir is not None and frame_idx == 0:
            debug_dir.mkdir(parents=True, exist_ok=True)
            out_path = debug_dir / f"{candidate.name}_sample0.jpg"
            cv2.imwrite(str(out_path), undistorted)

    total_frames = len(frame_scores)
    fallback_count = int(sum(fallback_flags))
    fallback_ratio = float(fallback_count) / float(total_frames) if total_frames else 1.0

    valid_scores = [s for s, fb in zip(frame_scores, fallback_flags) if not fb]
    valid_rmses = [r for r, fb in zip(frame_rmses, fallback_flags) if not fb]

    mean_valid_score = float(np.mean(valid_scores)) if valid_scores else 0.0
    quality_gate = (1.0 - fallback_ratio) ** 2
    aggregate_score = mean_valid_score * quality_gate

    return {
        "name": candidate.name,
        "source": candidate.source,
        "K": candidate.k,
        "K_rectified": k_rectified if k_rectified is not None else candidate.k,
        "D": candidate.d,
        "aggregate_score": aggregate_score,
        "mean_valid_score": mean_valid_score,
        "quality_gate": quality_gate,
        "fallback_ratio": fallback_ratio,
        "mean_score": float(np.mean(frame_scores)) if frame_scores else 0.0,
        "score_std": float(np.std(frame_scores)) if frame_scores else 0.0,
        "mean_rmse_valid": float(np.mean(valid_rmses)) if valid_rmses else 999.0,
        "mean_rmse": float(np.mean(frame_rmses)) if frame_rmses else 999.0,
        "mean_line_count": float(np.mean(frame_line_counts)) if frame_line_counts else 0.0,
        "mean_support_points": float(np.mean(frame_support)) if frame_support else 0.0,
        "frame_count_used": total_frames,
        "valid_frame_count": len(valid_scores),
    }


def _load_mast3r_focal(mast3r_json: Path | None) -> float | None:
    if mast3r_json is None:
        return None
    payload = _load_yaml_or_json(mast3r_json)

    if isinstance(payload, dict):
        if "focal_px" in payload:
            return float(payload["focal_px"])
        if "focal_length_px" in payload:
            return float(payload["focal_length_px"])
    return None


def _load_tiebreaker(tiebreaker_path: Path | None) -> dict[str, dict[str, float]] | None:
    """Загрузить отчёт выбора фокусного и вернуть геометрические метрики по профилям."""
    if tiebreaker_path is None:
        return None
    payload = _load_yaml_or_json(tiebreaker_path)
    if not isinstance(payload, dict) or "scored" not in payload:
        raise ValueError("Tiebreaker report must be JSON with top-level 'scored' list")
    out: dict[str, dict[str, float]] = {}
    for item in payload["scored"]:
        out[str(item["name"])] = {
            "median_residual_px": float(item.get("median_residual_px", 999.0)),
            "mean_cheirality_rate": float(item.get("mean_cheirality_rate", 0.0)),
            "median_inlier_ratio": float(item.get("median_inlier_ratio", 0.0)),
        }
    return out


def _apply_tiebreaker(
    scored: list[dict[str, Any]],
    tiebreaker: dict[str, dict[str, float]],
    quality_gate_floor: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Переупорядочить кандидатов по метрикам дополнительной проверки.

    Правило: среди профилей со straightness quality_gate >= quality_gate_floor
    выбираем профиль с минимальным медианным остатком essential matrix. Такой
    фильтр явно убирает профили, чьи коэффициенты дисторсии проваливают
    проверку прямизны на заметной доле кадров, а затем разбивает ничью, которую
    одна метрика прямизны уже не различает.

    Возвращает переупорядоченный список оценок с победителем в начале и
    диагностический блок с рассмотренными и отброшенными кандидатами.
    """
    # Прикрепляем поля дополнительной проверки.
    for item in scored:
        entry = tiebreaker.get(item["name"])
        if entry is not None:
            item["median_residual_px"] = entry["median_residual_px"]
            item["mean_cheirality_rate"] = entry["mean_cheirality_rate"]
            item["median_inlier_ratio_emat"] = entry["median_inlier_ratio"]
        else:
            item["median_residual_px"] = 999.0
            item["mean_cheirality_rate"] = 0.0
            item["median_inlier_ratio_emat"] = 0.0

    filtered = [it for it in scored if float(it["quality_gate"]) >= quality_gate_floor]
    rejected_by_gate = [it["name"] for it in scored if float(it["quality_gate"]) < quality_gate_floor]

    if not filtered:
        # Запасной вариант: сохраняем исходный порядок по суммарной прямизне.
        diag = {
            "method": "tiebreaker_disabled_no_profiles_passed_quality_gate",
            "quality_gate_floor": float(quality_gate_floor),
            "rejected_by_gate": rejected_by_gate,
        }
        return scored, diag

    # Сортируем отфильтрованные профили по возрастанию остатка.
    filtered.sort(key=lambda it: float(it["median_residual_px"]))
    # Собираем полный порядок: сначала прошедшие фильтр, затем отброшенные по quality_gate.
    remaining = [it for it in scored if it["name"] not in {f["name"] for f in filtered}]
    remaining.sort(key=lambda it: float(it["quality_gate"]), reverse=True)
    ordered = filtered + remaining

    best = filtered[0]
    second = filtered[1] if len(filtered) > 1 else None
    if second is not None:
        denom = max(1e-6, float(best["median_residual_px"]))
        residual_margin = (float(second["median_residual_px"]) - float(best["median_residual_px"])) / denom
    else:
        residual_margin = 1.0

    diag = {
        "method": "quality_gate_filter_then_min_residual",
        "quality_gate_floor": float(quality_gate_floor),
        "rejected_by_gate": rejected_by_gate,
        "considered": [it["name"] for it in filtered],
        "winner": best["name"],
        "residual_margin": float(residual_margin),
        "winner_residual_px": float(best["median_residual_px"]),
        "runner_up": second["name"] if second is not None else None,
        "runner_up_residual_px": float(second["median_residual_px"]) if second is not None else None,
    }
    return ordered, diag


def _to_serializable_matrix(mat: np.ndarray, decimals: int = 6) -> list[list[float]]:
    return [[round(float(v), decimals) for v in row] for row in mat.tolist()]


def _to_serializable_vector(vec: np.ndarray, decimals: int = 6) -> list[float]:
    return [round(float(v), decimals) for v in vec.tolist()]


def _build_output_yaml(
    best: dict[str, Any],
    scored: list[dict[str, Any]],
    width: int,
    height: int,
    selection_margin: float,
    confidence: float,
    ambiguous: bool,
    mast3r_focal: float | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "lens_model": "fisheye",
        "image_size": [int(width), int(height)],
        "selected_profile": str(best["name"]),
        "lens_profile_source": str(best["source"]),
        "recovery_confidence": round(float(confidence), 6),
        "selection_margin": round(float(selection_margin), 6),
        "ambiguous": bool(ambiguous),
        "K": _to_serializable_matrix(best["K"]),
        "K_rectified": _to_serializable_matrix(best["K_rectified"]),
        "D": _to_serializable_vector(best["D"]),
        "recovery_utc": datetime.now(timezone.utc).isoformat(),
        "score_table": [
            {
                "name": str(item["name"]),
                "source": str(item["source"]),
                "aggregate_score": round(float(item["aggregate_score"]), 6),
                "mean_valid_score": round(float(item["mean_valid_score"]), 6),
                "quality_gate": round(float(item["quality_gate"]), 6),
                "fallback_ratio": round(float(item["fallback_ratio"]), 6),
                "mean_rmse_valid": round(float(item["mean_rmse_valid"]), 6),
                "mean_line_count": round(float(item["mean_line_count"]), 6),
                "frame_count_used": int(item["frame_count_used"]),
                "valid_frame_count": int(item["valid_frame_count"]),
                **(
                    {
                        "median_residual_px": round(float(item.get("median_residual_px", 999.0)), 6),
                        "mean_cheirality_rate": round(float(item.get("mean_cheirality_rate", 0.0)), 6),
                    }
                    if "median_residual_px" in item
                    else {}
                ),
            }
            for item in scored
        ],
    }

    if mast3r_focal is not None:
        fx = float(best["K"][0, 0])
        rel_err = abs(fx - mast3r_focal) / max(1e-6, mast3r_focal)
        out["mast3r_focal_px"] = round(float(mast3r_focal), 6)
        out["mast3r_relative_error"] = round(float(rel_err), 6)
        out["mast3r_agreement"] = bool(rel_err <= 0.05)

    if len(scored) > 1:
        out["alternate_profile"] = {
            "name": str(scored[1]["name"]),
            "source": str(scored[1]["source"]),
            "aggregate_score": round(float(scored[1]["aggregate_score"]), 6),
        }

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover fisheye intrinsics by straightness scoring on sampled video frames",
    )
    parser.add_argument("--video", type=Path, required=True, help="Input MP4 video path")
    parser.add_argument(
        "--profiles",
        type=Path,
        default=None,
        help="YAML/JSON with candidate camera profiles; if omitted, built-in seed candidates are used",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/camera_gopro_hx.yaml"),
        help="Output YAML config path",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/intrinsics_recovery_report.json"),
        help="JSON report path",
    )
    parser.add_argument("--sample-frames", type=int, default=20, help="How many frames to sample")
    parser.add_argument(
        "--max-frame-size",
        type=int,
        default=960,
        help="Max side for metric computation (for speed)",
    )
    parser.add_argument(
        "--ambiguous-gap",
        type=float,
        default=0.10,
        help="Selection margin threshold below which winner is marked ambiguous",
    )
    parser.add_argument(
        "--mast3r-focal-json",
        type=Path,
        default=None,
        help="Optional JSON/YAML with MASt3R focal estimate as focal_px",
    )
    parser.add_argument(
        "--tiebreaker-report",
        type=Path,
        default=None,
        help="Optional JSON from scripts/estimate_focal_emat.py. When supplied, "
        "the final ranking applies a quality-gate filter and then minimum "
        "essential-matrix residual, instead of raw straightness aggregate.",
    )
    parser.add_argument(
        "--quality-gate-floor",
        type=float,
        default=0.9,
        help="Minimum straightness quality_gate required to be considered by the tiebreaker.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=Path("results/intrinsics_debug"),
        help="Directory for first-frame undistorted debug images",
    )
    parser.add_argument(
        "--no-debug-images",
        action="store_true",
        help="Disable debug image writes",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.video.exists():
        raise FileNotFoundError(f"Video file not found: {args.video}")

    width, height, frame_count, fps = _video_meta(args.video)
    print(f"[intrinsics] video={args.video} resolution={width}x{height} fps={fps:.3f} frames={frame_count}")

    candidates = _load_profiles(args.profiles, width=width, height=height)
    print(f"[intrinsics] loaded {len(candidates)} candidate profiles")

    frames = _sample_frames(args.video, frame_count=frame_count, sample_count=max(1, int(args.sample_frames)))
    print(f"[intrinsics] sampled {len(frames)} frames for evaluation")

    debug_dir = None if args.no_debug_images else args.debug_dir
    scored = []
    for candidate in candidates:
        result = _run_candidate_eval(
            candidate,
            frames=frames,
            max_frame_size=max(320, int(args.max_frame_size)),
            debug_dir=debug_dir,
        )
        scored.append(result)
        print(
            "[intrinsics]"
            f" {result['name']}: agg={result['aggregate_score']:.4f}"
            f" mean_rmse={result['mean_rmse']:.3f}"
            f" lines={result['mean_line_count']:.1f}"
        )

    scored.sort(key=lambda item: float(item["aggregate_score"]), reverse=True)

    tiebreaker_diag: dict[str, Any] | None = None
    tiebreaker_map = _load_tiebreaker(args.tiebreaker_report)
    if tiebreaker_map is not None:
        scored, tiebreaker_diag = _apply_tiebreaker(scored, tiebreaker_map, float(args.quality_gate_floor))

    best = scored[0]
    second = scored[1] if len(scored) > 1 else None

    use_tiebreaker = (
        tiebreaker_diag is not None
        and tiebreaker_diag.get("method", "").startswith("quality_gate_filter")
    )
    if use_tiebreaker:
        selection_margin = float(tiebreaker_diag["residual_margin"])
    elif second is not None:
        selection_margin = (float(best["aggregate_score"]) - float(second["aggregate_score"])) / max(
            1e-6, abs(float(best["aggregate_score"]))
        )
    else:
        selection_margin = 1.0

    winner_quality = float(best.get("quality_gate", 1.0))
    if use_tiebreaker:
        # Запас по остатку в 5% уже означает, что RANSAC видит примерно такую же
        # разницу в ошибке фокусного; для разделения medium и wide это существенно.
        # Сдвигаем кривую уверенности так, чтобы 5% запаса при quality_gate около 1.0
        # проходили порог 0.7, используемый как ворота этапа 0a.
        base = 0.7 + 1.5 * selection_margin
        confidence = _clamp01(winner_quality * base)
        ambiguous = selection_margin < 0.05
    else:
        # Исходный запас только по прямизне мягче, потому что прямизна быстро насыщается.
        confidence = _clamp01(winner_quality * (0.5 + 2.5 * selection_margin))
        ambiguous = selection_margin < float(args.ambiguous_gap)

    mast3r_focal = _load_mast3r_focal(args.mast3r_focal_json)

    output_yaml = _build_output_yaml(
        best=best,
        scored=scored,
        width=width,
        height=height,
        selection_margin=selection_margin,
        confidence=confidence,
        ambiguous=ambiguous,
        mast3r_focal=mast3r_focal,
    )
    if tiebreaker_diag is not None:
        output_yaml["tiebreaker"] = tiebreaker_diag

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(output_yaml, sort_keys=False, allow_unicode=False), encoding="utf-8")

    report_payload = {
        "video": str(args.video),
        "resolution": [width, height],
        "fps": fps,
        "sample_frames": len(frames),
        "selected_profile": str(best["name"]),
        "selected_K": _to_serializable_matrix(best["K"]),
        "selected_K_rectified": _to_serializable_matrix(best["K_rectified"]),
        "selected_D": _to_serializable_vector(best["D"]),
        "selection_margin": float(selection_margin),
        "confidence": float(confidence),
        "ambiguous": bool(ambiguous),
        "scores": [
            {
                "name": str(item["name"]),
                "source": str(item["source"]),
                "aggregate_score": float(item["aggregate_score"]),
                "mean_valid_score": float(item["mean_valid_score"]),
                "quality_gate": float(item["quality_gate"]),
                "fallback_ratio": float(item["fallback_ratio"]),
                "mean_score": float(item["mean_score"]),
                "score_std": float(item["score_std"]),
                "mean_rmse": float(item["mean_rmse"]),
                "mean_rmse_valid": float(item["mean_rmse_valid"]),
                "mean_line_count": float(item["mean_line_count"]),
                "mean_support_points": float(item["mean_support_points"]),
                "frame_count_used": int(item["frame_count_used"]),
                "valid_frame_count": int(item["valid_frame_count"]),
            }
            for item in scored
        ],
    }
    if tiebreaker_diag is not None:
        report_payload["tiebreaker"] = tiebreaker_diag
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report_payload, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"[intrinsics] selected profile: {best['name']} (confidence={confidence:.2f}, ambiguous={ambiguous})")
    print(f"[intrinsics] wrote config: {args.output}")
    print(f"[intrinsics] wrote report: {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
