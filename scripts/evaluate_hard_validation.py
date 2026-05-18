"""Посчитать строгие метрики валидации GP010269.MP4 по состоянию пайплайна и ориентирам."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.frame_bridge import FrameBridge


@dataclass(frozen=True)
class Segment:
    id: str
    name: str
    start_s: float
    end_s: float
    kind: str = ""


@dataclass(frozen=True)
class Landmark:
    id: str
    t_sec: float
    lat: float
    lon: float
    sigma_m: float
    source: str = ""
    note: str = ""


def _load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return payload


def _float(value: object, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid numeric value for {name}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise SystemExit(f"Invalid numeric value for {name}: {value!r}")
    return parsed


def _load_segments(payload: dict) -> list[Segment]:
    items = payload.get("segments")
    if not isinstance(items, list) or not items:
        raise SystemExit("Landmark file must define a non-empty 'segments' list")
    segments: list[Segment] = []
    seen_ids: set[str] = set()
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise SystemExit(f"segments[{idx}] must be an object")
        seg_id = str(item.get("id", "")).strip()
        name = str(item.get("name", seg_id)).strip()
        if not seg_id:
            raise SystemExit(f"segments[{idx}].id is required")
        if seg_id in seen_ids:
            raise SystemExit(f"Duplicate segment id: {seg_id}")
        seen_ids.add(seg_id)
        start_s = _float(item.get("start_s"), f"segments[{idx}].start_s")
        end_s = _float(item.get("end_s"), f"segments[{idx}].end_s")
        if end_s <= start_s:
            raise SystemExit(f"segments[{idx}] end_s must be greater than start_s")
        segments.append(Segment(seg_id, name, start_s, end_s, str(item.get("kind", ""))))
    segments.sort(key=lambda segment: segment.start_s)
    for prev, cur in zip(segments, segments[1:]):
        if cur.start_s < prev.end_s:
            raise SystemExit(
                f"Overlapping segments: {prev.id} [{prev.start_s}, {prev.end_s}) and "
                f"{cur.id} [{cur.start_s}, {cur.end_s})"
            )
    return segments


def _load_landmarks(payload: dict) -> list[Landmark]:
    items = payload.get("landmarks")
    if not isinstance(items, list):
        raise SystemExit("Landmark file must define a 'landmarks' list")
    landmarks: list[Landmark] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise SystemExit(f"landmarks[{idx}] must be an object")
        lm_id = str(item.get("id", f"lm_{idx:03d}")).strip()
        t_sec = _float(item.get("t_sec"), f"landmarks[{idx}].t_sec")
        lat = _float(item.get("lat"), f"landmarks[{idx}].lat")
        lon = _float(item.get("lon"), f"landmarks[{idx}].lon")
        sigma_m = _float(item.get("sigma_m", 100.0), f"landmarks[{idx}].sigma_m")
        if sigma_m <= 0:
            raise SystemExit(f"landmarks[{idx}].sigma_m must be positive")
        landmarks.append(
            Landmark(
                id=lm_id,
                t_sec=t_sec,
                lat=lat,
                lon=lon,
                sigma_m=sigma_m,
                source=str(item.get("source", "")),
                note=str(item.get("note", "")),
            )
        )
    landmarks.sort(key=lambda lm: lm.t_sec)
    if len(landmarks) < 2:
        raise SystemExit("Landmark file must contain at least two points for interpolation")
    for prev, cur in zip(landmarks, landmarks[1:]):
        if cur.t_sec <= prev.t_sec:
            raise SystemExit("landmarks[].t_sec must be strictly increasing")
    return landmarks


def _load_state_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"State CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"State CSV is empty: {path}")
    required = {"t_sec", "lat", "lon", "sigma_pos_m", "map_accepted", "obstructed"}
    missing = sorted(required - set(rows[0].keys()))
    if missing:
        raise SystemExit(f"State CSV is missing columns: {', '.join(missing)}")
    rows.sort(key=lambda row: _float(row["t_sec"], "state.t_sec"))
    return rows


def _make_bridge(payload: dict, landmarks: list[Landmark], rows: list[dict[str, str]]) -> FrameBridge:
    origin = payload.get("origin")
    if isinstance(origin, dict):
        lat0 = _float(origin.get("lat"), "origin.lat")
        lon0 = _float(origin.get("lon"), "origin.lon")
        alt0 = _float(origin.get("alt_msl", 0.0), "origin.alt_msl")
        return FrameBridge(lat0=lat0, lon0=lon0, alt0_msl=alt0)
    if landmarks:
        return FrameBridge(lat0=landmarks[0].lat, lon0=landmarks[0].lon, alt0_msl=0.0)
    return FrameBridge(
        lat0=_float(rows[0]["lat"], "state.lat"),
        lon0=_float(rows[0]["lon"], "state.lon"),
        alt0_msl=0.0,
    )


def _landmark_truth_interpolator(
    landmarks: list[Landmark],
    bridge: FrameBridge,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.array([lm.t_sec for lm in landmarks], dtype=np.float64)
    enu = np.array([bridge.wgs84_to_enu(lm.lat, lm.lon, 0.0)[:2] for lm in landmarks])
    sigmas = np.array([lm.sigma_m for lm in landmarks], dtype=np.float64)
    return times, enu, sigmas


def _truth_at(
    t_sec: float,
    times: np.ndarray,
    enu: np.ndarray,
    sigmas: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    if len(times) < 2 or t_sec < times[0] or t_sec > times[-1]:
        return None
    x = float(np.interp(t_sec, times, enu[:, 0]))
    y = float(np.interp(t_sec, times, enu[:, 1]))
    sigma = float(np.interp(t_sec, times, sigmas))
    return np.array([x, y], dtype=np.float64), sigma


def _segment_for_time(t_sec: float, segments: list[Segment]) -> Segment | None:
    for seg in segments:
        if seg.start_s <= t_sec < seg.end_s:
            return seg
    if segments and math.isclose(t_sec, segments[-1].end_s):
        return segments[-1]
    return None


def _bool_int(row: dict[str, str], key: str) -> int:
    value = row.get(key)
    try:
        parsed_float = _float(value, f"state.{key}")
    except SystemExit as exc:
        raise SystemExit(f"Invalid numeric flag for state.{key}: {value!r}") from exc
    parsed = int(parsed_float)
    if parsed_float != parsed or parsed not in (0, 1):
        raise SystemExit(f"state.{key} must be 0 or 1, got {value!r}")
    return parsed


def _percent(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return 100.0 * num / den


def _fmt(value: float | None, precision: int = 1) -> str:
    if value is None or not math.isfinite(value):
        return "N/A"
    return f"{value:.{precision}f}"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    summary_rows: list[dict[str, object]],
    landmarks: list[Landmark],
    state_csv: Path,
    landmark_path: Path,
    plot_path: Path | None,
) -> None:
    lines = [
        "# GP010269 hard-validation report",
        "",
        f"- state CSV: `{state_csv}`",
        f"- landmarks: `{landmark_path}`",
        f"- landmark count: `{len(landmarks)}`",
        "",
        "## Segment metrics",
        "",
        "| Segment | Window, s | GT rows | ATE mean, m | ATE p95, m | % TRACK | False fixes | Recovery, s | DR behavior |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['segment_id']} {row['name']} | {row['start_s']:.0f}-{row['end_s']:.0f} | "
            f"{row['gt_rows']} | {_fmt(row['ate_mean_m'])} | {_fmt(row['ate_p95_m'])} | "
            f"{_fmt(row['track_pct'])} | {row['false_fix_count']} | {_fmt(row['recovery_time_s'])} | "
            f"{row['dead_reckoning_behavior']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- ATE is computed only where landmark interpolation covers the state timestamp.",
        "- `% TRACK` is the share of rows with `sigma_pos_m <= max_sigma_pos_m` and no obstruction flag.",
        "- `false_fix_count` is map-fix count while obstructed, or all map-fixes inside expected dead-reckoning segments.",
        "- `recovery_time_s` is measured after an expected dead-reckoning segment until the first accepted map fix.",
        "- Replace the example landmark file with visually confirmed landmarks before using the numbers in the thesis.",
    ])
    if plot_path is not None:
        lines.extend(["", f"Plot: `{plot_path}`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(
    output: Path,
    rows: list[dict[str, object]],
    landmarks: list[Landmark],
    bridge: FrameBridge,
) -> Path | None:
    if not rows:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    xs = np.array([float(row["x_e"]) for row in rows])
    ys = np.array([float(row["y_n"]) for row in rows])
    colors = np.array([float(row["ate_m"]) if row["ate_m"] is not None else np.nan for row in rows])
    lm_enu = np.array([bridge.wgs84_to_enu(lm.lat, lm.lon, 0.0)[:2] for lm in landmarks])

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot(xs, ys, color="tab:blue", lw=1.0, label="EKF state")
    if np.isfinite(colors).any():
        sc = ax.scatter(xs, ys, c=colors, s=18, cmap="magma", label="ATE samples")
        fig.colorbar(sc, ax=ax, label="ATE, m")
    if len(lm_enu):
        ax.scatter(lm_enu[:, 0], lm_enu[:, 1], marker="x", s=90, color="tab:red", label="landmarks")
    ax.set_xlabel("east, m")
    ax.set_ylabel("north, m")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    ax.set_title("GP010269 hard-validation trajectory")
    fig.tight_layout()
    path = output / "trajectory_vs_landmarks.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-csv", type=Path, default=Path("results/full_pipeline/state.csv"))
    parser.add_argument(
        "--landmarks",
        type=Path,
        default=Path("data/ground_truth/gp010269_landmarks.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("results/hard_validation"))
    parser.add_argument("--allow-example-landmarks", action="store_true")
    parser.add_argument("--max-sigma-pos-m", type=float, default=None)
    args = parser.parse_args()

    if not args.landmarks.exists():
        raise SystemExit(
            f"Landmark file not found: {args.landmarks}. "
            "Copy data/ground_truth/gp010269_landmarks.example.json and replace examples."
        )
    payload = _load_json(args.landmarks)
    if args.landmarks.name.endswith(".example.json") and not args.allow_example_landmarks:
        raise SystemExit("Refusing to evaluate example landmarks without --allow-example-landmarks")

    segments = _load_segments(payload)
    landmarks = _load_landmarks(payload)
    rows = _load_state_rows(args.state_csv)
    bridge = _make_bridge(payload, landmarks, rows)
    threshold_payload = payload.get("track_thresholds", {})
    threshold_from_file = (
        threshold_payload.get("max_sigma_pos_m")
        if isinstance(threshold_payload, dict) else None
    )
    raw_max_sigma = args.max_sigma_pos_m if args.max_sigma_pos_m is not None else threshold_from_file
    max_sigma = _float(
        300.0 if raw_max_sigma is None else raw_max_sigma,
        "track_thresholds.max_sigma_pos_m",
    )
    if max_sigma <= 0:
        raise SystemExit("track_thresholds.max_sigma_pos_m must be positive")
    raw_expected_dr = payload.get("expected_dead_reckoning_segments", [])
    if not isinstance(raw_expected_dr, list):
        raise SystemExit("'expected_dead_reckoning_segments' must be a list")
    expected_dr = {str(item).strip() for item in raw_expected_dr if str(item).strip()}
    unknown_expected_dr = expected_dr - {seg.id for seg in segments}
    if unknown_expected_dr:
        raise SystemExit(
            "Unknown expected_dead_reckoning_segments: "
            + ", ".join(sorted(unknown_expected_dr))
        )

    truth_times, truth_enu, truth_sigmas = _landmark_truth_interpolator(landmarks, bridge)
    detail_rows: list[dict[str, object]] = []
    per_segment: dict[str, list[dict[str, object]]] = {seg.id: [] for seg in segments}

    for row in rows:
        t_sec = _float(row["t_sec"], "state.t_sec")
        segment = _segment_for_time(t_sec, segments)
        if segment is None:
            continue
        x_e = _float(row.get("x_e"), "state.x_e") if row.get("x_e") else bridge.wgs84_to_enu(
            _float(row["lat"], "state.lat"), _float(row["lon"], "state.lon"), 0.0,
        )[0]
        y_n = _float(row.get("y_n"), "state.y_n") if row.get("y_n") else bridge.wgs84_to_enu(
            _float(row["lat"], "state.lat"), _float(row["lon"], "state.lon"), 0.0,
        )[1]
        truth = _truth_at(t_sec, truth_times, truth_enu, truth_sigmas)
        ate_m: float | None = None
        gt_sigma_m: float | None = None
        if truth is not None:
            truth_xy, gt_sigma_m = truth
            ate_m = float(np.linalg.norm(np.array([x_e, y_n]) - truth_xy))
        sigma_pos_m = _float(row["sigma_pos_m"], "state.sigma_pos_m")
        obstructed = _bool_int(row, "obstructed")
        map_accepted = _bool_int(row, "map_accepted")
        track = int(sigma_pos_m <= max_sigma and not obstructed)
        detail = {
            "segment_id": segment.id,
            "segment_name": segment.name,
            "t_sec": t_sec,
            "lat": _float(row["lat"], "state.lat"),
            "lon": _float(row["lon"], "state.lon"),
            "x_e": x_e,
            "y_n": y_n,
            "sigma_pos_m": sigma_pos_m,
            "map_accepted": map_accepted,
            "obstructed": obstructed,
            "track": track,
            "ate_m": ate_m,
            "gt_sigma_m": gt_sigma_m,
        }
        detail_rows.append(detail)
        per_segment[segment.id].append(detail)

    recovery_by_segment: dict[str, float | None] = {}
    for idx, seg in enumerate(segments[:-1]):
        if seg.id not in expected_dr:
            recovery_by_segment[seg.id] = None
            continue
        next_seg = segments[idx + 1]
        candidates = [
            row for row in detail_rows
            if row["segment_id"] == next_seg.id
            and int(row["map_accepted"]) > 0
        ]
        recovery_by_segment[seg.id] = (
            float(candidates[0]["t_sec"]) - seg.end_s if candidates else None
        )

    summary_rows: list[dict[str, object]] = []
    for seg in segments:
        seg_rows = per_segment[seg.id]
        ate_values = np.array(
            [float(row["ate_m"]) for row in seg_rows if row["ate_m"] is not None],
            dtype=np.float64,
        )
        map_fixes = int(sum(int(row["map_accepted"]) for row in seg_rows))
        obstructed_fixes = int(
            sum(int(row["map_accepted"]) for row in seg_rows if int(row["obstructed"]) > 0)
        )
        false_fix_count = map_fixes if seg.id in expected_dr else obstructed_fixes
        track_count = int(sum(int(row["track"]) for row in seg_rows))
        sigma_values = [float(row["sigma_pos_m"]) for row in seg_rows]
        sigma_start = sigma_values[0] if sigma_values else None
        sigma_end = sigma_values[-1] if sigma_values else None
        sigma_grew = (
            sigma_start is not None and sigma_end is not None and sigma_end >= sigma_start
        )
        summary_rows.append({
            "segment_id": seg.id,
            "name": seg.name,
            "kind": seg.kind,
            "start_s": seg.start_s,
            "end_s": seg.end_s,
            "rows": len(seg_rows),
            "gt_rows": len(ate_values),
            "ate_mean_m": float(np.mean(ate_values)) if len(ate_values) else None,
            "ate_median_m": float(np.median(ate_values)) if len(ate_values) else None,
            "ate_p95_m": float(np.percentile(ate_values, 95)) if len(ate_values) else None,
            "track_count": track_count,
            "track_pct": _percent(track_count, len(seg_rows)),
            "map_fix_count": map_fixes,
            "false_fix_count": false_fix_count,
            "obstructed_rows": int(sum(int(row["obstructed"]) for row in seg_rows)),
            "recovery_time_s": recovery_by_segment.get(seg.id),
            "sigma_start_m": sigma_start,
            "sigma_end_m": sigma_end,
            "dead_reckoning_behavior": (
                "sigma_grew" if seg.id in expected_dr and sigma_grew else
                "sigma_not_growing" if seg.id in expected_dr else "N/A"
            ),
        })

    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "segment_metrics.csv", summary_rows)
    _write_csv(args.output / "timeline_metrics.csv", detail_rows)
    plot_path = _plot(args.output, detail_rows, landmarks, bridge)
    json_payload = {
        "state_csv": str(args.state_csv),
        "landmarks": str(args.landmarks),
        "max_sigma_pos_m": max_sigma,
        "landmark_count": len(landmarks),
        "segments": summary_rows,
        "outputs": {
            "segment_metrics": str(args.output / "segment_metrics.csv"),
            "timeline_metrics": str(args.output / "timeline_metrics.csv"),
            "plot": str(plot_path) if plot_path is not None else None,
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown(
        args.output / "summary.md",
        summary_rows,
        landmarks,
        args.state_csv,
        args.landmarks,
        plot_path,
    )
    print(f"[hard-validation] summary -> {args.output / 'summary.md'}")
    print(f"[hard-validation] csv     -> {args.output / 'segment_metrics.csv'}")


if __name__ == "__main__":
    main()
