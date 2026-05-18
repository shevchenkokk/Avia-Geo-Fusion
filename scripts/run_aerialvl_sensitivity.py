"""Сценарная матрица AerialVL: высота, размер crop и oracle/non-oracle режим.

Один красивый прогон на AerialVL мало что доказывает. Этот скрипт запускает
`run_aerialvl_pipeline.py` много раз и показывает, где ломается система:

- если `oracle_map_crop=True` работает, а обычный режим нет — проблема в VO/масштабе;
- если не работает даже oracle — проблема в матчинге кадр↔карта или ориентации;
- если высокая AGL ломает всё — текущая последовательность не соответствует такому масштабу.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _safe_name(value: object) -> str:
    return str(value).replace("/", "_").replace(".", "p").replace(" ", "")


def _run_one(
    args: argparse.Namespace, backend: str, agl: float, crop: float, oracle: bool
) -> dict[str, Any]:
    tag = f"{backend}_agl{_safe_name(int(agl))}_crop{_safe_name(int(crop))}_{'oracle' if oracle else 'ekf'}"
    out_dir = args.output / tag
    cmd = [
        args.python,
        "scripts/run_aerialvl_pipeline.py",
        "--dataset-root",
        str(args.dataset_root),
        "--manifest-dir",
        str(args.manifest_dir),
        "--sequence",
        args.sequence,
        "--backend",
        backend,
        "--output",
        str(out_dir),
        "--max-frames",
        str(args.max_frames),
        "--map-period-frames",
        str(args.map_period_frames),
        "--map-crop-span-m",
        str(crop),
        "--map-crop-size",
        str(args.map_crop_size),
        "--agl-m",
        str(agl),
        "--pass-p95-threshold-m",
        str(args.pass_p95_threshold_m),
        "--pass-track-pct-threshold",
        str(args.pass_track_pct_threshold),
    ]
    if oracle:
        cmd.append("--oracle-map-crop")
    else:
        cmd.append("--no-oracle-map-crop")

    print(f"[sensitivity] {tag}")
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=args.timeout_s,
    )
    duration_s = time.time() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runner_stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (out_dir / "runner_stderr.txt").write_text(proc.stderr, encoding="utf-8")

    summary_path = out_dir / "summary.json"
    row: dict[str, Any] = {
        "tag": tag,
        "backend_requested": backend,
        "agl_m": agl,
        "map_crop_span_m": crop,
        "oracle_map_crop": int(oracle),
        "returncode": proc.returncode,
        "duration_s": round(duration_s, 3),
        "output": str(out_dir),
    }
    if proc.returncode != 0 or not summary_path.exists():
        row.update(
            {
                "passed": False,
                "error": "runner_failed_or_summary_missing",
            }
        )
        print(f"  -> FAIL rc={proc.returncode}")
        return row

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in (
        "passed",
        "backend",
        "median_error_m",
        "p95_error_m",
        "mean_error_m",
        "final_error_m",
        "track_pct",
        "vo_valid_pct",
        "map_attempts",
        "map_accepted",
        "map_accept_pct",
    ):
        row[key] = summary.get(key)
    print(
        f"  -> {'PASS' if row.get('passed') else 'FAIL'}  "
        f"p95={row.get('p95_error_m'):.1f}м  "
        f"track={row.get('track_pct'):.1f}%  "
        f"map={row.get('map_accepted')}/{row.get('map_attempts')}"
    )
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# AerialVL: sensitivity matrix",
        "",
        "| tag | pass | p95, м | median, м | track, % | map accepted | backend |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('tag')} | {row.get('passed')} | "
            f"{float(row.get('p95_error_m', float('nan'))):.1f} | "
            f"{float(row.get('median_error_m', float('nan'))):.1f} | "
            f"{float(row.get('track_pct', float('nan'))):.1f} | "
            f"{row.get('map_accepted')}/{row.get('map_attempts')} | "
            f"{row.get('backend')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("data/external/aerialvl")
    )
    parser.add_argument(
        "--manifest-dir", type=Path, default=Path("data/aerialvl/manifests")
    )
    parser.add_argument("--sequence", default="2023-03-11-11-48-35")
    parser.add_argument(
        "--output", type=Path, default=Path("results/aerialvl_sensitivity")
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--backend", nargs="+", default=["orb", "xfeat"])
    parser.add_argument(
        "--agl-m", nargs="+", type=float, default=[150.0, 300.0, 500.0, 750.0]
    )
    parser.add_argument(
        "--map-crop-span-m", nargs="+", type=float, default=[500.0, 1200.0]
    )
    parser.add_argument(
        "--oracle-mode", choices=["both", "ekf", "oracle"], default="both"
    )
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument("--map-period-frames", type=int, default=5)
    parser.add_argument("--map-crop-size", type=int, default=900)
    parser.add_argument("--pass-p95-threshold-m", type=float, default=100.0)
    parser.add_argument("--pass-track-pct-threshold", type=float, default=80.0)
    parser.add_argument("--timeout-s", type=int, default=1200)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    if args.oracle_mode == "both":
        oracle_values = [False, True]
    elif args.oracle_mode == "oracle":
        oracle_values = [True]
    else:
        oracle_values = [False]

    rows: list[dict[str, Any]] = []
    for backend in args.backend:
        for agl in args.agl_m:
            for crop in args.map_crop_span_m:
                for oracle in oracle_values:
                    rows.append(_run_one(args, backend, agl, crop, oracle))

    _write_csv(args.output / "matrix.csv", rows)
    (args.output / "summary.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_md(args.output / "summary.md", rows)
    print(f"[sensitivity] matrix  -> {args.output / 'matrix.csv'}")
    print(f"[sensitivity] summary -> {args.output / 'summary.md'}")


if __name__ == "__main__":
    main()
