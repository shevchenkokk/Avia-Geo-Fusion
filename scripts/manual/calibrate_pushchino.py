"""Grid search калибровки камеры Пущино.

Тестирует комбинации (focal_scale, AGL, pitch) на small sample кадров и
выбирает конфигурации по нескольким диагностическим метрикам матчера.

Не идеально (нет GT trajectory чтобы измерить error), но при near-zero
baseline это единственный способ быстро дискриминировать config.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


GRID_FOCAL = [0.5, 0.7, 1.0, 1.3]   # focal_scale → f = scale × max(w,h)
GRID_AGL = [400, 620, 700, 1000]
GRID_PITCH = [-30, -45, -60, -75]


def _parse_float_grid(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_int_grid(raw: str) -> list[int]:
    return [int(float(x.strip())) for x in raw.split(",") if x.strip()]


def _finite_or_inf(value: float) -> float:
    try:
        return value if value == value else float("inf")
    except Exception:
        return float("inf")


def run_one(
    video: str,
    output: Path,
    cam_yaml: Path,
    cruise_start: float,
    cruise_end: float,
    sample_period: float,
    window_radius: int,
    pitch_deg: float,
    agl_m: float,
    retriever_top_k: int,
) -> dict:
    cmd = [
        sys.executable, "scripts/end_to_end_smoke.py",
        "--video", video,
        "--retriever-db", "data/retrieval/pushchino/db_z14",
        "--camera-config", str(cam_yaml),
        "--cruise-start-s", str(cruise_start),
        "--cruise-end-s", str(cruise_end),
        "--sample-period-s", str(sample_period),
        "--window-radius", str(window_radius),
        "--pitch-deg", str(pitch_deg),
        "--agl-m", str(agl_m),
        "--retriever-top-k", str(retriever_top_k),
        "--backend", "xfeat",
        "--output", str(output),
    ]
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "run.log"
    import os
    env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    with log_path.open("w") as f:
        subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    csv_path = output / "smoke.csv"
    if not csv_path.exists():
        return {"accepted": 0, "total": 0, "rate": 0.0, "error": "no_csv"}
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return {"accepted": 0, "total": 0, "rate": 0.0}
    n_acc = sum(1 for r in rows if r.get("accepted") in ("1", "True", "true"))
    accepted_rows = [r for r in rows if r.get("accepted") in ("1", "True", "true")]
    inliers = [
        int(r["num_inliers"]) for r in accepted_rows
        if r.get("num_inliers", "").isdigit()
    ]
    residuals_km = [
        float(r["residual_ret_km"]) for r in accepted_rows
        if r.get("residual_ret_km") not in ("", "nan", "NaN", None)
    ]
    match_ms = [
        float(r["match_ms"]) for r in rows
        if r.get("match_ms") not in ("", "nan", "NaN", None)
    ]
    reject_counts: dict[str, int] = {}
    for r in rows:
        if r.get("accepted") in ("1", "True", "true"):
            continue
        reason = r.get("reject_reason", "") or "unknown"
        reject_counts[reason] = reject_counts.get(reason, 0) + 1
    top_reject = sorted(reject_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "accepted": n_acc,
        "total": len(rows),
        "rate": 100.0 * n_acc / len(rows),
        "med_inliers": sorted(inliers)[len(inliers)//2] if inliers else 0,
        "med_residual_km": (
            sorted(residuals_km)[len(residuals_km)//2] if residuals_km else float("nan")
        ),
        "med_match_ms": sorted(match_ms)[len(match_ms)//2] if match_ms else float("nan"),
        "top_reject": top_reject[0][0] if top_reject else "",
        "top_reject_count": top_reject[0][1] if top_reject else 0,
    }


def build_camera_yaml(template_path: Path, dest: Path,
                       focal_scale: float, pitch_deg: float,
                       image_size: tuple[int, int]) -> None:
    """Создаёт временный yaml с заданными focal/pitch."""
    import yaml
    cfg = yaml.safe_load(template_path.read_text())
    w, h = image_size
    f = focal_scale * max(w, h)
    cfg["image_size"] = [w, h]
    cfg["pitch_deg"] = pitch_deg
    cfg["K"] = [[f, 0.0, w/2], [0.0, f, h/2], [0.0, 0.0, 1.0]]
    cfg["K_rectified"] = cfg["K"]
    dest.write_text(yaml.safe_dump(cfg, sort_keys=False))


def main():
    if sys.version_info < (3, 11):
        raise SystemExit(
            "This project requires Python 3.11. Run via .venv/bin/python "
            "or activate the project virtualenv first."
        )

    p = argparse.ArgumentParser()
    p.add_argument("--video", default="data/videos/Пущино 54_792600 37_637845.MP4")
    p.add_argument("--template-yaml", default="configs/camera_pushchino.yaml")
    p.add_argument("--cruise-start-s", type=float, default=100.0)
    p.add_argument("--cruise-end-s", type=float, default=180.0)
    p.add_argument("--sample-period-s", type=float, default=10.0)
    p.add_argument("--window-radius", type=int, default=2)
    p.add_argument("--retriever-top-k", type=int, default=3)
    p.add_argument("--focal-grid", default=",".join(str(x) for x in GRID_FOCAL))
    p.add_argument("--agl-grid", default=",".join(str(x) for x in GRID_AGL))
    p.add_argument("--pitch-grid", default=",".join(str(x) for x in GRID_PITCH))
    p.add_argument("--out-root", default="results/pushchino_calib")
    args = p.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    tmpl = Path(args.template_yaml)

    image_size = (3840, 2160)
    focal_grid = _parse_float_grid(args.focal_grid)
    agl_grid = _parse_int_grid(args.agl_grid)
    pitch_grid = _parse_float_grid(args.pitch_grid)
    results = []
    for fs in focal_grid:
        for agl_m in agl_grid:
            for pitch in pitch_grid:
                tag = f"fs{fs:g}_agl{agl_m}_pitch{pitch:g}"
                yaml_path = out_root / f"cam_{tag}.yaml"
                build_camera_yaml(tmpl, yaml_path, fs, pitch, image_size)
                out_dir = out_root / tag
                print(f"\n=== focal_scale={fs}  agl={agl_m}m  pitch={pitch}° ===")
                r = run_one(
                    args.video,
                    out_dir,
                    yaml_path,
                    args.cruise_start_s,
                    args.cruise_end_s,
                    args.sample_period_s,
                    args.window_radius,
                    pitch,
                    agl_m,
                    args.retriever_top_k,
                )
                r["focal_scale"] = fs
                r["agl_m"] = agl_m
                r["pitch"] = pitch
                results.append(r)
                print(
                    f"  → {r['accepted']}/{r['total']} ({r['rate']:.1f}%)  "
                    f"med_inl={r.get('med_inliers', 0)}  "
                    f"med_resid={r.get('med_residual_km', float('nan')):.2f}km  "
                    f"top_reject={r.get('top_reject', '')}"
                )

    print("\n\n=== GRID RESULTS (sorted by accept rate) ===")
    results.sort(
        key=lambda r: (
            -r["rate"],
            -r.get("med_inliers", 0),
            _finite_or_inf(r.get("med_residual_km", float("inf"))),
        )
    )
    for r in results:
        print(
            f"  fs={r['focal_scale']}  agl={r['agl_m']:>4}m  pitch={r['pitch']:>4}°  "
            f"{r['accepted']:>2}/{r['total']:>2}  rate={r['rate']:5.1f}%  "
            f"med_inl={r.get('med_inliers', 0):>2}  "
            f"med_resid={r.get('med_residual_km', float('nan')):>5.2f}km  "
            f"reject={r.get('top_reject', '')}"
        )

    out_csv = out_root / "grid_summary.csv"
    with out_csv.open("w", newline="") as f:
        wr = csv.DictWriter(
            f,
            fieldnames=[
                "focal_scale",
                "agl_m",
                "pitch",
                "accepted",
                "total",
                "rate",
                "med_inliers",
                "med_residual_km",
                "med_match_ms",
                "top_reject",
                "top_reject_count",
            ],
        )
        wr.writeheader()
        for r in results:
            wr.writerow({k: r.get(k, 0) for k in wr.fieldnames})
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
