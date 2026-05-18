"""Запускает e2e_smoke ablation на всех учебных видео из data/videos/.
Для каждого видео делает 3 run'а:
  baseline (no semantic) / +sem_old (Phase B baseline) / +sem_new (Phase C+OSM)
И собирает accept rate в таблицу.

Координаты старта берутся из имени файла (формат "<регион> <lat>_<...>_<lon>...").
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


VIDEOS = [
    # (filename, retriever_db_dir, start_lat, start_lon, label)
    ("data/videos/GP010269.MP4", "data/retrieval/db_z14",
     55.086025, 38.149033, "GP010269 (Раменское)"),
    ("data/videos/Алферьево 56_093488 35_888165.MP4", "data/retrieval/alferyevo/db_z14",
     56.093488, 35.888165, "Алферьево"),
    ("data/videos/Борки 56_798047 37_329262.MP4", "data/retrieval/borki/db_z14",
     56.798047, 37.329262, "Борки 1"),
    ("data/videos/Борки3  56_798047 37_329262.MP4", "data/retrieval/borki/db_z14",
     56.798047, 37.329262, "Борки 3"),
    ("data/videos/Пущино 54_792600 37_637845.MP4", "data/retrieval/pushchino/db_z14",
     54.792600, 37.637845, "Пущино"),
]


CONFIGS = [
    ("baseline", []),
    ("sem_old (Phase B, mIoU 50.75)",
     ["--semantic-mask",
      "--seg-config", "results/segformer_overture_b0_focus_resume_wclean_v3/focus_refine/segformer_overture_focus_cfg.py",
      "--seg-checkpoint", "results/segformer_overture_b0_focus_resume_wclean_v3/focus_refine/best_mIoU_iter_220.pth"]),
    ("sem_new (Phase C+OSM, mIoU 47.71)",
     ["--semantic-mask",
      "--seg-config", "results/segformer_overture_b0_phase_c_osm_manualcw/segformer_overture_quick_cfg.py",
      "--seg-checkpoint", "results/segformer_overture_b0_phase_c_osm_manualcw/best_mIoU_iter_1165.pth"]),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", default="results/teacher_videos_ablation")
    p.add_argument("--cruise-start-s", type=float, default=60.0,
                   help="всего ~5 минут крейсера через пайплайн на видео")
    p.add_argument("--cruise-end-s", type=float, default=180.0)
    p.add_argument("--sample-period-s", type=float, default=4.0)
    p.add_argument("--window-radius", type=int, default=2)
    p.add_argument("--device-flag", default="",
                   help="игнорируется — e2e_smoke использует CPU автоматически")
    p.add_argument("--only-video", default="",
                   help="имя файла подмножества видео (фрагмент имени для match)")
    return p.parse_args()


def parse_smoke_csv(csv_path: Path):
    if not csv_path.exists():
        return None
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return None
    n_total = len(rows)
    n_acc = sum(1 for r in rows if r.get("accepted", "0") in ("1", "True", "true"))
    inliers = [int(r["num_inliers"]) for r in rows
               if r.get("accepted", "0") in ("1", "True", "true") and r.get("num_inliers", "").isdigit()]
    return {
        "frames_tested": n_total,
        "frames_accepted": n_acc,
        "rate_pct": 100.0 * n_acc / n_total if n_total else 0.0,
        "median_inliers": sorted(inliers)[len(inliers)//2] if inliers else 0,
        "max_inliers": max(inliers) if inliers else 0,
    }


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    videos = VIDEOS
    if args.only_video:
        videos = [v for v in videos if args.only_video.lower() in v[0].lower()]
        if not videos:
            print(f"no video matched '{args.only_video}'")
            return

    for video, retr_db, lat, lon, label in videos:
        if not Path(video).exists():
            print(f"SKIP missing video: {video}")
            continue
        if not Path(retr_db + ".json").exists():
            print(f"SKIP missing retriever DB: {retr_db}")
            continue
        for cfg_name, cfg_args in CONFIGS:
            run_dir = out_root / re.sub(r"[^A-Za-z0-9_]", "_", label) / cfg_name.replace(" ", "_").replace("(", "").replace(")", "")
            run_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable, "scripts/end_to_end_smoke.py",
                "--video", video,
                "--retriever-db", retr_db,
                "--cruise-start-s", str(args.cruise_start_s),
                "--cruise-end-s", str(args.cruise_end_s),
                "--sample-period-s", str(args.sample_period_s),
                "--window-radius", str(args.window_radius),
                "--backend", "xfeat",
                "--output", str(run_dir),
            ] + cfg_args

            print(f"\n=== {label}  /  {cfg_name}")
            print(f"    cmd: ... {' '.join(cmd[-5:])}")
            env = {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}
            import os
            full_env = {**os.environ, **env}
            log_path = run_dir / "run.log"
            with log_path.open("w") as logf:
                proc = subprocess.run(cmd, env=full_env, stdout=logf, stderr=subprocess.STDOUT, cwd=str(ROOT))
            stats = parse_smoke_csv(run_dir / "smoke.csv")
            tail_log = log_path.read_text().splitlines()[-10:] if log_path.exists() else []
            tail_str = "\n".join(tail_log)
            if stats is None:
                print(f"    FAILED: {tail_str[-300:]}")
                stats = {"frames_tested": 0, "frames_accepted": 0, "rate_pct": 0.0,
                         "median_inliers": 0, "max_inliers": 0, "error": "no_csv"}
            else:
                print(f"    frames {stats['frames_accepted']}/{stats['frames_tested']} "
                      f"({stats['rate_pct']:.1f}%)  med_inl={stats['median_inliers']}")
            results.append({"label": label, "config": cfg_name, **stats})

    summary_path = out_root / "ablation_summary.csv"
    if results:
        with summary_path.open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["label", "config", "frames_tested",
                                                "frames_accepted", "rate_pct",
                                                "median_inliers", "max_inliers"])
            wr.writeheader()
            for r in results:
                wr.writerow({k: r.get(k, "") for k in wr.fieldnames})
        print(f"\nWrote {summary_path}")

    print("\n\n=== SUMMARY ===\n")
    print(f"{'label':<28}{'config':<40}{'rate':>10}")
    for r in results:
        print(f"{r['label']:<28}{r['config']:<40}{r.get('rate_pct',0):>9.1f}%")


if __name__ == "__main__":
    main()
