"""Thesis-ready figure для AerialVL ablation: 3 trajectory + metrics table.

Слева: trajectory plot всех трёх runs (baseline / +sem old / +sem new).
Справа: per-frame error curve.
Снизу: bar chart по ключевым метрикам.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


RUNS = [
    ("baseline (no semantic)", "results/aerialvl_ablation_baseline", "#999999"),
    ("+sem_mask Phase B (old, mIoU 50.75)", "results/aerialvl_ablation_sem_old", "#cc4444"),
    ("+sem_mask Phase C+OSM (new, mIoU 47.71)", "results/aerialvl_ablation_sem_phase_c_osm", "#3366cc"),
]


def load_run(out_dir: str):
    summary = json.loads((Path(out_dir) / "summary.json").read_text())
    rows = list(csv.DictReader((Path(out_dir) / "state.csv").open()))
    return summary, rows


def main():
    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(
        "AerialVL ablation: вклад сегментации в локализацию (out-of-domain — Циндао, Китай)",
        fontsize=14, fontweight="bold")

    gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.30,
                          top=0.92, bottom=0.08, left=0.07, right=0.96,
                          height_ratios=[1.4, 1])

    # Top-left: trajectory in ENU
    ax_traj = fig.add_subplot(gs[0, 0:2])
    ax_traj.set_title("Траектория (ENU): GT vs estimated")
    ax_traj.set_xlabel("east, м")
    ax_traj.set_ylabel("north, м")
    ax_traj.set_aspect("equal")

    # Top-right: error vs frame
    ax_err = fig.add_subplot(gs[0, 2])
    ax_err.set_title("Position error vs frame")
    ax_err.set_xlabel("frame index")
    ax_err.set_ylabel("error, м")
    ax_err.axhline(100, color="grey", linestyle="--", alpha=0.4, label="track threshold 100м")

    gt_drawn = False
    for label, out_dir, color in RUNS:
        summary, rows = load_run(out_dir)
        x_e = np.array([float(r["x_e"]) for r in rows])
        y_n = np.array([float(r["y_n"]) for r in rows])
        gt_x = np.array([float(r["gt_x_e"]) for r in rows])
        gt_y = np.array([float(r["gt_y_n"]) for r in rows])
        err = np.array([float(r["error_m"]) for r in rows])

        if not gt_drawn:
            ax_traj.plot(gt_x, gt_y, "k--", linewidth=1.5, label="ground truth", alpha=0.8)
            gt_drawn = True
        ax_traj.plot(x_e, y_n, color=color, linewidth=1.6, label=label, alpha=0.85)
        ax_err.plot(np.arange(len(err)), err, color=color, linewidth=1.4, alpha=0.85, label=label)

    ax_traj.legend(loc="best", fontsize=8)
    ax_err.legend(loc="best", fontsize=8)

    # Bottom: metrics bar chart
    ax_bar = fig.add_subplot(gs[1, :])
    metrics = ["median", "mean", "p95", "final", "map_acc"]
    metric_labels = ["median err, м", "mean err, м", "p95 err, м", "final err, м", "map accepts"]
    metric_keys = ["median_error_m", "mean_error_m", "p95_error_m", "final_error_m"]
    width = 0.25
    x = np.arange(len(metrics))

    summaries = [load_run(out_dir)[0] for _, out_dir, _ in RUNS]
    for i, ((label, _, color), summ) in enumerate(zip(RUNS, summaries)):
        vals = [summ.get(k, 0.0) for k in metric_keys]
        # map_accept: посчитаем из state
        _, rows = load_run(RUNS[i][1])
        n_acc = sum(int(r["map_accepted"]) for r in rows)
        n_att = sum(1 for r in rows if r["map_meas_inliers"] not in ("", "0", None))
        vals.append(n_acc)  # raw count for bar (will scale)
        offsets = (i - 1) * width
        ax_bar.bar(x + offsets, vals, width=width, label=label, color=color, edgecolor="black", linewidth=0.5)
        for xi, v in zip(x + offsets, vals):
            ax_bar.text(xi, v + 1.0, f"{v:.1f}", ha="center", fontsize=8)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_labels)
    ax_bar.set_title("Сводные метрики")
    ax_bar.legend(loc="upper left", fontsize=8)
    ax_bar.grid(axis="y", alpha=0.3)

    out_path = Path("docs/figures/phase4/fig_aerialvl_ablation.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
