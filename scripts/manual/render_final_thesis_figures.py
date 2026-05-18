"""Финальные thesis-ready графики для главы «Эксперимент».

1. fig_aerialvl_5seqs_ablation.png — bar chart по 5 sequences x 3 configs
2. fig_drift_vs_map.png — error vs frame для baseline / VO-only / sem_new
3. fig_runtime_fps.png — горизонтальный bar с FPS по стадиям
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SEQS = [
    ("short_03-11", "short_trajtr_2023-03-11-11-48-35"),
    ("short_03-16", "short_trajtr_2023-03-16-16-58-43"),
    ("short_03-18a", "short_trajtr_2023-03-18-16-43-16"),
    ("long_03-18a", "long_trajtr_2023-03-18-14-38-32"),
    ("long_03-18b", "long_trajtr_2023-03-18-15-01-14"),
]

CONFIGS = [
    ("baseline", "baseline", "#999999"),
    ("sem_old\n(Phase B, mIoU 50.75)", "sem_old", "#cc4444"),
    ("sem_new\n(Phase C+OSM, mIoU 47.71)", "sem_new", "#3366cc"),
]


def load(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def fig_5seqs_bar():
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.suptitle("AerialVL ablation: 5 sequences × 3 configs (median position error, м)",
                 fontsize=13, fontweight="bold")

    n_seqs = len(SEQS)
    n_cfg = len(CONFIGS)
    width = 0.27
    x = np.arange(n_seqs)

    for i, (cfg_label, cfg_tag, color) in enumerate(CONFIGS):
        vals = []
        for seq_label, seq_tag in SEQS:
            summary = load(f"results/aerialvl_full_ablation/{seq_tag}__{cfg_tag}/summary.json")
            vals.append(summary["median_error_m"] if summary else 0)
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width=width, label=cfg_label, color=color,
                      edgecolor="black", linewidth=0.6)
        for xi, v in zip(x + offset, vals):
            ax.text(xi, v + 8, f"{v:.0f}", ha="center", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in SEQS])
    ax.set_ylabel("median position error, м")
    ax.set_xlabel("sequence")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Aggregate annotation
    means_per_cfg = []
    for _, cfg_tag, _ in CONFIGS:
        vals = []
        for _, seq_tag in SEQS:
            summary = load(f"results/aerialvl_full_ablation/{seq_tag}__{cfg_tag}/summary.json")
            if summary:
                vals.append(summary["median_error_m"])
        means_per_cfg.append(sum(vals) / len(vals))
    annotation = (
        f"Mean median across 5 sequences:\n"
        f"  baseline = {means_per_cfg[0]:.1f}м\n"
        f"  sem_old (Phase B) = {means_per_cfg[1]:.1f}м (+{(means_per_cfg[1]/means_per_cfg[0]-1)*100:.0f}%)\n"
        f"  sem_new (Phase C+OSM) = {means_per_cfg[2]:.1f}м ({(means_per_cfg[2]/means_per_cfg[0]-1)*100:+.0f}%)"
    )
    ax.text(0.99, 0.97, annotation,
            transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.92))

    out = Path("docs/figures/phase4/fig_aerialvl_5seqs_ablation.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Wrote {out}")


def fig_drift_vs_map():
    """Per-frame error vs frame: VO-only vs baseline+map vs +sem_new"""
    runs = [
        ("VO-only (no map channel)", "results/aerialvl_drift_only_vo", "#cc8800"),
        ("baseline (map, no semantic)", "results/aerialvl_ablation_baseline", "#888888"),
        ("+sem_mask Phase C+OSM", "results/aerialvl_ablation_sem_phase_c_osm", "#3366cc"),
    ]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.suptitle("Per-frame error: важность map-канала и сегментации",
                 fontsize=13, fontweight="bold")

    for label, run_dir, color in runs:
        csv_path = Path(run_dir) / "state.csv"
        if not csv_path.exists():
            continue
        rows = list(csv.DictReader(csv_path.open()))
        err = [float(r["error_m"]) for r in rows]
        ax.plot(np.arange(len(err)), err, color=color, linewidth=1.6, alpha=0.85, label=label)

    ax.axhline(100, color="grey", linestyle="--", alpha=0.5, label="track threshold 100м")
    ax.set_xlabel("frame index (short_trajtr/2023-03-11, 150 frames)")
    ax.set_ylabel("position error, м")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    out = Path("docs/figures/phase4/fig_drift_vs_map.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Wrote {out}")


def fig_runtime():
    stages = [
        ("video decode (cv2)", 1.85, "#88cc88"),
        ("BEV warp", 0.81, "#88cc88"),
        ("SegFormer-B0 inference", 30.08, "#cc8888"),
        ("XFeat keypoint match", 205.94, "#cc4444"),
    ]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.suptitle("Runtime per stage on Mac M-series MPS, PyTorch 2.11",
                 fontsize=13, fontweight="bold")

    labels = [s[0] for s in stages]
    times = [s[1] for s in stages]
    colors = [s[2] for s in stages]
    fps = [1000.0 / t for t in times]

    y = np.arange(len(stages))
    bars = ax.barh(y, times, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("median time per call, ms")
    ax.set_xscale("log")
    ax.invert_yaxis()
    for yi, t, f in zip(y, times, fps):
        ax.text(t * 1.1, yi, f"{t:.1f} ms ({f:.0f} FPS)", va="center", fontsize=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, which="both")

    plt.subplots_adjust(left=0.32, right=0.95)
    out = Path("docs/figures/phase4/fig_runtime_fps.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Wrote {out}")


def main():
    fig_5seqs_bar()
    plt.close()
    fig_drift_vs_map()
    plt.close()
    fig_runtime()
    plt.close()


if __name__ == "__main__":
    main()
