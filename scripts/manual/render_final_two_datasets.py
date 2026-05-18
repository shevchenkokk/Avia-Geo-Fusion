"""Финальные thesis-figures для главы «Эксперимент»: AerialVL ablation +
VPair semantic re-rank — две независимых валидации.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---- AerialVL ----
SEQS = [
    ("s/03-11", "short_trajtr_2023-03-11-11-48-35"),
    ("s/03-16", "short_trajtr_2023-03-16-16-58-43"),
    ("s/03-18a", "short_trajtr_2023-03-18-16-43-16"),
    ("l/03-18a", "long_trajtr_2023-03-18-14-38-32"),
    ("l/03-18b", "long_trajtr_2023-03-18-15-01-14"),
]
CONFIGS_AVL = [
    ("baseline", "baseline", "#888888"),
    ("sem_mask\n(filter)", "sem_mask", "#3366cc"),
    ("sem_struct\n(channel)", "sem_struct", "#cc8833"),
    ("sem_both\n(filter+channel)", "sem_both", "#2c8c2c"),
]


def fig_aerialvl():
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle("AerialVL ablation: 5 sequences × 4 configs (median position error)",
                 fontsize=13, fontweight="bold")
    n_seqs = len(SEQS)
    width = 0.20
    x = np.arange(n_seqs)
    for i, (cfg_label, cfg_tag, color) in enumerate(CONFIGS_AVL):
        vals = []
        for seq_label, seq_tag in SEQS:
            p = Path(f"results/aerialvl_honest_ablation/{seq_tag}__{cfg_tag}/summary.json")
            vals.append(json.loads(p.read_text())["median_error_m"] if p.exists() else 0)
        offset = (i - 1.5) * width
        ax.bar(x + offset, vals, width=width, label=cfg_label, color=color,
               edgecolor="black", linewidth=0.6)
        for xi, v in zip(x + offset, vals):
            ax.text(xi, v + 8, f"{v:.0f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in SEQS])
    ax.set_ylabel("median position error, м")
    ax.set_xlabel("sequence")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(50, color="green", linestyle="--", alpha=0.4, label="target 30-50м")
    ax.text(-0.45, 55, "target ≤ 50м", color="green", fontsize=9)

    # Aggregate annotation
    means = []
    for _, cfg_tag, _ in CONFIGS_AVL:
        v = [json.loads(Path(f"results/aerialvl_honest_ablation/{seq_tag}__{cfg_tag}/summary.json").read_text())["median_error_m"]
             for _, seq_tag in SEQS]
        means.append(sum(v) / len(v))
    annotation = (
        f"Mean median across 5 sequences:\n"
        f"  baseline = {means[0]:.0f}м\n"
        f"  sem_mask = {means[1]:.0f}м ({(means[1]/means[0]-1)*100:+.0f}%)\n"
        f"  sem_struct = {means[2]:.0f}м ({(means[2]/means[0]-1)*100:+.0f}%)\n"
        f"  sem_both = {means[3]:.0f}м ({(means[3]/means[0]-1)*100:+.0f}%)"
    )
    ax.text(0.99, 0.97, annotation, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.92))

    out = Path("docs/figures/phase4/fig_aerialvl_final_4configs.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Wrote {out}")


def fig_vpair():
    s = json.loads(Path("results/vpair_semantic_rerank_full/rerank_summary.json").read_text())
    radii = [25, 50, 100, 200]
    dinov2 = [s["recall_R1"][f"R@1_{r}m"]["dinov2"] for r in radii]
    rerank = [s["recall_R1"][f"R@1_{r}m"]["rerank"] for r in radii]
    delta = [s["recall_R1"][f"R@1_{r}m"]["delta_pp"] for r in radii]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.suptitle("VPair: semantic re-ranking поднимает Recall@1 на всех радиусах (199 queries)",
                 fontsize=13, fontweight="bold")
    width = 0.35
    x = np.arange(len(radii))
    bars1 = ax.bar(x - width/2, dinov2, width, label="DINOv2 only (baseline)",
                    color="#888888", edgecolor="black", linewidth=0.6)
    bars2 = ax.bar(x + width/2, rerank, width, label="+ semantic re-rank (Phase 4)",
                    color="#2c8c2c", edgecolor="black", linewidth=0.6)
    for xi, v in zip(x - width/2, dinov2):
        ax.text(xi, v + 1.5, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
    for xi, v, d in zip(x + width/2, rerank, delta):
        ax.text(xi, v + 1.5, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
        ax.text(xi, v + 6, f"(+{d:.1f}pp)", ha="center", fontsize=9, color="#2c8c2c")

    ax.set_xticks(x)
    ax.set_xticklabels([f"R@1@{r}m" for r in radii])
    ax.set_ylabel("Recall@1, %")
    ax.set_xlabel("radius threshold")
    ax.set_ylim(0, 80)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    m = s.get("median_top1_dist_m", {})
    annotation = (
        f"VPair dataset: 199 queries × 399 gallery (199 ref + 200 distractors)\n"
        f"Cologne/Bonn, Germany (cross-domain — модель училась на России)\n"
        f"\n"
        f"Median top-1 distance:\n"
        f"  DINOv2 only:  {m['dinov2']:.0f} м\n"
        f"  + sem rerank: {m['rerank']:.0f} м  ({m['rerank']-m['dinov2']:+.0f} м)"
    )
    ax.text(0.99, 0.40, annotation, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.92))

    out = Path("docs/figures/phase4/fig_vpair_semantic_rerank.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    fig_aerialvl()
    plt.close()
    fig_vpair()
    plt.close()
