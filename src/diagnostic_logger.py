"""Покадровый диагностический логгер пайплайна локализации.

Логгер записывает одну строку на каждый обработанный кадр в CSV и в конце
запуска строит сводный график из четырёх панелей. Формат подобран так, чтобы
по графику быстро было видно, какой из описанных режимов отказа
(см. PROJECT_PLAN.md §1) доминирует на конкретном пролёте.

Пример использования::

    logger = DiagnosticLogger(output_dir=Path("results/diag/run_001"))
    for frame_idx, ...:
        logger.log_frame(FrameDiagnostic(frame_idx=..., ...))
    logger.close()  # сбрасывает CSV, строит графики, выводит сводку
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np


_FIELDS: tuple[str, ...] = (
    "frame_idx",
    "timestamp_seconds",
    "wallclock_ms",
    "backend",
    "num_keypoints_frame",
    "num_keypoints_tile",
    "num_matches",
    "num_inliers",
    "inlier_ratio",
    "reprojection_error_px",
    "homography_scale",
    "confidence",
    "state",
    "reject_reason",
    "raw_lat",
    "raw_lon",
    "kalman_lat",
    "kalman_lon",
    "predicted_lat",
    "predicted_lon",
    "tile_id",
    "tile_changed",
    "is_keyframe",
    "matcher_ms",
    "geolocator_ms",
    "obstructed",
    "obstr_std",
    "obstr_entropy",
    "obstr_edge_density",
    "obstr_low_var_patch_frac",
    "obstr_votes",
)


@dataclass
class FrameDiagnostic:
    """Одна строка диагностического CSV. Необязательные поля по умолчанию NaN/None."""

    frame_idx: int
    timestamp_seconds: float
    wallclock_ms: float = float("nan")
    backend: str = ""
    num_keypoints_frame: int = -1
    num_keypoints_tile: int = -1
    num_matches: int = -1
    num_inliers: int = -1
    inlier_ratio: float = float("nan")
    reprojection_error_px: float = float("nan")
    homography_scale: float = float("nan")
    confidence: float = float("nan")
    state: str = ""
    reject_reason: str = ""
    raw_lat: float = float("nan")
    raw_lon: float = float("nan")
    kalman_lat: float = float("nan")
    kalman_lon: float = float("nan")
    predicted_lat: float = float("nan")
    predicted_lon: float = float("nan")
    tile_id: str = ""
    tile_changed: bool = False
    is_keyframe: bool = False
    matcher_ms: float = float("nan")
    geolocator_ms: float = float("nan")
    obstructed: bool = False
    obstr_std: float = float("nan")
    obstr_entropy: float = float("nan")
    obstr_edge_density: float = float("nan")
    obstr_low_var_patch_frac: float = float("nan")
    obstr_votes: int = 0

    def to_row(self) -> dict[str, Any]:
        return {k: asdict(self)[k] for k in _FIELDS}


class DiagnosticLogger:
    """Логгер CSV и сводных графиков для пайплайна локализации."""

    def __init__(self, output_dir: Path, run_name: str | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if run_name is None:
            run_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name
        self.csv_path = self.output_dir / f"frames_{run_name}.csv"
        self._fp = self.csv_path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._fp, fieldnames=_FIELDS)
        self._writer.writeheader()
        self._rows: list[dict[str, Any]] = []

    def log_frame(self, diag: FrameDiagnostic) -> None:
        row = diag.to_row()
        self._writer.writerow(row)
        self._rows.append(row)

    def close(self) -> dict[str, Any]:
        self._fp.flush()
        self._fp.close()
        summary = self._summary()
        self._render_plot(summary)
        self._render_state_timeline()
        return summary

    def _summary(self) -> dict[str, Any]:
        if not self._rows:
            return {"frames": 0}
        n = len(self._rows)
        states = [r["state"] for r in self._rows]
        rejects = [r["reject_reason"] for r in self._rows if r["reject_reason"]]
        inliers = np.array(
            [r["num_inliers"] if r["num_inliers"] >= 0 else 0 for r in self._rows], dtype=float
        )
        ratios = np.array(
            [r["inlier_ratio"] for r in self._rows], dtype=float
        )
        ratios = ratios[np.isfinite(ratios)]
        reproj = np.array([r["reprojection_error_px"] for r in self._rows], dtype=float)
        reproj = reproj[np.isfinite(reproj)]

        state_counts = {s: states.count(s) for s in set(states)}
        reject_counts: dict[str, int] = {}
        for r in rejects:
            reject_counts[r] = reject_counts.get(r, 0) + 1

        summary = {
            "frames": n,
            "state_counts": state_counts,
            "reject_counts": reject_counts,
            "track_pct": 100.0 * state_counts.get("TRACK", 0) / max(1, n),
            "weak_pct": 100.0 * state_counts.get("WEAK", 0) / max(1, n),
            "relocalize_pct": 100.0 * state_counts.get("RELOCALIZE", 0) / max(1, n),
            "median_inliers": float(np.median(inliers)) if inliers.size else 0.0,
            "median_inlier_ratio": float(np.median(ratios)) if ratios.size else 0.0,
            "median_reproj_px": float(np.median(reproj)) if reproj.size else 0.0,
            "tile_changes": sum(1 for r in self._rows if r["tile_changed"]),
            "csv": str(self.csv_path),
        }
        return summary

    def _render_plot(self, summary: dict[str, Any]) -> None:
        if not self._rows:
            return
        t = np.array([r["timestamp_seconds"] for r in self._rows], dtype=float)
        inliers = np.array(
            [r["num_inliers"] if r["num_inliers"] >= 0 else 0 for r in self._rows], dtype=float
        )
        ratio = np.array([r["inlier_ratio"] for r in self._rows], dtype=float)
        reproj = np.array([r["reprojection_error_px"] for r in self._rows], dtype=float)
        scale = np.array([r["homography_scale"] for r in self._rows], dtype=float)

        states = [r["state"] for r in self._rows]
        track_mask = np.array([s == "TRACK" for s in states])
        weak_mask = np.array([s == "WEAK" for s in states])
        reloc_mask = np.array([s == "RELOCALIZE" for s in states])

        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

        ax = axes[0]
        ax.plot(t, inliers, color="tab:blue", linewidth=0.7, alpha=0.5, label="inliers")
        if track_mask.any():
            ax.scatter(t[track_mask], inliers[track_mask], s=4, c="tab:green", alpha=0.6, label="TRACK")
        if weak_mask.any():
            ax.scatter(t[weak_mask], inliers[weak_mask], s=4, c="tab:orange", alpha=0.6, label="WEAK")
        if reloc_mask.any():
            ax.scatter(t[reloc_mask], inliers[reloc_mask], s=6, c="tab:red", alpha=0.7, label="RELOCALIZE")
        ax.axhline(8, color="black", linestyle=":", linewidth=0.6, alpha=0.5)
        ax.set_ylabel("num_inliers")
        ax.set_title(
            f"Run {self.run_name} — TRACK {summary.get('track_pct', 0):.1f}%  "
            f"WEAK {summary.get('weak_pct', 0):.1f}%  "
            f"RELOC {summary.get('relocalize_pct', 0):.1f}%"
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1]
        ax.plot(t, ratio, color="tab:purple", linewidth=0.7)
        ax.axhline(0.15, color="black", linestyle=":", linewidth=0.6, alpha=0.5,
                   label="reject threshold (0.15)")
        ax.set_ylabel("inlier_ratio")
        ax.set_ylim(0, 1.0)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[2]
        finite = np.isfinite(reproj)
        if finite.any():
            ax.plot(t[finite], reproj[finite], color="tab:red", linewidth=0.7)
        ax.set_ylabel("reproj err (px)")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)

        ax = axes[3]
        finite_s = np.isfinite(scale)
        if finite_s.any():
            ax.plot(t[finite_s], scale[finite_s], color="tab:cyan", linewidth=0.7)
        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.6, alpha=0.5)
        ax.set_ylabel("homography scale")
        ax.set_xlabel("time (s)")
        ax.grid(alpha=0.3)

        fig.tight_layout()
        plot_path = self.output_dir / f"summary_{self.run_name}.png"
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        summary["plot"] = str(plot_path)

    def _render_state_timeline(self) -> None:
        """Второй график: временная шкала состояния и причин отбраковки.

        Нужен, чтобы основной источник сбоев был виден сразу.
        """
        if not self._rows:
            return

        t = np.array([r["timestamp_seconds"] for r in self._rows], dtype=float)
        states = [r["state"] for r in self._rows]
        rejects = [r["reject_reason"] for r in self._rows]

        fig, axes = plt.subplots(2, 1, figsize=(14, 4), sharex=True)

        ax = axes[0]
        colour_map = {"TRACK": "tab:green", "WEAK": "tab:orange", "RELOCALIZE": "tab:red"}
        for s, c in colour_map.items():
            mask = np.array([x == s for x in states])
            if mask.any():
                ax.scatter(t[mask], np.zeros(mask.sum()) + 0.5, s=8, c=c, label=s)
        ax.set_yticks([])
        ax.set_ylim(0, 1)
        ax.set_title("State timeline")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3, axis="x")

        ax = axes[1]
        unique_rejects = sorted(set(r for r in rejects if r))
        if unique_rejects:
            for i, reason in enumerate(unique_rejects):
                mask = np.array([r == reason for r in rejects])
                if mask.any():
                    ax.scatter(t[mask], np.full(mask.sum(), i), s=8, label=reason)
            ax.set_yticks(range(len(unique_rejects)))
            ax.set_yticklabels(unique_rejects, fontsize=8)
        else:
            ax.text(0.5, 0.5, "no rejects", transform=ax.transAxes, ha="center")
        ax.set_xlabel("time (s)")
        ax.set_title("Reject reasons over time")
        ax.grid(alpha=0.3, axis="x")

        fig.tight_layout()
        plot_path = self.output_dir / f"state_timeline_{self.run_name}.png"
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)


def classify_state(num_inliers: int, inlier_ratio: float, reject_reason: str) -> str:
    """Единое место, где определяется состояние трека."""
    if reject_reason or num_inliers <= 0:
        return "RELOCALIZE"
    if num_inliers >= 30 and inlier_ratio >= 0.30:
        return "TRACK"
    return "WEAK"


def homography_scale(H: np.ndarray | None) -> float:
    """Средний геометрический масштаб из 2x2 affine-блока H.

    Используется как грубая проверка: внезапное изменение в 2 раза между
    соседними успешными кадрами обычно означает, что матчер зацепился за
    неправильный участок тайла, например за группу зданий вместо реального
    вида с самолёта.
    """
    if H is None:
        return float("nan")
    a = np.asarray(H, dtype=float)[:2, :2]
    det = float(np.linalg.det(a))
    if det <= 0:
        return float("nan")
    return float(np.sqrt(det))


def reprojection_error(
    pts0: np.ndarray, pts1: np.ndarray, H: np.ndarray | None
) -> float:
    """Среднее L2-расстояние после применения H к pts0 относительно pts1, в пикселях."""
    if H is None or len(pts0) == 0:
        return float("nan")
    src = np.asarray(pts0, dtype=np.float64).reshape(-1, 1, 2)
    try:
        import cv2

        warped = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    except Exception:
        return float("nan")
    diffs = warped - np.asarray(pts1, dtype=np.float64).reshape(-1, 2)
    return float(np.sqrt((diffs ** 2).sum(axis=1)).mean())


def confidence_from_inliers(num_inliers: int, inlier_ratio: float, reproj_px: float) -> float:
    """Эвристическая уверенность в диапазоне [0, 1].

    Сочетает насыщаемую функцию от числа inlier-точек, сам inlier ratio и
    мягкий штраф за большую ошибку репроекции. Это не байесовская вероятность,
    а одно число для примерного ранжирования кадров по надёжности измерения;
    дальше оно удобно для ворот фильтра.
    """
    if num_inliers <= 0 or not np.isfinite(inlier_ratio):
        return 0.0
    inl = 1.0 - np.exp(-num_inliers / 30.0)
    ratio = float(np.clip(inlier_ratio, 0.0, 1.0))
    reproj_term = 1.0
    if np.isfinite(reproj_px) and reproj_px > 0:
        reproj_term = float(np.clip(1.0 - (reproj_px - 2.0) / 8.0, 0.2, 1.0))
    return float(np.clip(inl * (0.4 + 0.6 * ratio) * reproj_term, 0.0, 1.0))
