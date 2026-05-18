"""Аудит выравнивания image/mask пар в Overture-RU датасете.

Старый qc_report.json пометил 5405/6143 тайлов как `likely_shifted`, но
без измеренной величины смещения и со score=0.0 на каждом примере. Скорее
всего, флаг ставился на основе отсутствия структуры (лес, поля), а не
реального шифта. Этот скрипт меряет смещение напрямую через phase
correlation между edge-картой изображения и edge-картой маски и решает:
шифт систематический, случайный или фантомный.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


PALETTE = {
    0: (0, 0, 0),
    1: (255, 80, 0),
    2: (40, 170, 40),
    3: (220, 220, 220),
    4: (70, 70, 230),
}


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in PALETTE.items():
        out[mask == class_id] = color
    return out


def mask_edges(mask: np.ndarray) -> np.ndarray:
    """Бинарная карта границ между классами в маске."""
    grad_x = cv2.Sobel(mask.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(mask.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    edges = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    return (edges > 0).astype(np.float32)


def image_edges(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges.astype(np.float32) / 255.0


def hann_window(h: int, w: int) -> np.ndarray:
    wy = np.hanning(h)
    wx = np.hanning(w)
    return np.outer(wy, wx).astype(np.float32)


def estimate_shift(img_edges: np.ndarray, mk_edges: np.ndarray) -> tuple[float, float, float]:
    """Возвращает (dx, dy, response). dx>0 — маска смещена вправо относительно изображения."""
    h, w = img_edges.shape
    win = hann_window(h, w)
    a = img_edges * win
    b = mk_edges * win
    if a.std() < 1e-6 or b.std() < 1e-6:
        return float("nan"), float("nan"), 0.0
    (dx, dy), response = cv2.phaseCorrelate(a, b)
    return float(dx), float(dy), float(response)


def region_of(tile_id: str) -> str:
    return tile_id.split("_17_")[0]


def render_overlay(img: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return cv2.addWeighted(img, 1.0 - alpha, colorize_mask(mask), alpha, 0)


def shift_mask_visual(img: np.ndarray, mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Сдвигает маску на (-dx, -dy), чтобы показать, как выглядел бы align."""
    h, w = mask.shape
    M = np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float32)
    shifted = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return shifted


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--samples-per-region", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--worst-n", type=int, default=24, help="Worst tiles by shift magnitude to render")
    p.add_argument("--shift-threshold-px", type=float, default=2.0,
                   help="Tiles with |shift| above this are reported as truly shifted")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    root = Path(args.dataset_root)
    img_dir = root / "images" / "tiles"
    mask_dir = root / "masks" / "tiles"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = sorted(p.name for p in img_dir.glob("*.png") if (mask_dir / p.name).exists())
    by_region: dict[str, list[str]] = defaultdict(list)
    for name in all_pairs:
        by_region[region_of(name.replace(".png", ""))].append(name)

    sampled: list[str] = []
    for region, names in by_region.items():
        random.shuffle(names)
        sampled.extend(names[: args.samples_per_region])
    print(f"Sampled {len(sampled)} tiles across {len(by_region)} regions")

    rows = []
    for i, name in enumerate(sampled):
        img = cv2.imread(str(img_dir / name), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_dir / name), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        if img.shape[:2] != mask.shape[:2]:
            continue

        ie = image_edges(img)
        me = mask_edges(mask)
        fg_share = float((mask > 0).mean())
        edge_share = float(me.mean())

        if edge_share < 0.005 or fg_share < 0.02:
            rows.append({
                "tile_id": name.replace(".png", ""),
                "region": region_of(name.replace(".png", "")),
                "dx": float("nan"),
                "dy": float("nan"),
                "shift_mag": float("nan"),
                "response": 0.0,
                "fg_share": fg_share,
                "edge_share": edge_share,
                "status": "no_signal",
            })
            continue

        dx, dy, resp = estimate_shift(ie, me)
        mag = float(np.hypot(dx, dy)) if np.isfinite(dx) and np.isfinite(dy) else float("nan")
        rows.append({
            "tile_id": name.replace(".png", ""),
            "region": region_of(name.replace(".png", "")),
            "dx": dx,
            "dy": dy,
            "shift_mag": mag,
            "response": resp,
            "fg_share": fg_share,
            "edge_share": edge_share,
            "status": "ok",
        })

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(sampled)} processed")

    csv_path = out_dir / "shift_per_tile.csv"
    with csv_path.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"Wrote {csv_path}")

    measured = [r for r in rows if r["status"] == "ok" and np.isfinite(r["shift_mag"])]
    no_signal = [r for r in rows if r["status"] == "no_signal"]

    summary = {
        "total_sampled": len(rows),
        "measured": len(measured),
        "no_signal_skipped": len(no_signal),
        "shift_threshold_px": args.shift_threshold_px,
        "global": {},
        "per_region": {},
    }

    if measured:
        mags = np.array([r["shift_mag"] for r in measured])
        dxs = np.array([r["dx"] for r in measured])
        dys = np.array([r["dy"] for r in measured])
        summary["global"] = {
            "mag_median": float(np.median(mags)),
            "mag_p90": float(np.percentile(mags, 90)),
            "mag_p99": float(np.percentile(mags, 99)),
            "dx_median": float(np.median(dxs)),
            "dy_median": float(np.median(dys)),
            "dx_iqr": [float(np.percentile(dxs, 25)), float(np.percentile(dxs, 75))],
            "dy_iqr": [float(np.percentile(dys, 25)), float(np.percentile(dys, 75))],
            "above_threshold_share": float((mags > args.shift_threshold_px).mean()),
        }

        per_reg = defaultdict(list)
        for r in measured:
            per_reg[r["region"]].append(r)
        for reg, items in per_reg.items():
            mags_r = np.array([x["shift_mag"] for x in items])
            dxs_r = np.array([x["dx"] for x in items])
            dys_r = np.array([x["dy"] for x in items])
            summary["per_region"][reg] = {
                "n": len(items),
                "mag_median": float(np.median(mags_r)),
                "mag_p90": float(np.percentile(mags_r, 90)),
                "dx_median": float(np.median(dxs_r)),
                "dy_median": float(np.median(dys_r)),
                "above_threshold_share": float((mags_r > args.shift_threshold_px).mean()),
            }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")

    if measured:
        worst = sorted(measured, key=lambda r: r["shift_mag"], reverse=True)[: args.worst_n]
        panels = []
        for r in worst:
            name = r["tile_id"] + ".png"
            img = cv2.imread(str(img_dir / name), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_dir / name), cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue
            overlay = render_overlay(img, mask)
            corrected = shift_mask_visual(img, mask, r["dx"], r["dy"])
            corrected_overlay = render_overlay(img, corrected)
            row = np.hstack([img, overlay, corrected_overlay])
            label = f"{r['tile_id']}  dx={r['dx']:+.2f} dy={r['dy']:+.2f} mag={r['shift_mag']:.2f}px"
            cv2.putText(row, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(row)
        if panels:
            mosaic = np.vstack(panels)
            cv2.imwrite(str(out_dir / "worst_overlays.png"), mosaic)
            print(f"Wrote {out_dir / 'worst_overlays.png'} ({len(panels)} rows)")

    print("\nGlobal summary:")
    print(json.dumps(summary["global"], indent=2))
    print("\nPer-region summary:")
    print(json.dumps(summary["per_region"], indent=2))


if __name__ == "__main__":
    main()
