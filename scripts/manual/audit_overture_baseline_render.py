"""Сравнение baseline тайлов moscow_city (флаг 'shifted' стоит у 21/759, 3%)
с тайлами, попавшими в qc_report.likely_shifted, чтобы глазом убедиться:
проблема в маске (smeared rasterization), а не в трансляции."""

from __future__ import annotations

import argparse
import json
import random
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


def colorize(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in PALETTE.items():
        out[mask == cid] = color
    return out


def overlay(img: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return cv2.addWeighted(img, 1.0 - alpha, colorize(mask), alpha, 0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-good", type=int, default=12)
    p.add_argument("--n-bad", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def label_strip(width: int, text: str) -> np.ndarray:
    strip = np.full((22, width, 3), 30, dtype=np.uint8)
    cv2.putText(strip, text, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    return strip


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    root = Path(args.dataset_root)
    img_dir = root / "images" / "tiles"
    mask_dir = root / "masks" / "tiles"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qc = json.loads((root / "qc_report.json").read_text())
    bad_ids = {x["tile_id"] for x in qc.get("likely_shifted", [])}
    smeared_ids = {x["tile_id"] for x in qc.get("smeared_tiles", [])}

    all_pairs = sorted(p.name for p in img_dir.glob("*.png") if (mask_dir / p.name).exists())

    moscow_city = [p for p in all_pairs if p.startswith("moscow_city_small_17_")]
    good = [p for p in moscow_city if p.replace(".png", "") not in bad_ids and p.replace(".png", "") not in smeared_ids]

    bad = [p for p in all_pairs if p.replace(".png", "") in smeared_ids]

    random.shuffle(good)
    random.shuffle(bad)

    good_panel = render_panel(img_dir, mask_dir, good[: args.n_good], "GOOD (moscow_city, not flagged)")
    bad_panel = render_panel(img_dir, mask_dir, bad[: args.n_bad], "BAD (smeared in qc_report)")

    if good_panel is None or bad_panel is None:
        print("Failed to render panels")
        return

    max_w = max(good_panel.shape[1], bad_panel.shape[1])
    if good_panel.shape[1] < max_w:
        pad = np.full((good_panel.shape[0], max_w - good_panel.shape[1], 3), 30, dtype=np.uint8)
        good_panel = np.hstack([good_panel, pad])
    if bad_panel.shape[1] < max_w:
        pad = np.full((bad_panel.shape[0], max_w - bad_panel.shape[1], 3), 30, dtype=np.uint8)
        bad_panel = np.hstack([bad_panel, pad])

    out = np.vstack([good_panel, bad_panel])
    out_path = out_dir / "good_vs_bad.png"
    cv2.imwrite(str(out_path), out)
    print(f"Wrote {out_path}")


def render_panel(img_dir: Path, mask_dir: Path, names: list[str], header: str) -> np.ndarray | None:
    rows = []
    for name in names:
        img = cv2.imread(str(img_dir / name), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_dir / name), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        row = np.hstack([img, colorize(mask), overlay(img, mask)])
        cv2.putText(row, name.replace(".png", ""), (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(row)
    if not rows:
        return None
    body = np.vstack(rows)
    head = label_strip(body.shape[1], header)
    return np.vstack([head, body])


if __name__ == "__main__":
    main()
