"""Быстрый визуальный смок: рендерим N патчей по регионам, чтобы убедиться,
что после фильтра+регенерации patches не вернулись монокультуры."""

from __future__ import annotations

import argparse
import csv
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


def colorize(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, c in PALETTE.items():
        out[mask == cid] = c
    return out


def overlay(img: np.ndarray, mask: np.ndarray, a: float = 0.45) -> np.ndarray:
    return cv2.addWeighted(img, 1 - a, colorize(mask), a, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--per-region", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    root = Path(args.dataset_root)
    by_region: dict[str, list] = defaultdict(list)
    with (root / "meta_patches.csv").open() as f:
        for row in csv.DictReader(f):
            by_region[row["region_id"]].append(row)

    rows = []
    for reg in sorted(by_region):
        chosen = random.sample(by_region[reg], min(args.per_region, len(by_region[reg])))
        for r in chosen:
            img = cv2.imread(str(root / r["image_path"]))
            mask = cv2.imread(str(root / r["mask_path"]), cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue
            row = np.hstack([img, colorize(mask), overlay(img, mask)])
            label = f"{reg}  {r['patch_id']}  classes={r['classes_present']}"
            cv2.putText(row, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            rows.append(row)

    if not rows:
        print("No patches rendered")
        return

    max_w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.full((r.shape[0], max_w - r.shape[1], 3), 30, dtype=np.uint8)
            r = np.hstack([r, pad])
        padded.append(r)
    cv2.imwrite(args.output, np.vstack(padded))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
