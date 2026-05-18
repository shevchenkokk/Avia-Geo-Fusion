"""Сравниваем старые маски (бэкап) и новые после Phase B (UTM + class-aware
буфер + drop sidewalks). Рендерим: image | mask_old | mask_new | overlay_old | overlay_new"""

from __future__ import annotations

import argparse
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
    for cid, c in PALETTE.items():
        out[mask == cid] = c
    return out


def overlay(img: np.ndarray, mask: np.ndarray, a: float = 0.45) -> np.ndarray:
    return cv2.addWeighted(img, 1 - a, colorize(mask), a, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--region", default="moscow_city_small")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    root = Path(args.dataset_root)
    img_dir = root / "images" / "tiles"
    new_mask_dir = root / "masks" / "tiles"
    old_mask_dir = root / "masks" / "tiles.old_buffer_backup"

    if not old_mask_dir.exists():
        raise SystemExit(f"Need old backup at {old_mask_dir}")

    candidates = sorted(p.name for p in img_dir.glob(f"{args.region}_*.png")
                        if (new_mask_dir / p.name).exists() and (old_mask_dir / p.name).exists())
    chosen = random.sample(candidates, min(args.samples, len(candidates)))

    rows = []
    for name in chosen:
        img = cv2.imread(str(img_dir / name))
        mold = cv2.imread(str(old_mask_dir / name), cv2.IMREAD_GRAYSCALE)
        mnew = cv2.imread(str(new_mask_dir / name), cv2.IMREAD_GRAYSCALE)
        if img is None or mold is None or mnew is None:
            continue
        old_road = float((mold == 4).mean()) * 100
        new_road = float((mnew == 4).mean()) * 100
        row = np.hstack([img, colorize(mold), colorize(mnew), overlay(img, mold), overlay(img, mnew)])
        label = f"{name}  road: old={old_road:5.1f}%  new={new_road:5.1f}%"
        cv2.putText(row, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(row)

    if not rows:
        print("Nothing rendered")
        return

    head_h = 28
    head = np.full((head_h, rows[0].shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(head, "image | mask_old | mask_new | overlay_old | overlay_new",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(args.output, np.vstack([head, *rows]))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
