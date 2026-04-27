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

CLASS_NAMES = {
    0: "background",
    1: "water",
    2: "vegetation",
    3: "buildings",
    4: "roads_runway_taxiway",
}


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in PALETTE.items():
        out[mask == class_id] = color
    return out


def render_overlay(image_bgr: np.ndarray, mask_gray: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    colored = colorize_mask(mask_gray)
    return cv2.addWeighted(image_bgr, 1.0 - alpha, colored, alpha, 0)


def draw_legend(canvas: np.ndarray, start_x: int, start_y: int) -> None:
    y = start_y
    for class_id in sorted(CLASS_NAMES.keys()):
        color = PALETTE[class_id]
        cv2.rectangle(canvas, (start_x, y - 12), (start_x + 20, y + 8), color, -1)
        text = f"{class_id}: {CLASS_NAMES[class_id]}"
        cv2.putText(canvas, text, (start_x + 30, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        y += 24


def save_preview_grid(dataset_root: Path, out_dir: Path, samples: int = 16) -> None:
    img_dir = dataset_root / "images" / "tiles"
    mask_dir = dataset_root / "masks" / "tiles"
    out_dir.mkdir(parents=True, exist_ok=True)

    tile_names = sorted([p.name for p in img_dir.glob("*.png") if (mask_dir / p.name).exists()])
    if not tile_names:
        raise RuntimeError("No image/mask pairs found")

    scored: list[tuple[float, str]] = []
    for name in tile_names:
        mask = cv2.imread(str(mask_dir / name), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        vals, cnts = np.unique(mask, return_counts=True)
        hist = {int(v): int(c) for v, c in zip(vals, cnts)}
        total = int(mask.size)
        non_bg = 1.0 - (hist.get(0, 0) / max(1, total))
        classes_non_bg = sum(1 for k in hist.keys() if k > 0 and hist[k] > 0)

        # Балансируем между разнообразием классов и адекватной долей foreground,
        # чтобы мозаика не состояла из пустых/монотонных тайлов.
        score = classes_non_bg * 2.0 - abs(non_bg - 0.45)
        scored.append((score, name))

    scored.sort(reverse=True)
    chosen = [name for _, name in scored[: min(samples, len(scored))]]

    rows = []
    for name in chosen:
        img = cv2.imread(str(img_dir / name), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_dir / name), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue

        overlay = render_overlay(img, mask)
        color_mask = colorize_mask(mask)
        row = np.hstack([img, color_mask, overlay])
        rows.append(row)

        cv2.imwrite(str(out_dir / f"preview_{name}"), row)

    if rows:
        mosaic = np.vstack(rows)
        panel = np.full((mosaic.shape[0], 320, 3), 25, dtype=np.uint8)
        draw_legend(panel, 12, 24)
        full = np.hstack([mosaic, panel])
        cv2.imwrite(str(out_dir / "preview_mosaic.png"), full)


def print_stats(dataset_root: Path) -> None:
    mask_dir = dataset_root / "masks" / "tiles"
    patch_mask_dir = dataset_root / "masks" / "patches_512"

    hist_tiles = np.zeros(256, dtype=np.int64)
    hist_patches = np.zeros(256, dtype=np.int64)

    for p in mask_dir.glob("*.png"):
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        vals, cnts = np.unique(m, return_counts=True)
        hist_tiles[vals] += cnts

    for p in patch_mask_dir.glob("*.png"):
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        vals, cnts = np.unique(m, return_counts=True)
        hist_patches[vals] += cnts

    print("Tile class histogram:")
    for class_id in sorted(CLASS_NAMES):
        print(class_id, CLASS_NAMES[class_id], int(hist_tiles[class_id]))

    total = int(hist_tiles.sum())
    fg = int(hist_tiles[1:].sum())
    print("tile_foreground_ratio", round(fg / max(1, total), 6))

    if int(hist_patches.sum()) > 0:
        print("Patch class histogram:")
        for class_id in sorted(CLASS_NAMES):
            print(class_id, CLASS_NAMES[class_id], int(hist_patches[class_id]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual and numeric validation for Overture dataset")
    parser.add_argument("--dataset-root", required=True, help="Path to dataset root")
    parser.add_argument("--samples", type=int, default=16, help="How many random tiles to preview")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    root = Path(args.dataset_root)
    out_dir = root / "quality_preview"

    print_stats(root)
    save_preview_grid(root, out_dir, samples=args.samples)
    print("preview_dir", out_dir)


if __name__ == "__main__":
    main()
