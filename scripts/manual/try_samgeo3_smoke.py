"""SamGeo3 (SAM3 + transformers) smoke на одном тайле.
Промпты: water, vegetation, buildings, roads. Рендерим композит.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch


PROMPTS = {
    1: ("water", (255, 80, 0)),
    2: ("tree", (40, 170, 40)),
    3: ("building", (220, 220, 220)),
    4: ("road", (70, 70, 230)),
}


def colorize(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, (_, color) in PROMPTS.items():
        out[mask == cid] = color
    return out


def overlay_img(img: np.ndarray, mask: np.ndarray, a: float = 0.45) -> np.ndarray:
    return cv2.addWeighted(img, 1 - a, colorize(mask), a, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--model-id", default="facebook/sam3")
    args = ap.parse_args()

    print(f"device={args.device}  model_id={args.model_id}")
    print("loading SamGeo3...")
    from samgeo.samgeo3 import SamGeo3
    t0 = time.time()
    sam = SamGeo3(
        model_id=args.model_id,
        device=args.device,
        backend="transformers",
        confidence_threshold=0.30,
        mask_threshold=0.40,
    )
    print(f"  loaded in {time.time()-t0:.1f}s")

    img_path = Path(args.image)
    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit(f"failed to read {img_path}")
    h, w = img.shape[:2]
    print(f"image {img_path.name}  {w}x{h}")

    sam.set_image(str(img_path))

    composite = np.zeros((h, w), dtype=np.uint8)
    per_class_panels = []
    paint_order = [2, 3, 1, 4]
    for cid in paint_order:
        text, _ = PROMPTS[cid]
        t0 = time.time()
        try:
            sam.generate_masks(prompt=text, quiet=True)
        except Exception as e:
            print(f"  '{text}' FAILED: {type(e).__name__}: {e}")
            per_class_panels.append((cid, np.zeros_like(img)))
            continue
        masks = getattr(sam, "masks", None)
        if masks is None:
            mask_bool = np.zeros((h, w), dtype=bool)
        else:
            arr = np.asarray(masks)
            if arr.ndim == 3 and arr.shape[0] > 0:
                mask_bool = arr.any(axis=0).astype(bool)
            elif arr.ndim == 2:
                mask_bool = arr.astype(bool)
            else:
                mask_bool = np.zeros((h, w), dtype=bool)
        composite[mask_bool] = cid
        single = np.zeros_like(composite)
        single[mask_bool] = cid
        per_class_panels.append((cid, overlay_img(img, single)))
        print(f"  '{text}' {time.time()-t0:.1f}s  pixels={int(mask_bool.sum())}")

    full_overlay = overlay_img(img, composite)
    panels = [img, full_overlay] + [p for _, p in per_class_panels]
    row = np.hstack(panels)
    head = np.full((26, row.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(head, "image | composite | water | vegetation | buildings | roads",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    out = np.vstack([head, row])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
