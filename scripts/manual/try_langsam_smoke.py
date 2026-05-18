"""LangSAM (SAM2 + GroundingDINO) smoke на одном тайле.
Текст-промпты: buildings, roads, water, vegetation. Сохраняет визуализацию.
Если работает — масштабируем до полного сравнения с SegFormer/Overture.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def _patch_transformers_for_groundingdino() -> None:
    """transformers 5.x удалил BertModel.get_head_mask, который groundingdino
    оборачивает в BertModelWarper. Возвращаем заглушку — для нашего случая
    head_mask всегда None, так что это просто [None]*num_layers.
    """
    from transformers import BertModel

    if hasattr(BertModel, "get_head_mask"):
        return

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is None:
            return [None] * num_hidden_layers
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        if is_attention_chunked:
            head_mask = head_mask.unsqueeze(-1)
        return head_mask

    BertModel.get_head_mask = get_head_mask


_patch_transformers_for_groundingdino()


PROMPTS = {
    1: ("water", (255, 80, 0), 0.25),
    2: ("vegetation, trees, grass, fields, forest", (40, 170, 40), 0.25),
    3: ("buildings, houses, rooftops", (220, 220, 220), 0.30),
    4: ("roads, streets, asphalt", (70, 70, 230), 0.25),
}


def colorize(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, (_, color, _) in PROMPTS.items():
        out[mask == cid] = color
    return out


def overlay_img(img: np.ndarray, mask: np.ndarray, a: float = 0.45) -> np.ndarray:
    return cv2.addWeighted(img, 1 - a, colorize(mask), a, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    print(f"device={args.device}")
    print(f"loading LangSAM...")
    from samgeo.text_sam import LangSAM
    t0 = time.time()
    sam = LangSAM(model_type="sam2-hiera-small")
    print(f"  loaded in {time.time()-t0:.1f}s")

    img_path = Path(args.image)
    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit(f"failed to read {img_path}")
    h, w = img.shape[:2]
    print(f"image {img_path.name}  {w}x{h}")

    composite = np.zeros((h, w), dtype=np.uint8)
    per_class_panels = []
    paint_order = [2, 3, 1, 4]
    for cid in paint_order:
        text, _, box_th = PROMPTS[cid]
        t0 = time.time()
        try:
            sam.predict(str(img_path), text, box_threshold=box_th, text_threshold=0.20)
        except Exception as e:
            print(f"  '{text}' FAILED: {e}")
            per_class_panels.append((cid, np.zeros_like(img)))
            continue
        masks = getattr(sam, "masks", None)
        if masks is None or len(masks) == 0:
            mask_bool = np.zeros((h, w), dtype=bool)
        else:
            mask_bool = np.asarray(masks).any(axis=0).astype(bool)
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
