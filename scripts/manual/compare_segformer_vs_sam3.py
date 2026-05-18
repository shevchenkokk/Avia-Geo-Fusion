"""Сравнение качества масок: Overture-GT vs наш SegFormer (Phase B) vs SAM3.
На репрезентативных тайлах из 4 регионов.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from mmseg.models.backbones import mit as _mit
from mmseg.models.decode_heads import segformer_head as _shead
from mmseg.models.utils import resize as _resize


def _patch_mmseg():
    def patched_attn_forward(self, x, hw_shape, identity=None):
        from mmseg.models.utils import nchw_to_nlc, nlc_to_nchw
        x_q = x
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x
        if identity is None:
            identity = x_q
        if self.batch_first:
            x_q = x_q.transpose(0, 1).contiguous()
            x_kv = x_kv.transpose(0, 1).contiguous()
        out = self.attn(query=x_q, key=x_kv, value=x_kv)[0]
        if self.batch_first:
            out = out.transpose(0, 1).contiguous()
        return identity + self.dropout_layer(self.proj_drop(out))

    def patched_head_forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx].contiguous()
            up = _resize(
                input=self.convs[idx](x),
                size=inputs[0].shape[2:],
                mode=self.interpolate_mode,
                align_corners=self.align_corners,
            )
            outs.append(up.contiguous())
        out = self.fusion_conv(torch.cat(outs, dim=1).contiguous())
        return self.cls_seg(out)

    _mit.EfficientMultiheadAttention.forward = patched_attn_forward
    _shead.SegformerHead.forward = patched_head_forward


_patch_mmseg()

_orig_load = torch.load
def _load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _load_compat


PALETTE = {
    0: (0, 0, 0),
    1: (255, 80, 0),
    2: (40, 170, 40),
    3: (220, 220, 220),
    4: (70, 70, 230),
}

SAM3_PROMPTS = {
    1: "water",
    2: "tree",
    3: "building",
    4: "road",
}


def colorize(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in PALETTE.items():
        out[mask == cid] = color
    return out


def overlay(img: np.ndarray, mask: np.ndarray, a: float = 0.45) -> np.ndarray:
    return cv2.addWeighted(img, 1 - a, colorize(mask), a, 0)


def predict_segformer(model, img_bgr: np.ndarray) -> np.ndarray:
    from mmseg.apis import inference_model
    res = inference_model(model, img_bgr)
    return res.pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8)


def predict_sam3(sam, img_path: Path) -> np.ndarray:
    sam.set_image(str(img_path))
    h, w = sam.image_height, sam.image_width
    composite = np.zeros((h, w), dtype=np.uint8)
    paint_order = [2, 3, 1, 4]
    for cid in paint_order:
        prompt = SAM3_PROMPTS[cid]
        try:
            sam.generate_masks(prompt=prompt, quiet=True)
        except Exception:
            continue
        masks = getattr(sam, "masks", None)
        if masks is None:
            continue
        arr = np.asarray(masks)
        if arr.ndim == 3 and arr.shape[0] > 0:
            mask_bool = arr.any(axis=0).astype(bool)
        elif arr.ndim == 2:
            mask_bool = arr.astype(bool)
        else:
            continue
        composite[mask_bool] = cid
    return composite


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="data/overture_ru_dataset_starter")
    p.add_argument("--seg-cfg", required=True)
    p.add_argument("--seg-ckpt", required=True)
    p.add_argument("--sam-model", default="facebook/sam3")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--tiles", nargs="+", required=True,
                   help="Explicit tile names (without .png) to compare")
    p.add_argument("--subset", default="tiles", choices=["tiles", "patches_512"],
                   help="Which subdir of images/ and masks/ to read from")
    p.add_argument("--sam-conf", type=float, default=0.30)
    p.add_argument("--sam-mask-th", type=float, default=0.40)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root)
    img_dir = root / "images" / args.subset
    mask_dir = root / "masks" / args.subset

    print("loading SegFormer...")
    from mmseg.apis import init_model
    seg_model = init_model(args.seg_cfg, args.seg_ckpt, device=args.device)

    print("loading SAM3...")
    from samgeo.samgeo3 import SamGeo3
    t0 = time.time()
    sam = SamGeo3(
        model_id=args.sam_model,
        device=args.device,
        backend="transformers",
        confidence_threshold=args.sam_conf,
        mask_threshold=args.sam_mask_th,
    )
    print(f"  loaded in {time.time()-t0:.1f}s")

    rows = []
    for name in args.tiles:
        path = img_dir / f"{name}.png"
        gt_path = mask_dir / f"{name}.png"
        if not path.exists() or not gt_path.exists():
            print(f"missing {name}, skip")
            continue
        img = cv2.imread(str(path))
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            continue
        print(f"\n{name}")
        t0 = time.time()
        seg_pred = predict_segformer(seg_model, img)
        print(f"  segformer {time.time()-t0:.1f}s")
        t0 = time.time()
        sam_pred = predict_sam3(sam, path)
        print(f"  sam3 {time.time()-t0:.1f}s")

        row = np.hstack([
            img,
            overlay(img, gt),
            overlay(img, seg_pred),
            overlay(img, sam_pred),
        ])
        cv2.putText(row, name, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(row)

    if not rows:
        return
    head_w = rows[0].shape[1]
    head = np.full((28, head_w, 3), 30, dtype=np.uint8)
    cv2.putText(head, "image  |  overture_GT  |  segformer_phaseB  |  sam3",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    out = np.vstack([head, *rows])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, out)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
