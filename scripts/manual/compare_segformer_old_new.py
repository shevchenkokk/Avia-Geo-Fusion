"""Сравнение старой (mIoU 50.75 на moscow_city val) и новой (mIoU 52.03
на 4-region val после Phase B) SegFormer-моделей на репрезентативных
тайлах из всех регионов датасета.

Рендерит для каждого тайла строку:
  image | new_overture_mask | old_segformer_pred | new_segformer_pred
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Те же патчи на mmseg, что в финетюне — иначе модель не загрузится.
import torch  # noqa: E402

from mmseg.models.backbones import mit as _mit
from mmseg.models.decode_heads import segformer_head as _shead
from mmseg.models.utils import resize as _resize


def _patch_mit():
    def patched_forward(self, x, hw_shape, identity=None):
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
    _mit.EfficientMultiheadAttention.forward = patched_forward


def _patch_head():
    def patched_forward(self, inputs):
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
    _shead.SegformerHead.forward = patched_forward


_patch_mit()
_patch_head()

# torch.load weights_only=False для совместимости с mmengine checkpoints в PyTorch 2.6+.
_orig_load = torch.load
def _load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _load_compat

from mmseg.apis import init_model, inference_model  # noqa: E402


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


def overlay(img: np.ndarray, mask: np.ndarray, a: float = 0.45) -> np.ndarray:
    return cv2.addWeighted(img, 1 - a, colorize(mask), a, 0)


def predict(model, img_bgr: np.ndarray) -> np.ndarray:
    res = inference_model(model, img_bgr)
    mask = res.pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8)
    return mask


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="data/overture_ru_dataset_starter")
    p.add_argument("--old-cfg", required=True)
    p.add_argument("--old-ckpt", required=True)
    p.add_argument("--new-cfg", required=True)
    p.add_argument("--new-ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--per-region", type=int, default=2)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    print(f"Loading OLD model: {args.old_ckpt}")
    old_model = init_model(args.old_cfg, args.old_ckpt, device=args.device)
    print(f"Loading NEW model: {args.new_ckpt}")
    new_model = init_model(args.new_cfg, args.new_ckpt, device=args.device)

    root = Path(args.dataset_root)
    img_dir = root / "images" / "tiles"
    mask_dir = root / "masks" / "tiles"

    by_region: dict[str, list] = {}
    for p in sorted(img_dir.glob("*.png")):
        if not (mask_dir / p.name).exists():
            continue
        reg = p.stem.split("_17_")[0]
        by_region.setdefault(reg, []).append(p)

    sampled = []
    for reg in sorted(by_region):
        names = by_region[reg][:]
        random.shuffle(names)
        sampled.extend(names[: args.per_region])

    rows = []
    for img_path in sampled:
        img = cv2.imread(str(img_path))
        gt_mask = cv2.imread(str(mask_dir / img_path.name), cv2.IMREAD_GRAYSCALE)
        if img is None or gt_mask is None:
            continue
        old_pred = predict(old_model, img)
        new_pred = predict(new_model, img)

        row = np.hstack([
            img,
            overlay(img, gt_mask),
            overlay(img, old_pred),
            overlay(img, new_pred),
        ])
        label = f"{img_path.stem}"
        cv2.putText(row, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(row)
        print(f"  done: {img_path.stem}")

    if not rows:
        print("Nothing to render")
        return

    head_w = rows[0].shape[1]
    head = np.full((28, head_w, 3), 30, dtype=np.uint8)
    cv2.putText(head, "image  |  overture_GT  |  old_segformer_pred  |  new_segformer_pred",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    out = np.vstack([head, *rows])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
