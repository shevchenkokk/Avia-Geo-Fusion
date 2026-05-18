"""Cross-domain test: SegFormer Phase C vs Phase C+OSM manual_cw на
AerialVL (Циндао, Китай) и VPair (Кёльн, Германия) — наша модель тренирована
на России, эти изображения out-of-domain.

Рендерит compact figure с краеугольными примерами + caption под каждой
строкой описывающий что заметить (cross-region generalization).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmseg.models.backbones import mit as _mit
from mmseg.models.decode_heads import segformer_head as _shead
from mmseg.models.utils import resize as _resize


def _patch_mmseg():
    def patched_attn(self, x, hw_shape, identity=None):
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

    def patched_head(self, inputs):
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

    _mit.EfficientMultiheadAttention.forward = patched_attn
    _shead.SegformerHead.forward = patched_head


_patch_mmseg()
_orig_load = torch.load
torch.load = lambda *a, **k: (k.update({"weights_only": False}) or _orig_load(*a, **k)) if "weights_only" not in k else _orig_load(*a, **k)


CLASS_NAMES = {0: "background", 1: "water", 2: "vegetation", 3: "buildings", 4: "roads"}
PALETTE = {0: (0, 0, 0), 1: (255, 80, 0), 2: (40, 170, 40), 3: (220, 220, 220), 4: (70, 70, 230)}


def _font(size: int, bold: bool = False):
    cands = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else None,
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for p in cands:
        if p and Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_text(img: np.ndarray, text: str, xy, color=(255, 255, 255), size=18, bold=False, anchor="lt"):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text(xy, text, fill=color, font=_font(size, bold), anchor=anchor)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def colorize(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cid, color in PALETTE.items():
        out[mask == cid] = color
    return out


def overlay(img, mask, a=0.45):
    return cv2.addWeighted(img, 1 - a, colorize(mask), a, 0)


def predict(model, img):
    from mmseg.apis import inference_model
    return inference_model(model, img).pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8)


def make_legend(width: int, height: int = 60):
    bar = np.full((height, width, 3), 28, dtype=np.uint8)
    items = [(c, CLASS_NAMES[c], PALETTE[c]) for c in (0, 1, 2, 3, 4)]
    spacing = width // (len(items) + 1)
    for i, (_, name, color) in enumerate(items):
        x0 = (i + 1) * spacing - 110
        cv2.rectangle(bar, (x0, 18), (x0 + 28, 44), color, -1)
        cv2.rectangle(bar, (x0, 18), (x0 + 28, 44), (200, 200, 200), 1)
        bar = draw_text(bar, name, (x0 + 38, height // 2),
                        color=(245, 245, 245), size=16, anchor="lm")
    return bar


def make_header(width: int, columns: list[str], col_w: int, left_pad: int, height: int = 64):
    bar = np.full((height, width, 3), 28, dtype=np.uint8)
    for i, label in enumerate(columns):
        cx = left_pad + i * col_w + col_w // 2
        cy = height // 2
        bar = draw_text(bar, label, (cx, cy), color=(255, 255, 255), size=18, bold=True, anchor="mm")
    return bar


def make_row_label(text: str, width: int, height: int):
    bar = np.full((height, width, 3), 22, dtype=np.uint8)
    lines = text.split("\n")
    line_h = max(20, height // (len(lines) + 1))
    start = (height - line_h * len(lines)) // 2 + line_h // 2
    for i, line in enumerate(lines):
        bar = draw_text(bar, line, (width // 2, start + i * line_h),
                        color=(235, 235, 235), size=13, anchor="mm")
    return bar


def central_crop_resize(img: np.ndarray, target: int) -> np.ndarray:
    h, w = img.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    crop = img[y0:y0 + side, x0:x0 + side]
    return cv2.resize(crop, (target, target), interpolation=cv2.INTER_AREA)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seg-c-cfg", required=True)
    p.add_argument("--seg-c-ckpt", required=True)
    p.add_argument("--seg-cosm-cfg", required=True)
    p.add_argument("--seg-cosm-ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--panel-size", type=int, default=384)
    p.add_argument("--label-w", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()

    print("loading SegFormer Phase C (no OSM)...")
    from mmseg.apis import init_model
    model_c = init_model(args.seg_c_cfg, args.seg_c_ckpt, device=args.device)

    print("loading SegFormer Phase C+OSM manual_cw...")
    model_cosm = init_model(args.seg_cosm_cfg, args.seg_cosm_ckpt, device=args.device)

    # Подбираем разнообразные кадры по индексам в датасетах
    aerialvl_dir = Path("data/external/aerialvl/short_trajtr/2023-03-11-11-48-35")
    vpair_q_dir = Path("data/external/vpair_sample/queries")
    vpair_r_dir = Path("data/external/vpair_sample/reference_views")

    aerialvl_frames = sorted(aerialvl_dir.glob("*.png"))
    vpair_queries = sorted(vpair_q_dir.glob("*.png"))
    vpair_refs = sorted(vpair_r_dir.glob("*.png"))

    # Берём 2 кадра из AerialVL (начало + середина), 2 query из VPair (разные сцены),
    # 2 reference из VPair (другая широта/мест).
    samples = []
    if len(aerialvl_frames) >= 2:
        samples.append((aerialvl_frames[0], "AerialVL drone\nЦиндао, Китай\nкадр #1"))
        samples.append((aerialvl_frames[len(aerialvl_frames) // 2], "AerialVL drone\nЦиндао, Китай\nкадр #75"))
    if len(vpair_queries) >= 4:
        samples.append((vpair_queries[0], "VPair query #1\nКёльн/Бонн, Германия"))
        samples.append((vpair_queries[len(vpair_queries) // 2], "VPair query #99\nдругой район"))
    if len(vpair_refs) >= 2:
        samples.append((vpair_refs[0], "VPair reference #1\nspatial overhead view"))
        samples.append((vpair_refs[len(vpair_refs) // 2], "VPair reference #99"))

    P = args.panel_size
    L = args.label_w
    rows = []
    for path, descr in samples:
        img = cv2.imread(str(path))
        if img is None:
            continue
        img_sq = central_crop_resize(img, P)
        pred_c = predict(model_c, img_sq)
        pred_cosm = predict(model_cosm, img_sq)
        panels = [img_sq, overlay(img_sq, pred_c), overlay(img_sq, pred_cosm)]
        rowimg = np.hstack(panels)
        label = make_row_label(descr, L, P)
        rows.append(np.hstack([label, rowimg]))
        print(f"  done {path.name}")

    if not rows:
        print("no samples")
        return

    full_w = rows[0].shape[1]
    body = np.vstack(rows)
    title = np.full((52, full_w, 3), 18, dtype=np.uint8)
    title = draw_text(title, "Cross-domain test: SegFormer на AerialVL (Китай) и VPair (Германия)",
                      (16, 26), color=(255, 255, 255), size=20, bold=True, anchor="lm")
    columns = ["исходное изображение", "Phase C (no OSM)\nmIoU 52.14, water 75",
               "Phase C+OSM manual_cw\nmIoU 47.71, water 61"]
    header = make_header(full_w, columns, P, L)
    legend = make_legend(full_w)

    out_arr = np.vstack([title, header, body, legend])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out_arr)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
