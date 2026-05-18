"""Финальные thesis-ready figures с понятными подписями и легендой.

Производит:
- fig_segformer_old_vs_new.png  — 4 панели на тайл, 6 регионов
- fig_segformer_vs_sam3.png     — 4 панели на тайл, 6 patches_512
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def _find_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    if bold:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/HelveticaNeue.ttc",
        ] + candidates
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_text_pil(img_bgr: np.ndarray, text: str, xy: tuple[int, int],
                   color: tuple[int, int, int] = (255, 255, 255),
                   size: int = 18, bold: bool = False,
                   anchor: str = "lt") -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _find_font(size, bold=bold)
    draw.text(xy, text, fill=color, font=font, anchor=anchor)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from mmseg.models.backbones import mit as _mit
from mmseg.models.decode_heads import segformer_head as _shead
from mmseg.models.utils import resize as _resize


def _patch_mmseg() -> None:
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


CLASS_NAMES = {
    0: "background",
    1: "water",
    2: "vegetation",
    3: "buildings",
    4: "roads",
}
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


def upscale(img: np.ndarray, target: int) -> np.ndarray:
    return cv2.resize(img, (target, target), interpolation=cv2.INTER_NEAREST)


def make_header_strip(width: int, columns: list[str], col_w: int,
                      left_pad: int, height: int = 72) -> np.ndarray:
    bar = np.full((height, width, 3), 28, dtype=np.uint8)
    for i, label in enumerate(columns):
        cx = left_pad + i * col_w + col_w // 2
        cy = height // 2
        bar = draw_text_pil(bar, label, (cx, cy), color=(255, 255, 255),
                             size=20, bold=True, anchor="mm")
    return bar


def make_legend_strip(width: int, height: int = 70) -> np.ndarray:
    bar = np.full((height, width, 3), 28, dtype=np.uint8)
    items = [(cid, CLASS_NAMES[cid], PALETTE[cid]) for cid in (0, 1, 2, 3, 4)]
    spacing = width // (len(items) + 1)
    for i, (_, name, color) in enumerate(items):
        x0 = (i + 1) * spacing - 130
        cv2.rectangle(bar, (x0, 22), (x0 + 32, 50), color, -1)
        cv2.rectangle(bar, (x0, 22), (x0 + 32, 50), (200, 200, 200), 1)
        bar = draw_text_pil(bar, name, (x0 + 42, height // 2),
                             color=(245, 245, 245), size=18, anchor="lm")
    return bar


def make_row_label(text: str, width: int, height: int) -> np.ndarray:
    bar = np.full((height, width, 3), 22, dtype=np.uint8)
    lines = text.split("\n")
    line_h = max(22, height // max(len(lines) + 1, 4))
    total_h = line_h * len(lines)
    start_y = (height - total_h) // 2 + line_h // 2
    for i, line in enumerate(lines):
        bar = draw_text_pil(bar, line, (width // 2, start_y + i * line_h),
                             color=(235, 235, 235), size=15, anchor="mm")
    return bar


def predict_segformer(model, img_bgr: np.ndarray) -> np.ndarray:
    from mmseg.apis import inference_model
    res = inference_model(model, img_bgr)
    return res.pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8)


def predict_sam3(sam, img_path: Path) -> np.ndarray:
    sam.set_image(str(img_path))
    h, w = sam.image_height, sam.image_width
    composite = np.zeros((h, w), dtype=np.uint8)
    for cid in (2, 3, 1, 4):
        try:
            sam.generate_masks(prompt=SAM3_PROMPTS[cid], quiet=True)
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
    p.add_argument("--seg-old-cfg", required=True)
    p.add_argument("--seg-old-ckpt", required=True)
    p.add_argument("--seg-new-cfg", required=True)
    p.add_argument("--seg-new-ckpt", required=True)
    p.add_argument("--out-dir", default="docs/figures/phase4")
    p.add_argument("--device", default="cpu")
    p.add_argument("--panel-size", type=int, default=384)
    p.add_argument("--label-width", type=int, default=180)
    p.add_argument("--skip-sam3", action="store_true")
    return p.parse_args()


def render_segformer_old_vs_new(args, model_old, model_new) -> None:
    root = Path(args.dataset_root)
    img_dir = root / "images" / "tiles"
    mask_dir = root / "masks" / "tiles"

    rows_def = [
        ("moscow_city_small_17_79220_40973",
         "Moscow city\nплотная застройка\n(старая модель «заливает» дороги)"),
        ("moscow_city_small_17_79227_40965",
         "Moscow city\n(улицы тоньше у новой)"),
        ("moscow_suburb_small_17_79301_40861",
         "Moscow suburb\n(промзона, ангары)"),
        ("krasnodar_fields_small_17_79661_46782",
         "Krasnodar fields\n(агро поля)"),
        ("karelia_forest_small_17_76718_36731",
         "Karelia forest\n(монокультура vegetation)"),
        ("svo_airport_small_17_79170_40819",
         "SVO airport\n(окрестности аэродрома)"),
    ]

    P = args.panel_size
    L = args.label_width
    columns = ["сатспутник Z=17", "Overture GT", "SegFormer старая\n(mIoU 50.75)",
               "SegFormer новая\n(mIoU 52.03, Phase B)"]
    rows_built = []
    for tile, descr in rows_def:
        img_p = img_dir / f"{tile}.png"
        msk_p = mask_dir / f"{tile}.png"
        if not img_p.exists() or not msk_p.exists():
            print(f"skip {tile}")
            continue
        img = cv2.imread(str(img_p))
        gt = cv2.imread(str(msk_p), cv2.IMREAD_GRAYSCALE)
        old_pred = predict_segformer(model_old, img)
        new_pred = predict_segformer(model_new, img)
        panels = [
            upscale(img, P),
            upscale(overlay(img, gt), P),
            upscale(overlay(img, old_pred), P),
            upscale(overlay(img, new_pred), P),
        ]
        rowimg = np.hstack(panels)
        label = make_row_label(descr, L, P)
        rows_built.append(np.hstack([label, rowimg]))
        print(f"  done {tile}")

    if not rows_built:
        return
    full_w = rows_built[0].shape[1]
    body = np.vstack(rows_built)
    title = np.full((52, full_w, 3), 18, dtype=np.uint8)
    title = draw_text_pil(title, "Сравнение сегментации: старая vs Phase B SegFormer-B0",
                           (16, 26), color=(255, 255, 255), size=22, bold=True, anchor="lm")

    # 4 столбца справа от label
    header = make_header_strip(full_w, columns, P, L)
    legend = make_legend_strip(full_w)
    out = np.vstack([title, header, body, legend])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig_segformer_old_vs_new.png"
    cv2.imwrite(str(out_path), out)
    print(f"Wrote {out_path}")


def render_segformer_vs_sam3(args, model_new, sam) -> None:
    root = Path(args.dataset_root)
    img_dir = root / "images" / "patches_512"
    mask_dir = root / "masks" / "patches_512"

    rows_def = [
        ("moscow_city_small_17_79218_40964",
         "Moscow city\nплотная городская застройка"),
        ("moscow_city_small_17_79224_40970",
         "Moscow city\nкрыши + улицы"),
        ("moscow_suburb_small_17_79294_40840",
         "Moscow suburb\nпромзона + дорога"),
        ("krasnodar_fields_small_17_79648_46777",
         "Krasnodar fields\n(SAM3 находит водоём,\nкоторого нет в Overture)"),
        ("karelia_forest_small_17_76711_36721",
         "Karelia forest with lake\n(SAM3 сегментирует озеро,\nOverture его пропускает)"),
        ("svo_airport_small_17_79149_40819",
         "SVO airport\n(окрестности)"),
    ]

    P = args.panel_size
    L = args.label_width
    columns = ["спутник Z=17 (512)", "Overture GT", "SegFormer Phase B\n(mIoU 52.03)",
               "SAM3 (text-prompt)\nfacebook/sam3"]
    rows_built = []
    for tile, descr in rows_def:
        img_p = img_dir / f"{tile}.png"
        msk_p = mask_dir / f"{tile}.png"
        if not img_p.exists() or not msk_p.exists():
            print(f"skip {tile}")
            continue
        img = cv2.imread(str(img_p))
        gt = cv2.imread(str(msk_p), cv2.IMREAD_GRAYSCALE)
        new_pred = predict_segformer(model_new, img)
        sam_pred = predict_sam3(sam, img_p)
        panels = [
            upscale(img, P),
            upscale(overlay(img, gt), P),
            upscale(overlay(img, new_pred), P),
            upscale(overlay(img, sam_pred), P),
        ]
        rowimg = np.hstack(panels)
        label = make_row_label(descr, L, P)
        rows_built.append(np.hstack([label, rowimg]))
        print(f"  done {tile}")

    if not rows_built:
        return
    full_w = rows_built[0].shape[1]
    body = np.vstack(rows_built)
    title = np.full((52, full_w, 3), 18, dtype=np.uint8)
    title = draw_text_pil(title, "Phase B SegFormer-B0 vs SAM3 (text-prompt сегментация)",
                           (16, 26), color=(255, 255, 255), size=22, bold=True, anchor="lm")
    header = make_header_strip(full_w, columns, P, L)
    legend = make_legend_strip(full_w)
    out = np.vstack([title, header, body, legend])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig_segformer_vs_sam3.png"
    cv2.imwrite(str(out_path), out)
    print(f"Wrote {out_path}")


def main() -> None:
    args = parse_args()

    print("loading SegFormer old...")
    from mmseg.apis import init_model
    seg_old = init_model(args.seg_old_cfg, args.seg_old_ckpt, device=args.device)
    print("loading SegFormer new (Phase B)...")
    seg_new = init_model(args.seg_new_cfg, args.seg_new_ckpt, device=args.device)

    render_segformer_old_vs_new(args, seg_old, seg_new)

    if args.skip_sam3:
        return

    print("loading SAM3...")
    from samgeo.samgeo3 import SamGeo3
    sam = SamGeo3(
        model_id="facebook/sam3",
        device=args.device,
        backend="transformers",
        confidence_threshold=0.30,
        mask_threshold=0.40,
    )
    render_segformer_vs_sam3(args, seg_new, sam)


if __name__ == "__main__":
    main()
