"""Сравнительное демо-видео для защиты: OLD (Phase B baseline mIoU 50.75)
vs NEW (Phase C+OSM manual_cw) — на одном видео GP010269 в реальном времени.
Главный визуальный артефакт диплома: видно как старая модель красит
зимние тайлы как-попало (большинство = bg), новая делает аккуратные маски.

Layout каждого frame:
+------------------+------------------+------------------+
| drone frame      | OLD overlay      | NEW overlay      |
+------------------+------------------+------------------+
| легенда + текст  ...
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
            up = _resize(input=self.convs[idx](x), size=inputs[0].shape[2:],
                         mode=self.interpolate_mode, align_corners=self.align_corners)
            outs.append(up.contiguous())
        out = self.fusion_conv(torch.cat(outs, dim=1).contiguous())
        return self.cls_seg(out)

    _mit.EfficientMultiheadAttention.forward = patched_attn
    _shead.SegformerHead.forward = patched_head


_patch_mmseg()
_orig_load = torch.load
torch.load = lambda *a, **k: (_orig_load(*a, **{**k, "weights_only": False}) if "weights_only" not in k else _orig_load(*a, **k))


CLASS_NAMES = {0: "background", 1: "water", 2: "vegetation", 3: "buildings", 4: "roads"}
PALETTE = {0: (0, 0, 0), 1: (255, 80, 0), 2: (40, 170, 40), 3: (220, 220, 220), 4: (70, 70, 230)}


def colorize(mask):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cid, color in PALETTE.items():
        out[mask == cid] = color
    return out


def overlay(img, mask, a=0.50):
    return cv2.addWeighted(img, 1 - a, colorize(mask), a, 0)


def predict(model, img):
    from mmseg.apis import inference_model
    return inference_model(model, img).pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8)


def _font(size, bold=False):
    cands = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else None,
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for p in cands:
        if p and Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_text(img, text, xy, color=(255, 255, 255), size=18, bold=False, anchor="lt"):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text(xy, text, fill=color, font=_font(size, bold), anchor=anchor)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def make_legend(width, height=64):
    bar = np.full((height, width, 3), 28, dtype=np.uint8)
    items = [(c, CLASS_NAMES[c], PALETTE[c]) for c in (0, 1, 2, 3, 4)]
    spacing = width // (len(items) + 1)
    for i, (_, name, color) in enumerate(items):
        x0 = (i + 1) * spacing - 110
        cv2.rectangle(bar, (x0, 20), (x0 + 32, 48), color, -1)
        cv2.rectangle(bar, (x0, 20), (x0 + 32, 48), (200, 200, 200), 1)
        bar = draw_text(bar, name, (x0 + 42, height // 2),
                        color=(245, 245, 245), size=18, anchor="lm")
    return bar


def class_share(mask):
    vals, counts = np.unique(mask, return_counts=True)
    total = mask.size
    return {int(v): float(c) / total * 100 for v, c in zip(vals, counts)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="data/videos/GP010269.MP4")
    p.add_argument("--old-cfg",
                   default="results/segformer_overture_b0_focus_resume_wclean_v3/focus_refine/segformer_overture_focus_cfg.py")
    p.add_argument("--old-ckpt",
                   default="results/segformer_overture_b0_focus_resume_wclean_v3/focus_refine/best_mIoU_iter_220.pth")
    p.add_argument("--new-cfg",
                   default="results/segformer_overture_b0_phase_c_osm_manualcw/segformer_overture_quick_cfg.py")
    p.add_argument("--new-ckpt",
                   default="results/segformer_overture_b0_phase_c_osm_manualcw/best_mIoU_iter_1165.pth")
    p.add_argument("--out", default="results/demo/segformer_compare_old_vs_new.mp4")
    p.add_argument("--start-s", type=float, default=60.0)
    p.add_argument("--end-s", type=float, default=90.0)
    p.add_argument("--fps-out", type=float, default=6.0)
    p.add_argument("--panel-size", type=int, default=384)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-frames", type=int, default=180)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Loading OLD: {args.old_ckpt}")
    from mmseg.apis import init_model
    model_old = init_model(args.old_cfg, args.old_ckpt, device=args.device)
    print(f"Loading NEW: {args.new_ckpt}")
    model_new = init_model(args.new_cfg, args.new_ckpt, device=args.device)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"failed to open {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"video: {int(cap.get(3))}x{int(cap.get(4))} @ {src_fps:.1f}fps")

    P = args.panel_size
    out_w = P * 3
    out_h = P + 64 + 36 + 30  # row + legend + title + per-class stats
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps_out, (out_w, out_h))

    step = max(1, int(round(src_fps / args.fps_out)))
    start_frame = int(args.start_s * src_fps)
    end_frame = int(args.end_s * src_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    f_idx = start_frame
    rendered = 0
    while f_idx < end_frame and rendered < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if (f_idx - start_frame) % step != 0:
            f_idx += 1
            continue

        h, w = frame.shape[:2]
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        drone_sq = cv2.resize(frame[y0:y0 + side, x0:x0 + side], (P, P), interpolation=cv2.INTER_AREA)

        pred_old = predict(model_old, drone_sq)
        pred_new = predict(model_new, drone_sq)
        ov_old = overlay(drone_sq, pred_old)
        ov_new = overlay(drone_sq, pred_new)

        row = np.hstack([drone_sq, ov_old, ov_new])

        # Sub-headers
        row = draw_text(row, "drone (raw)", (12, 12), size=14, bold=True)
        row = draw_text(row, "OLD: Phase B baseline mIoU 50.75", (P + 12, 12), size=13, bold=True)
        row = draw_text(row, "NEW: Phase C+OSM manual_cw mIoU 47.71", (P * 2 + 12, 12), size=13, bold=True)

        # Title
        title = np.full((36, out_w, 3), 22, dtype=np.uint8)
        title = draw_text(title, f"GP010269.MP4 — t={f_idx/src_fps:.1f}s "
                                 f"(frame {f_idx})  |  сравнение SegFormer старая vs новая",
                          (12, 18), color=(255, 255, 255), size=15, bold=True, anchor="lm")

        # Per-class stats strip (под row)
        share_old = class_share(pred_old)
        share_new = class_share(pred_new)
        stats = np.full((30, out_w, 3), 18, dtype=np.uint8)
        old_str = "OLD: " + " ".join(f"{CLASS_NAMES[c][:3]}={share_old.get(c, 0):.0f}%"
                                      for c in (0, 1, 2, 3, 4))
        new_str = "NEW: " + " ".join(f"{CLASS_NAMES[c][:3]}={share_new.get(c, 0):.0f}%"
                                      for c in (0, 1, 2, 3, 4))
        stats = draw_text(stats, old_str, (P + 12, 14), color=(255, 200, 200), size=12, anchor="lm")
        stats = draw_text(stats, new_str, (P * 2 + 12, 14), color=(200, 255, 200), size=12, anchor="lm")

        legend = make_legend(out_w)
        out_frame = np.vstack([title, row, stats, legend])
        writer.write(out_frame)

        rendered += 1
        if rendered % 20 == 0:
            print(f"  rendered {rendered} frames")
        f_idx += 1

    cap.release()
    writer.release()
    print(f"Wrote {out_path} ({rendered} frames @ {args.fps_out} fps = {rendered / args.fps_out:.1f}s)")


if __name__ == "__main__":
    main()
