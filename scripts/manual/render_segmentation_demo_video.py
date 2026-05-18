"""Демо-видео для защиты диплома: сегментация Phase C+OSM в реальном времени
поверх кадров GP010269.MP4. Показывает что модель работает end-to-end на
реальном видео — не статичная картинка, а 30-60 секундный клип.

Layout каждого frame:
+------------------+------------------+
| drone frame      | drone + seg      |  ← живой ввод сегментации
+------------------+------------------+
| satellite tile   | satellite + seg  |  ← та же модель на спутнике
+------------------+------------------+
| легенда классов + text-метрики                   |
+--------------------------------------------------+
"""

from __future__ import annotations

import argparse
import math
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


def _patch_mmseg() -> None:
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
torch.load = lambda *a, **k: (_orig_load(*a, **{**k, "weights_only": False}) if "weights_only" not in k else _orig_load(*a, **k))


CLASS_NAMES = {0: "background", 1: "water", 2: "vegetation", 3: "buildings", 4: "roads"}
PALETTE = {0: (0, 0, 0), 1: (255, 80, 0), 2: (40, 170, 40), 3: (220, 220, 220), 4: (70, 70, 230)}


def colorize(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cid, color in PALETTE.items():
        out[mask == cid] = color
    return out


def overlay(img: np.ndarray, mask: np.ndarray, a: float = 0.50) -> np.ndarray:
    return cv2.addWeighted(img, 1 - a, colorize(mask), a, 0)


def predict(model, img):
    from mmseg.apis import inference_model
    return inference_model(model, img).pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8)


def _font(size: int, bold: bool = False):
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


def draw_text(img: np.ndarray, text: str, xy, color=(255, 255, 255), size=18, bold=False, anchor="lt"):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text(xy, text, fill=color, font=_font(size, bold), anchor=anchor)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def make_legend(width: int, height: int = 64):
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


def lat_lon_for_frame(frame_idx: int, fps: float, start_lat: float, start_lon: float):
    """Telemetry placeholder — для демо берём фиксированную точку.
    В production полная EKF трекинг даёт реальную траекторию."""
    return start_lat, start_lon


def fetch_satellite_tile(lat: float, lon: float, zoom: int = 17, size: int = 512):
    """Скачиваем спутниковую плитку через Google Satellite XYZ.
    Используем кэшированную плитку если есть, иначе сетевой запрос."""
    import requests
    n = 2 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    url = f"https://mt1.google.com/vt/lyrs=s&x={xtile}&y={ytile}&z={zoom}"
    cache = Path(f"/tmp/sat_demo_{zoom}_{xtile}_{ytile}.png")
    if cache.exists():
        return cv2.imread(str(cache))
    headers = {"User-Agent": "Mozilla/5.0 (Avia-Geo-Fusion-Demo)"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            cache.write_bytes(r.content)
            return cv2.imread(str(cache))
    except Exception as ex:
        print(f"  sat tile fetch failed: {ex}")
    return np.full((256, 256, 3), 80, dtype=np.uint8)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="data/videos/GP010269.MP4")
    p.add_argument("--seg-config",
                   default="results/segformer_overture_b0_phase_c_osm_manualcw/segformer_overture_quick_cfg.py")
    p.add_argument("--seg-checkpoint",
                   default="results/segformer_overture_b0_phase_c_osm_manualcw/best_mIoU_iter_1165.pth")
    p.add_argument("--out", default="results/demo/segformer_demo_phase_c_osm.mp4")
    p.add_argument("--start-s", type=float, default=60.0)
    p.add_argument("--end-s", type=float, default=90.0)
    p.add_argument("--fps-out", type=float, default=10.0)
    p.add_argument("--panel-size", type=int, default=384)
    p.add_argument("--start-lat", type=float, default=55.086025)
    p.add_argument("--start-lon", type=float, default=38.149033)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-frames", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Loading SegFormer ({args.seg_checkpoint})...")
    from mmseg.apis import init_model
    model = init_model(args.seg_config, args.seg_checkpoint, device=args.device)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"failed to open {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"video: {src_w}x{src_h} @ {src_fps:.1f}fps")

    P = args.panel_size
    out_w = P * 2  # drone | drone+seg
    out_h = P * 2 + 64 + 36  # row1 + row2 + legend + title
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps_out, (out_w, out_h))

    # Sample step
    step = max(1, int(round(src_fps / args.fps_out)))
    start_frame = int(args.start_s * src_fps)
    end_frame = int(args.end_s * src_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    sat_tile = fetch_satellite_tile(args.start_lat, args.start_lon, zoom=17, size=P)
    sat_resized = cv2.resize(sat_tile, (P, P), interpolation=cv2.INTER_AREA)
    print(f"sat tile loaded: {sat_resized.shape}")
    sat_pred = predict(model, sat_resized)
    sat_overlay = overlay(sat_resized, sat_pred)

    f_idx = start_frame
    rendered = 0
    while f_idx < end_frame and rendered < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if (f_idx - start_frame) % step != 0:
            f_idx += 1
            continue

        # Crop center square + resize
        h, w = frame.shape[:2]
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        drone_sq = cv2.resize(frame[y0:y0 + side, x0:x0 + side], (P, P), interpolation=cv2.INTER_AREA)

        drone_pred = predict(model, drone_sq)
        drone_ov = overlay(drone_sq, drone_pred)

        # Layout 2x2
        row1 = np.hstack([drone_sq, drone_ov])
        row2 = np.hstack([sat_resized, sat_overlay])

        title = np.full((36, out_w, 3), 22, dtype=np.uint8)
        title = draw_text(title, f"SegFormer Phase C+OSM на GP010269.MP4 — t={f_idx/src_fps:.1f}s "
                                 f"(frame {f_idx})",
                          (12, 18), color=(255, 255, 255), size=15, bold=True, anchor="lm")

        # Sub-headers поверх каждой строки
        row1 = draw_text(row1, "drone (raw)", (12, 12), color=(255, 255, 255), size=14, bold=True)
        row1 = draw_text(row1, "drone + segmentation overlay", (P + 12, 12), color=(255, 255, 255), size=14, bold=True)
        row2 = draw_text(row2, "satellite tile (Z=17)", (12, 12), color=(255, 255, 255), size=14, bold=True)
        row2 = draw_text(row2, "satellite + segmentation overlay", (P + 12, 12), color=(255, 255, 255), size=14, bold=True)

        legend = make_legend(out_w)
        out_frame = np.vstack([title, row1, row2, legend])
        writer.write(out_frame)

        rendered += 1
        if rendered % 20 == 0:
            print(f"  rendered {rendered} frames...")
        f_idx += 1

    cap.release()
    writer.release()
    print(f"\nWrote {out_path} ({rendered} frames @ {args.fps_out} fps = {rendered / args.fps_out:.1f}s)")


if __name__ == "__main__":
    main()
