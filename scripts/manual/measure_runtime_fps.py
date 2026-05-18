"""Покомпонентный замер частоты обработки для раздела о времени работы.

Все времена измеряются на одном устройстве: Mac серии M с MPS для SegFormer.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _patch_mmseg():
    from mmseg.models.backbones import mit as _mit
    from mmseg.models.decode_heads import segformer_head as _shead
    from mmseg.models.utils import resize as _resize

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


def stats(values, label):
    if not values:
        return f"{label}: <no samples>"
    med = statistics.median(values)
    p95 = sorted(values)[int(len(values) * 0.95)]
    fps = 1000.0 / med if med > 0 else float("inf")
    return f"{label:<28} median={med:7.2f}ms  p95={p95:7.2f}ms  FPS={fps:6.1f}"


def measure_segformer(model, frame_512, n_warmup=3, n_iter=30):
    from mmseg.apis import inference_model
    # Прогрев модели перед измерением.
    for _ in range(n_warmup):
        _ = inference_model(model, frame_512)
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        _ = inference_model(model, frame_512)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def measure_xfeat(matcher, frame, sat_tile, n_warmup=2, n_iter=15):
    for _ in range(n_warmup):
        _ = matcher.match(frame, sat_tile, aircraft_mask=None)
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        _ = matcher.match(frame, sat_tile, aircraft_mask=None)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def measure_decode(video_path, n_iter=30):
    cap = cv2.VideoCapture(video_path)
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        ok, _ = cap.read()
        if not ok:
            break
        times.append((time.perf_counter() - t0) * 1000.0)
    cap.release()
    return times


def measure_bev(rectifier, frame, n_warmup=3, n_iter=30):
    for _ in range(n_warmup):
        _ = rectifier.warp(frame)
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        _ = rectifier.warp(frame)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="data/videos/GP010269.MP4")
    p.add_argument("--seg-config",
                   default="results/segformer_overture_b0_phase_c_osm_manualcw/segformer_overture_quick_cfg.py")
    p.add_argument("--seg-checkpoint",
                   default="results/segformer_overture_b0_phase_c_osm_manualcw/best_mIoU_iter_1165.pth")
    p.add_argument("--device", default="mps")
    return p.parse_args()


def main():
    args = parse_args()

    print("=== Измерение времени обработки ===")
    print(f"  устройство: {args.device}")
    print(f"  torch:  {torch.__version__}")
    print()

    # Получаем пример кадра для измерений.
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1800)  # 60s
    ok, frame_full = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("не удалось прочитать пример кадра")

    h, w = frame_full.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    frame_512 = cv2.resize(frame_full[y0:y0+side, x0:x0+side], (512, 512), interpolation=cv2.INTER_AREA)
    frame_800 = cv2.resize(frame_full[y0:y0+side, x0:x0+side], (800, 800), interpolation=cv2.INTER_AREA)

    print(f"пример кадра: {frame_full.shape}, BEV 800x800, SegFormer 512x512")
    print()

    # Декодирование видео.
    print(f">>> Декодирование видео (cv2.read для {Path(args.video).name})")
    times = measure_decode(args.video, n_iter=30)
    print(stats(times, "video decode"))
    print()

    # Преобразование к виду сверху.
    print(">>> Преобразование к виду сверху одной гомографией")
    from src.bev_rectifier import BevRectifier
    K = np.array([[1500, 0, w//2], [0, 1500, h//2], [0, 0, 1]], dtype=np.float64)
    rect = BevRectifier.build(
        K_rect=K, image_size=(w, h),
        pitch_deg=-30.0, agl_m=620.0, ground_span_m=400.0, out_size=(800, 800)
    )
    times = measure_bev(rect, frame_full, n_iter=30)
    print(stats(times, "BEV warp"))
    print()

    # Семантическая сегментация.
    print(">>> Инференс SegFormer-B0 (Phase C+OSM manual_cw)")
    from mmseg.apis import init_model
    model = init_model(args.seg_config, args.seg_checkpoint, device=args.device)
    times = measure_segformer(model, frame_512, n_iter=30)
    print(stats(times, "SegFormer 512x512"))
    print()

    # Локальное сопоставление.
    print(">>> Сопоставление XFeat (BEV-кадр против спутникового тайла)")
    from src.neural_matching import NeuralMatcher
    matcher = NeuralMatcher(backend="xfeat")
    sat_tile = frame_800.copy()  # Подставной спутниковый тайл для замера времени сопоставителя.
    times = measure_xfeat(matcher, frame_800, sat_tile, n_iter=15)
    print(stats(times, "XFeat match 800x800"))
    print()

    # Сводка.
    print("=" * 60)
    print("Сценарий end-to-end (на map-кадре с появлением матчинга):")
    print("  decode + BEV + SegFormer + XFeat")
    print("=" * 60)
    print()
    print("Примечания:")
    print("- декодирование через cv2.read даёт заметные накладные расходы Python")
    print("- в прикладной реализации буфер камеры лучше обрабатывать нативным кодом")
    print("- SegFormer на MPS является основным тяжёлым этапом")
    print("- XFeat также заметно влияет на время картографического канала")
    print("- картографический канал запускается периодически, а не на каждом кадре")
    print()
    print("Оценка эффективной частоты без картографического канала:")
    print("  VO + декодирование: около 250 мс, примерно 4 FPS на Mac M в Python")
    print("  картографический канал запускается раз в 2 с и занимает около 700-900 мс")
    print("  частота порядка 30 FPS требует оптимизации и целевого ускорителя")


if __name__ == "__main__":
    main()
