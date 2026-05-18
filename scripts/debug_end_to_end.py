"""Visual diagnostic for the end-to-end smoke test.

Dumps for a single cruise frame:
  1. raw drone frame
  2. undistorted (rectified)
  3. BEV-warped at default pitch
  4. retriever's top-5 z=14 tiles
  5. MapManager z=17 window centred on the retriever's top-1

Reveals whether the failure is at retrieval (wrong tile picked) or
matching (right tile + right BEV but matcher can't bridge the modality
gap).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import certifi  # type: ignore
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except Exception:
    pass

from src.aircraft_mask import AircraftMaskTracker
from src.bev_rectifier import BevRectifier
from src.map_loader import MapDownloader
from src.map_manager import MapManager
from src.retriever import Retriever
from src.undistort import Undistorter
from src.video_processor import VideoProcessor


def _label(img, txt):
    o = img.copy()
    cv2.putText(o, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(o, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return o


def _resize_to_height(img, height):
    h, w = img.shape[:2]
    new_w = int(round(w * height / h))
    return cv2.resize(img, (new_w, height), interpolation=cv2.INTER_AREA)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, default=Path("data/videos/GP010269.MP4"))
    p.add_argument("--camera-config", type=Path, default=Path("configs/camera_gopro_hx.yaml"))
    p.add_argument("--anchors-dir", type=Path, default=Path("data/masks/anchors"))
    p.add_argument("--retriever-db", type=Path, default=Path("data/retrieval/db_z14"))
    p.add_argument("--frame-idx", type=int, default=2068,
                   help="cruise frame to inspect (default: best retriever score in smoke run)")
    p.add_argument("--output", type=Path, default=Path("results/end_to_end/debug"))
    p.add_argument("--zoom", type=int, default=17)
    p.add_argument("--window-radius", type=int, default=5)
    p.add_argument("--pitch-deg", type=float, default=-30.0)
    p.add_argument("--agl-m", type=float, default=620.0)
    p.add_argument("--ground-span-m", type=float, default=400.0)
    p.add_argument("--bev-out-size", type=int, default=600)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    proc = VideoProcessor(str(args.video), default_geo=(55.086025, 38.149033, 750.0))
    src_fps = proc.info.fps
    print(f"[debug] frame_idx = {args.frame_idx}  ({args.frame_idx / src_fps:.1f}s)")

    frame = proc.extract_frame(args.frame_idx)
    if frame is None:
        raise SystemExit("frame read failed")
    cv2.imwrite(str(args.output / "01_raw.png"), frame)

    undistorter = Undistorter.from_yaml(args.camera_config)
    rect = undistorter.undistort_image(frame)
    cv2.imwrite(str(args.output / "02_rectified.png"), rect)

    bev = BevRectifier.build(
        K_rect=undistorter.K_rect, image_size=undistorter.image_size,
        pitch_deg=args.pitch_deg, agl_m=args.agl_m,
        ground_span_m=args.ground_span_m,
        out_size=(args.bev_out_size, args.bev_out_size),
    )
    bev_img = bev.warp(rect)
    cv2.imwrite(str(args.output / "03_bev.png"), bev_img)

    retr = Retriever()
    retr.load_database(args.retriever_db)
    top = retr.query_image(frame, top_k=5)
    print("\n[debug] retriever top-5 (raw frame):")
    for i, t in enumerate(top):
        print(f"  {i}. score={t['score']:.3f}  tile={t['tile_id']}  "
              f"latlon=({t['lat']:.4f}, {t['lon']:.4f})")

    # Также пробуем retrieval на BEV-warped кадре: иногда это даёт лучший топ.
    top_bev = retr.query_image(bev_img, top_k=5)
    print("\n[debug] retriever top-5 (BEV-warped frame):")
    for i, t in enumerate(top_bev):
        print(f"  {i}. score={t['score']:.3f}  tile={t['tile_id']}  "
              f"latlon=({t['lat']:.4f}, {t['lon']:.4f})")

    # Скачиваем top-5 тайлов z=14 от retriever для визуального просмотра.
    loader = MapDownloader(zoom=14)
    panels = []
    for i, t in enumerate(top):
        z, tx, ty = (int(s) for s in t["tile_id"].split("/"))
        pil = loader.download_tile(tx, ty, z)
        if pil is None:
            continue
        tile_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        panels.append(_label(tile_bgr, f"#{i+1} {t['tile_id']} score={t['score']:.3f}"))
    if panels:
        contact = np.hstack([_resize_to_height(p, 240) for p in panels])
        cv2.imwrite(str(args.output / "04_retriever_top5.png"), contact)

    # Окно MapManager с центром в top-1 от retriever.
    mm = MapManager(zoom=args.zoom, window_radius=args.window_radius, closer_threshold=1)
    map_cv2, bbox = mm.initialize(top[0]["lat"], top[0]["lon"])
    print(f"\n[debug] MapManager z={args.zoom} window: shape={map_cv2.shape}  bbox={bbox}")
    # Уменьшаем для contact sheet, чтобы картинка не была гигантской.
    map_small = _resize_to_height(map_cv2, 600)
    cv2.imwrite(str(args.output / "05_mm_window.png"), map_small)

    # Рядом: BEV и окно MapManager — то, что реально видит матчер.
    bev_lbl = _label(bev_img, f"BEV (drone, pitch={args.pitch_deg}°)")
    mm_lbl = _label(map_small, f"MapManager z={args.zoom} window  (top1: {top[0]['tile_id']})")
    sheet = np.hstack([_resize_to_height(bev_lbl, 600), _resize_to_height(mm_lbl, 600)])
    cv2.imwrite(str(args.output / "06_bev_vs_window.png"), sheet)

    print(f"\n[debug] outputs -> {args.output}")
    for f in sorted(args.output.iterdir()):
        print(f"   {f.name}")


if __name__ == "__main__":
    main()
