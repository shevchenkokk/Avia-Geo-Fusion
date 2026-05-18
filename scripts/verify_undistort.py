"""Проверка этапа 3.1: убедиться, что канонический Undistorter корректно
проходит round-trip и даёт адекватный выпрямленный кадр на реальном видео
GP010269. Два этапа:

  Этап A — point round-trip. Берётся сетка 12×8 пикселей по raw-кадру, каждая
  точка проходит undistort, затем re-distort; проверяется, что восстановленный
  raw-пиксель совпадает с исходным в пределах субпиксельной ошибки. Так ловятся
  ошибки загрузки K/D или обратной формулы.

  Этап B — visual smoke. Берётся крейсерский кадр из GP010269.MP4 после наледи,
  выполняется undistort, raw и rectified сохраняются рядом, после чего
  проверяется, что выпрямленное изображение не вырождено: содержит данные, а
  не только чёрный цвет или NaN.

Выходы:
    results/stage3_1/roundtrip.txt
    results/stage3_1/before_after.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.undistort import Undistorter


def stage_a_roundtrip(undist: Undistorter) -> dict:
    w, h = undist.image_size
    xs = np.linspace(50, w - 50, 12)
    ys = np.linspace(50, h - 50, 8)
    X, Y = np.meshgrid(xs, ys)
    raw = np.column_stack([X.ravel(), Y.ravel()])

    rect = undist.undistort_points(raw)
    raw_back = undist.distort_points(rect)
    err = np.linalg.norm(raw_back - raw, axis=1)
    return {
        "n_points": len(raw),
        "max_err_px": float(err.max()),
        "mean_err_px": float(err.mean()),
    }


def stage_b_visual(
    undist: Undistorter, video_path: Path, frame_idx: int, out_path: Path,
) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return {"ok": False, "reason": "video read failed"}

    rect = undist.undistort_image(frame)
    h, w = frame.shape[:2]
    target_w = 800
    s = target_w / w
    th = int(h * s)
    raw_small = cv2.resize(frame, (target_w, th), interpolation=cv2.INTER_AREA)
    rect_small = cv2.resize(rect, (target_w, th), interpolation=cv2.INTER_AREA)

    # Подписываем обе панели для contact sheet.
    def _label(img, txt):
        out = img.copy()
        cv2.putText(out, txt, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, txt, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return out

    sheet = np.hstack([
        _label(raw_small, f"RAW   f={frame_idx}"),
        _label(rect_small, f"RECTIFIED  K_rect[0,0]={undist.K_rect[0,0]:.1f}"),
    ])
    cv2.imwrite(str(out_path), sheet)

    # Sanity-check: выпрямленный кадр должен содержать осмысленное изображение.
    rect_gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY) if rect.ndim == 3 else rect
    # Берём центр, чтобы пропустить чёрную рамку, возникающую при balance>0.
    cy0, cy1 = int(0.2 * h), int(0.8 * h)
    cx0, cx1 = int(0.2 * w), int(0.8 * w)
    centre = rect_gray[cy0:cy1, cx0:cx1]
    return {
        "ok": True,
        "raw_shape": frame.shape,
        "rect_shape": rect.shape,
        "centre_std": float(np.std(centre)),
        "centre_mean": float(np.mean(centre)),
        "out": str(out_path),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/camera_gopro_hx.yaml"))
    p.add_argument("--video", type=Path, default=Path("data/videos/GP010269.MP4"))
    p.add_argument("--frame", type=int, default=1500,
                   help="cruise frame index (post-frost)")
    p.add_argument("--output", type=Path, default=Path("results/stage3_1"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    undist = Undistorter.from_yaml(args.config)
    print(f"[verify] config       : {args.config}")
    print(f"[verify] image_size   : {undist.image_size}")
    print(f"[verify] K[0,0]       : {undist.K[0,0]:.2f}")
    print(f"[verify] K_rect[0,0]  : {undist.K_rect[0,0]:.2f}")

    sa = stage_a_roundtrip(undist)
    print()
    print("[stageA] point round-trip")
    print(f"  n_points    : {sa['n_points']}")
    print(f"  max  err px : {sa['max_err_px']:.4f}")
    print(f"  mean err px : {sa['mean_err_px']:.4f}")
    sa_ok = sa["max_err_px"] < 0.5
    print(f"  stageA -> {'OK' if sa_ok else 'FAIL'}")

    sb_path = args.output / "before_after.png"
    if args.video.exists():
        sb = stage_b_visual(undist, args.video, args.frame, sb_path)
        print()
        print("[stageB] visual smoke")
        if sb["ok"]:
            print(f"  raw  shape     : {sb['raw_shape']}")
            print(f"  rect shape     : {sb['rect_shape']}")
            print(f"  centre std/mean: {sb['centre_std']:.1f} / {sb['centre_mean']:.1f}")
            print(f"  output         : {sb['out']}")
            sb_ok = sb["centre_std"] > 5.0
        else:
            print(f"  {sb.get('reason')}")
            sb_ok = False
        print(f"  stageB -> {'OK' if sb_ok else 'FAIL'}")
    else:
        print(f"\n[stageB] video {args.video} not found — skipping")
        sb_ok = True  # advisory

    (args.output / "roundtrip.txt").write_text(
        f"max_err_px = {sa['max_err_px']:.6f}\nmean_err_px = {sa['mean_err_px']:.6f}\n",
        encoding="utf-8",
    )

    if sa_ok and sb_ok:
        print("\n[verify] CRITERION PASSED")
        return
    print("\n[verify] CRITERION NOT MET")
    sys.exit(2)


if __name__ == "__main__":
    main()
