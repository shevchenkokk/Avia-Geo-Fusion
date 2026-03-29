"""Stage 0 smoke test: render a short clip with fisheye undistort + the
nearest seed aircraft mask overlaid.

Inputs:
  * source video                          (data/videos/GP010269.MP4)
  * fisheye intrinsics from Stage 0a      (configs/camera_gopro_hx.yaml)
  * seed masks from Stage 0b              (data/masks/anchors/index.json)

Output:
  * an mp4 covering ``--duration`` seconds starting at ``--start-time`` with
    side-by-side panels: original frame on the left, undistorted-with-mask
    on the right. Lets us eyeball that the fisheye straightens and the
    aircraft body stays under the mask across a continuous segment.

The mask shown on each output frame is the seed mask of the temporally
closest anchor. This is a sanity check, not the runtime mask: real online
propagation (Stage 1.2) will follow the body via optical flow. The fact
that nearest-anchor masks visibly track the body across a 30 s window
already tells us Stage 0b masks are temporally usable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


def _load_camera(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    K = np.asarray(cfg["K"], dtype=np.float64)
    D = np.asarray(cfg["D"], dtype=np.float64).reshape(4, 1)
    Knew = np.asarray(cfg["K_rectified"], dtype=np.float64)
    w, h = cfg["image_size"]
    return K, D, Knew, (int(w), int(h))


def _load_anchors(index_path: Path) -> list[dict]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    anchors = sorted(payload["anchors"], key=lambda a: a["frame_index"])
    return anchors


def _nearest_anchor(frame_idx: int, anchors: list[dict]) -> dict:
    best = anchors[0]
    best_dist = abs(best["frame_index"] - frame_idx)
    for a in anchors[1:]:
        d = abs(a["frame_index"] - frame_idx)
        if d < best_dist:
            best, best_dist = a, d
    return best


def _overlay(frame: np.ndarray, mask: np.ndarray, colour=(0, 128, 255), alpha=0.40) -> np.ndarray:
    out = frame.copy()
    m = mask.astype(bool)
    if m.any():
        tint = np.zeros_like(frame)
        tint[:] = colour
        out[m] = (alpha * tint[m] + (1.0 - alpha) * frame[m]).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, colour, 2)
    return out


def _label(img: np.ndarray, text: str, origin=(20, 40)) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def run(args: argparse.Namespace) -> None:
    video_path = Path(args.video)
    cam_cfg_path = Path(args.camera_config)
    anchors_dir = Path(args.anchors_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    K, D, Knew, (w, h) = _load_camera(cam_cfg_path)
    anchors = _load_anchors(anchors_dir / "index.json")
    print(f"[smoke] loaded {len(anchors)} seed anchors")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(round(args.start_time * fps))
    end_frame = int(round((args.start_time + args.duration) * fps))
    end_frame = min(end_frame, total_frames - 1)
    if end_frame <= start_frame:
        raise SystemExit("requested clip is empty")

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), Knew, (w, h), cv2.CV_16SC2
    )

    panel_w = w // 2
    panel_h = h // 2
    out_w, out_h = panel_w * 2 + 8, panel_h
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, args.output_fps, (out_w, out_h))
    if not writer.isOpened():
        raise SystemExit(f"failed to open writer: {output_path}")

    sample_step = max(1, int(round(fps / args.output_fps)))
    print(
        f"[smoke] clip start={args.start_time}s end={args.start_time + args.duration}s "
        f"src_fps={fps:.2f} sample_step={sample_step} out_fps={args.output_fps}"
    )

    mask_cache: dict[str, np.ndarray] = {}
    written = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    cur = start_frame
    while cur <= end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(cur))
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[smoke] frame {cur}: read failed; stopping")
            break

        undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
        anchor = _nearest_anchor(cur, anchors)
        mask_path = anchors_dir / anchor["mask"]
        if mask_path.name not in mask_cache:
            m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if m is None or m.shape[:2] != (h, w):
                if m is not None:
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                else:
                    m = np.zeros((h, w), dtype=np.uint8)
            mask_cache[mask_path.name] = m
        mask = mask_cache[mask_path.name]
        overlaid = _overlay(undistorted, mask)

        left = cv2.resize(frame, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        right = cv2.resize(overlaid, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        ts = cur / max(fps, 1e-6)
        left = _label(left, f"raw fisheye  t={ts:5.1f}s f={cur}", origin=(15, 30))
        right = _label(
            right,
            f"undistort + mask  anchor f={anchor['frame_index']} ({mask_path.name})",
            origin=(15, 30),
        )
        gap = np.zeros((panel_h, 8, 3), dtype=np.uint8)
        composite = np.concatenate([left, gap, right], axis=1)
        writer.write(composite)
        written += 1
        cur += sample_step

    cap.release()
    writer.release()
    print(f"[smoke] wrote {written} frames -> {output_path}")
    print(f"[smoke] duration={written / max(args.output_fps, 1e-6):.1f}s @ {args.output_fps} fps")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=Path("data/videos/GP010269.MP4"))
    parser.add_argument("--camera-config", type=Path, default=Path("configs/camera_gopro_hx.yaml"))
    parser.add_argument("--anchors-dir", type=Path, default=Path("data/masks/anchors"))
    parser.add_argument("--start-time", type=float, default=200.0,
                        help="Clip start (seconds from video start). Default 200s = mid-cruise.")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--output-fps", type=float, default=10.0)
    parser.add_argument("--output-path", type=Path, default=Path("results/stage0_smoke.mp4"))
    return parser.parse_args()


if __name__ == "__main__":
    run(_parse_args())
