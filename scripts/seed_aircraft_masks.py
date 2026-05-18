"""Этап 0b: начальные маски корпуса самолёта на репрезентативных кадрах.

Запускает SAM 3 с текстовыми prompt'ами ("aircraft fuselage", "landing gear",
"wing strut") на стратифицированной выборке кадров из исходного полётного
видео. Для каждого якорного кадра сохраняются:

  * undistorted RGB-изображение с fisheye-профилем, выбранным на этапе 0a;
  * бинарная uint8-маска в выпрямленном кадре: 255 = aircraft, 0 = background;
  * overlay-изображение с подсвеченной маской для ручного QA;
  * JSON sidecar на кадр с boxes/scores/metadata.

Верхнеуровневый ``index.json`` перечисляет все якоря и отмечает те, которым
нужна ручная проверка: ноль детекций или детекции, отброшенные left-strip prior.

Так закрывается deliverable этапа 0b: ``data/masks/anchors/`` заполнен
начальными масками и overlay для быстрой ручной проверки перед smoke-тестом
этапа 0.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from transformers import Sam3Model, Sam3Processor


@dataclass
class CameraProfile:
    k: np.ndarray
    d: np.ndarray
    k_rectified: np.ndarray
    image_size: tuple[int, int]
    selected_profile: str


def _load_camera_config(path: Path) -> CameraProfile:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    k = np.asarray(data["K"], dtype=np.float64)
    d = np.asarray(data["D"], dtype=np.float64).reshape(4, 1)
    w, h = data["image_size"]
    if "K_rectified" in data:
        k_rect = np.asarray(data["K_rectified"], dtype=np.float64)
    else:
        k_rect = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            k, d, (int(w), int(h)), np.eye(3), balance=0.0, new_size=(int(w), int(h))
        )
    return CameraProfile(
        k=k,
        d=d,
        k_rectified=np.asarray(k_rect, dtype=np.float64),
        image_size=(int(w), int(h)),
        selected_profile=str(data.get("selected_profile", "unknown")),
    )


def _pick_frame_indices(
    total_frames: int,
    fps: float,
    num_frames: int,
    warmup_seconds: float = 2.0,
    cooldown_seconds: float = 2.0,
) -> list[int]:
    """Стратифицированная равномерная выборка с небольшим jitter.

    Первые и последние пару секунд пропускаются: динамика взлёта/посадки шумная.
    """
    if total_frames <= 0:
        raise ValueError("video has zero frames")

    start = int(min(total_frames - 1, max(0, warmup_seconds * fps)))
    end = int(max(0, total_frames - 1 - cooldown_seconds * fps))
    if end <= start:
        start, end = 0, total_frames - 1

    if num_frames <= 1:
        return [start]

    step = (end - start) / (num_frames - 1)
    rng = np.random.default_rng(seed=17)
    indices: list[int] = []
    for i in range(num_frames):
        base = start + i * step
        jitter = rng.integers(-int(step * 0.15), int(step * 0.15) + 1) if step > 4 else 0
        idx = int(np.clip(round(base + jitter), start, end))
        indices.append(idx)
    return sorted(set(indices))


def _build_undistort_maps(profile: CameraProfile) -> tuple[np.ndarray, np.ndarray]:
    w, h = profile.image_size
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        profile.k,
        profile.d,
        np.eye(3),
        profile.k_rectified,
        (w, h),
        cv2.CV_16SC2,
    )
    return map1, map2


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _mask_overlay(bgr: np.ndarray, mask: np.ndarray, colour=(0, 128, 255), alpha: float = 0.45) -> np.ndarray:
    out = bgr.copy()
    m = mask.astype(bool)
    if m.any():
        tint = np.zeros_like(bgr)
        tint[:] = colour
        out[m] = (alpha * tint[m] + (1.0 - alpha) * bgr[m]).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, colour, 2)
    return out


def _union_masks(
    masks: torch.Tensor | np.ndarray,
    scores: torch.Tensor | np.ndarray,
    boxes: torch.Tensor | np.ndarray,
    image_shape: tuple[int, int],
    score_threshold: float,
    left_only_frac: float,
    max_box_right_frac: float,
) -> tuple[np.ndarray, list[dict]]:
    """Combine per-instance masks into a single aircraft mask.

    Rejects detections whose bounding-box centroid lies to the right of
    ``left_only_frac`` of the image width — a simple geometric prior: on this
    flight the camera is mounted on the left side and the aircraft body is
    always in the left portion of the frame.
    """
    h, w = image_shape
    union = np.zeros((h, w), dtype=np.uint8)
    kept: list[dict] = []

    if isinstance(masks, torch.Tensor):
        masks_np = masks.detach().cpu().numpy()
    else:
        masks_np = np.asarray(masks)
    if isinstance(scores, torch.Tensor):
        scores_np = scores.detach().cpu().numpy()
    else:
        scores_np = np.asarray(scores)
    if isinstance(boxes, torch.Tensor):
        boxes_np = boxes.detach().cpu().numpy()
    else:
        boxes_np = np.asarray(boxes)

    if masks_np.ndim == 4:
        masks_np = masks_np[0]

    for i, (mask_i, score_i, box_i) in enumerate(zip(masks_np, scores_np, boxes_np)):
        score = float(score_i)
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = (float(v) for v in box_i.tolist())
        cx = 0.5 * (x1 + x2) / max(1.0, float(w))
        x2_frac = x2 / max(1.0, float(w))
        if cx > left_only_frac:
            kept.append(
                {
                    "index": int(i),
                    "score": score,
                    "box": [x1, y1, x2, y2],
                    "cx_frac": cx,
                    "accepted": False,
                    "reason": "right_of_left_prior",
                }
            )
            continue
        if x2_frac > max_box_right_frac:
            kept.append(
                {
                    "index": int(i),
                    "score": score,
                    "box": [x1, y1, x2, y2],
                    "cx_frac": cx,
                    "x2_frac": x2_frac,
                    "accepted": False,
                    "reason": "box_extends_past_max_right",
                }
            )
            continue

        if mask_i.shape != (h, w):
            mask_bin = cv2.resize(
                mask_i.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            mask_bin = mask_i.astype(bool)
        union[mask_bin] = 255
        kept.append(
            {
                "index": int(i),
                "score": score,
                "box": [x1, y1, x2, y2],
                "cx_frac": cx,
                "accepted": True,
            }
        )
    return union, kept


def _dilate(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return mask
    ksize = 2 * radius_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.dilate(mask, kernel)


def _close_small_gaps(mask: np.ndarray, radius_px: int = 6) -> np.ndarray:
    if radius_px <= 0:
        return mask
    ksize = 2 * radius_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _mask_coverage(mask: np.ndarray) -> float:
    return float(mask.astype(bool).sum()) / float(mask.size)


def run(args: argparse.Namespace) -> None:
    video_path = Path(args.video)
    camera_cfg = Path(args.camera_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = _load_camera_config(camera_cfg)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"failed to open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if res != profile.image_size:
        print(
            f"[seed] warning: camera config image_size={profile.image_size} "
            f"differs from video resolution {res}"
        )

    frame_indices = _pick_frame_indices(
        total_frames=total_frames,
        fps=fps,
        num_frames=args.num_frames,
        warmup_seconds=args.warmup_seconds,
        cooldown_seconds=args.cooldown_seconds,
    )
    print(f"[seed] video={video_path} frames={total_frames} fps={fps:.3f}")
    print(f"[seed] sampling {len(frame_indices)} anchor indices")

    map1, map2 = _build_undistort_maps(profile)

    device = _resolve_device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"[seed] loading {args.model_id} onto {device} (dtype={dtype})")
    processor = Sam3Processor.from_pretrained(args.model_id)
    model = Sam3Model.from_pretrained(args.model_id, dtype=dtype)
    model = model.to(device)
    model.eval()

    prompts = [p.strip() for p in args.text_prompt.split("|") if p.strip()]
    print(f"[seed] prompts: {prompts}")

    index_entries: list[dict] = []
    saved_count = 0
    empty_count = 0

    for anchor_i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[seed] frame {frame_idx}: read failed, skipping")
            continue

        undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
        pil = _to_pil(undistorted)
        h, w = undistorted.shape[:2]

        per_prompt_masks: list[np.ndarray] = []
        per_prompt_detections: list[dict] = []

        for prompt in prompts:
            inputs = processor(images=pil, text=prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=args.score_threshold,
                mask_threshold=args.mask_threshold,
                target_sizes=[(h, w)],
            )[0]
            union_mask, kept = _union_masks(
                results["masks"],
                results["scores"],
                results["boxes"],
                image_shape=(h, w),
                score_threshold=args.score_threshold,
                left_only_frac=args.left_only_frac,
                max_box_right_frac=args.max_box_right_frac,
            )
            per_prompt_masks.append(union_mask)
            per_prompt_detections.append(
                {
                    "prompt": prompt,
                    "instances": kept,
                    "num_accepted": sum(1 for k in kept if k.get("accepted")),
                }
            )

        merged = np.zeros((h, w), dtype=np.uint8)
        for m in per_prompt_masks:
            merged = np.maximum(merged, m)

        merged = _close_small_gaps(merged, radius_px=args.close_radius_px)
        merged = _dilate(merged, radius_px=args.dilation_px)

        coverage = _mask_coverage(merged)
        accepted_total = sum(d["num_accepted"] for d in per_prompt_detections)
        needs_review = accepted_total == 0 or coverage < args.min_coverage or coverage > args.max_coverage

        stem = f"anchor_{anchor_i:03d}_f{frame_idx:06d}"
        frame_path = output_dir / f"{stem}_frame.jpg"
        mask_path = output_dir / f"{stem}_mask.png"
        overlay_path = output_dir / f"{stem}_overlay.jpg"
        meta_path = output_dir / f"{stem}_meta.json"

        cv2.imwrite(str(frame_path), undistorted, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(mask_path), merged)
        cv2.imwrite(str(overlay_path), _mask_overlay(undistorted, merged), [cv2.IMWRITE_JPEG_QUALITY, 88])

        meta = {
            "anchor_index": anchor_i,
            "frame_index": int(frame_idx),
            "timestamp_seconds": float(frame_idx) / max(fps, 1e-6),
            "image_size": [w, h],
            "mask_coverage": coverage,
            "num_accepted_instances": accepted_total,
            "needs_review": bool(needs_review),
            "detections": per_prompt_detections,
            "camera_profile": profile.selected_profile,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        index_entries.append(
            {
                "anchor_index": anchor_i,
                "frame_index": int(frame_idx),
                "timestamp_seconds": meta["timestamp_seconds"],
                "frame": frame_path.name,
                "mask": mask_path.name,
                "overlay": overlay_path.name,
                "meta": meta_path.name,
                "mask_coverage": coverage,
                "num_accepted_instances": accepted_total,
                "needs_review": bool(needs_review),
            }
        )

        saved_count += 1
        if accepted_total == 0:
            empty_count += 1
        print(
            f"[seed] anchor {anchor_i:03d} frame={frame_idx} coverage={coverage*100:.2f}% "
            f"instances={accepted_total} {'NEEDS-REVIEW' if needs_review else ''}"
        )

    cap.release()

    review_count = sum(1 for e in index_entries if e["needs_review"])
    index_payload = {
        "video": str(video_path),
        "camera_config": str(camera_cfg),
        "camera_profile": profile.selected_profile,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "prompts": prompts,
        "score_threshold": args.score_threshold,
        "mask_threshold": args.mask_threshold,
        "dilation_px": args.dilation_px,
        "close_radius_px": args.close_radius_px,
        "left_only_frac": args.left_only_frac,
        "num_anchors": saved_count,
        "num_empty_detections": empty_count,
        "num_needs_review": review_count,
        "anchors": index_entries,
    }
    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")

    print(
        f"[seed] wrote {saved_count} anchors to {output_dir} "
        f"(empty={empty_count}, needs_review={review_count})"
    )
    print(f"[seed] index: {index_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=Path("data/videos/GP010269.MP4"))
    parser.add_argument("--camera-config", type=Path, default=Path("configs/camera_gopro_hx.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/masks/anchors"))
    parser.add_argument("--num-frames", type=int, default=30)
    parser.add_argument(
        "--text-prompt",
        type=str,
        default="plane|aircraft skin|metal panel|landing gear|wing strut",
        help="Pipe-separated list of SAM3 phrase prompts. Each phrase is "
        "run independently and the per-prompt masks are unioned. "
        "On this footage 'aircraft fuselage' fails (the visible strip is a "
        "textureless white panel); 'plane' / 'aircraft skin' / 'metal panel' "
        "do catch the fuselage edge, while 'landing gear' robustly catches the wheel.",
    )
    parser.add_argument("--score-threshold", type=float, default=0.2)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--dilation-px", type=int, default=25)
    parser.add_argument("--close-radius-px", type=int, default=6)
    parser.add_argument(
        "--left-only-frac",
        type=float,
        default=0.18,
        help="Reject detections whose box centroid sits to the right of this "
        "fraction of image width. Camera is mounted on the left side: in the "
        "undistorted frame the aircraft body strip is at cx<3%% and the visible "
        "landing-gear wheel sits at cx<10%%. Anything past 18%% is the ground.",
    )
    parser.add_argument(
        "--max-box-right-frac",
        type=float,
        default=0.20,
        help="Reject detections whose box right-edge extends past this fraction "
        "of image width. Real aircraft-body boxes end at x<13%; broad blobs "
        "spanning sky/cloud/terrain typically reach >25%.",
    )
    parser.add_argument("--min-coverage", type=float, default=0.02,
                        help="Flag frames whose mask covers less than this fraction for review.")
    parser.add_argument("--max-coverage", type=float, default=0.45,
                        help="Flag frames whose mask covers more than this fraction for review (false positive blew up).")
    parser.add_argument("--warmup-seconds", type=float, default=20.0,
                        help="Skip the first N seconds (takeoff/ground motion).")
    parser.add_argument("--cooldown-seconds", type=float, default=30.0,
                        help="Skip the last N seconds (landing/post-flight ground).")
    parser.add_argument("--model-id", type=str, default="facebook/sam3")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


if __name__ == "__main__":
    run(_parse_args())
