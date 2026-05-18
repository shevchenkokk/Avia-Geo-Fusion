"""Проверка переноса маски корпуса по оптическому потоку.

Полное видео GP010269 не хранится в репозитории, поэтому проверка использует
сохранённые якорные кадры этапа 0b. Скрипт применяет к якорям небольшие
известные аффинные преобразования, запускает онлайн-перенос маски и сравнивает
результат с тем же преобразованием, применённым к исходной маске.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aircraft_mask import AircraftMaskTracker


def _overlay(frame: np.ndarray, expected: np.ndarray, propagated: np.ndarray, label: str) -> np.ndarray:
    out = frame.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    exp_contours, _ = cv2.findContours(expected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    got_contours, _ = cv2.findContours(propagated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, exp_contours, -1, (0, 255, 255), 2)
    cv2.drawContours(out, got_contours, -1, (0, 0, 255), 2)
    cv2.putText(out, label, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, label, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a > 0
    bb = b > 0
    union = np.logical_or(aa, bb).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(aa, bb).sum() / union)


def _affine(shape: tuple[int, int], dx: float, dy: float, angle_deg: float) -> np.ndarray:
    h, w = shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    m[0, 2] += dx
    m[1, 2] += dy
    return m


def _warp(img: np.ndarray, m: np.ndarray, interp: int) -> np.ndarray:
    h, w = img.shape[:2]
    border = (0, 0, 0) if img.ndim == 3 else 0
    return cv2.warpAffine(img, m, (w, h), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anchors-dir", type=Path, default=Path("data/masks/anchors"))
    p.add_argument("--output", type=Path, default=Path("results/stage1_2_mask_propagation.png"))
    p.add_argument("--summary-json", type=Path, default=Path("results/stage1_2_mask_propagation.json"))
    p.add_argument("--max-anchors", type=int, default=12)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--min-iou", type=float, default=0.82)
    args = p.parse_args()
    if args.max_anchors < 1:
        p.error("--max-anchors must be >= 1")
    if args.steps < 1:
        p.error("--steps must be >= 1")
    if not (0.0 <= args.min_iou <= 1.0):
        p.error("--min-iou must be in [0, 1]")

    payload = json.loads((args.anchors_dir / "index.json").read_text(encoding="utf-8"))
    anchors = payload["anchors"][: args.max_anchors]
    if not anchors:
        raise SystemExit("no anchors found")

    transforms = [
        (-18.0, 2.0, -1.0),
        (12.0, 3.0, 0.8),
        (-8.0, -4.0, 1.2),
        (16.0, -2.0, -0.7),
    ]
    panels: list[np.ndarray] = []
    rows: list[dict] = []

    for idx, anchor_meta in enumerate(anchors):
        frame = cv2.imread(str(args.anchors_dir / anchor_meta["frame"]))
        mask = cv2.imread(str(args.anchors_dir / anchor_meta["mask"]), cv2.IMREAD_GRAYSCALE)
        if frame is None or mask is None:
            raise SystemExit(f"failed to read anchor {anchor_meta['anchor_index']}")
        dx, dy, angle = transforms[idx % len(transforms)]
        tracker = AircraftMaskTracker.from_index(args.anchors_dir)
        frame_idx = int(anchor_meta["frame_index"])
        _ = tracker.mask_for_frame(frame_idx, frame.shape[:2], frame=frame)
        propagated = mask
        expected = mask
        cur_frame = frame
        for step in range(1, args.steps + 1):
            alpha = step / args.steps
            m = _affine(frame.shape, dx * alpha, dy * alpha, angle * alpha)
            cur_frame = _warp(frame, m, cv2.INTER_LINEAR)
            expected = _warp(mask, m, cv2.INTER_NEAREST)
            propagated = tracker.mask_for_frame(frame_idx + step, cur_frame.shape[:2], frame=cur_frame)

        diag = tracker.last_diagnostics
        iou = _iou(expected, propagated)
        row = {
            "anchor": int(anchor_meta["anchor_index"]),
            "frame_index": frame_idx,
            "method": diag.method,
            "confidence": diag.confidence,
            "tracked_points": diag.num_tracked_points,
            "median_flow_px": diag.median_flow_px,
            "coverage": diag.coverage,
            "iou_vs_expected_transform": iou,
            "reject_reason": diag.reject_reason,
        }
        rows.append(row)
        print(
            f"[verify] anchor={row['anchor']:02d} method={diag.method:<15} "
            f"conf={diag.confidence:.2f} tracked={diag.num_tracked_points:<3} "
            f"flow={diag.median_flow_px:.1f}px iou={iou:.3f} reason={diag.reject_reason}"
        )
        panel = _overlay(
            cur_frame,
            expected,
            propagated,
            f"a{row['anchor']:02d} {diag.method} conf={diag.confidence:.2f} IoU={iou:.2f}",
        )
        panels.append(cv2.resize(panel, (640, 360), interpolation=cv2.INTER_AREA))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    cols = 2
    rows_img = []
    for i in range(0, len(panels), cols):
        row_panels = panels[i:i + cols]
        if len(row_panels) < cols:
            row_panels.append(np.full_like(row_panels[0], 255))
        rows_img.append(np.concatenate(row_panels, axis=1))
    ok = cv2.imwrite(str(args.output), np.concatenate(rows_img, axis=0))
    if not ok:
        raise SystemExit(f"[verify] failed to write overlay: {args.output}")

    ious = [r["iou_vs_expected_transform"] for r in rows]
    passed = all(r["method"] == "of_propagated" and r["iou_vs_expected_transform"] >= args.min_iou for r in rows)
    summary = {
        "passed": passed,
        "min_iou": args.min_iou,
        "num_anchors": len(rows),
        "mean_iou": float(np.mean(ious)),
        "min_observed_iou": float(np.min(ious)),
        "pairs": rows,
        "overlay": str(args.output),
    }
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[verify] overlay -> {args.output}")
    print(f"[verify] summary -> {args.summary_json}")
    if not passed:
        raise SystemExit("[verify] критерий проверки не выполнен")
    print("[verify] CRITERION PASSED")


if __name__ == "__main__":
    main()
