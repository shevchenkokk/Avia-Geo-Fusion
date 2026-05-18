"""Визуальная проверка этапа 1.2: убеждаемся, что маска самолёта действительно
подавляет ключевые точки на фюзеляже внутри пайплайна матчинга.

Для каждого выбранного кадра скрипт:
  1. достаёт кадр из видео;
  2. просит ``AircraftMaskTracker`` вернуть маску ближайшего якоря;
  3. запускает ``NeuralMatcher.match(frame, map_tile, aircraft_mask=mask)``
     на технической карте-заглушке: здесь важны только ключевые точки со
     стороны самолёта;
  4. рисует кадр с красным контуром маски и зелёными сохранёнными точками
     ``mkpts0``;
  5. проверяет, что **ни одна сохранённая точка не лежит внутри маски**.

На выход пишется один общий PNG с отдельной строкой на каждый кадр, а в
консоль печатается сводка вида ``frame=... raw=... kept=... inside_mask=0``.

Критерий из PROJECT_PLAN.md §1.2: на отладочном оверлее inlier-точки не
должны попадать на корпус самолёта. Зелёная точка внутри красной границы на
итоговом изображении считается ошибкой.
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

from src.aircraft_mask import (
    load_aircraft_mask_tracker_for_video,
    filter_keypoints_by_mask,
)
from src.neural_matching import NeuralMatcher
from src.video_processor import VideoProcessor


def _draw_overlay(frame: np.ndarray, mask: np.ndarray, kp: np.ndarray) -> np.ndarray:
    out = frame.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 0, 255), 2)

    if kp is not None and len(kp) > 0:
        for x, y in kp.astype(np.int32):
            if 0 <= x < out.shape[1] and 0 <= y < out.shape[0]:
                cv2.circle(out, (int(x), int(y)), 3, (0, 255, 0), -1, cv2.LINE_AA)
    return out


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, default=Path("data/videos/GP010269.MP4"))
    p.add_argument("--anchors-dir", type=Path, default=Path("data/masks/anchors"))
    p.add_argument("--samples", type=int, nargs="+",
                   default=[3000, 5000, 7500, 10500, 13000, 15500, 17000],
                   help="frame indices to verify")
    p.add_argument("--output", type=Path, default=Path("results/stage1_2_mask_verify.png"))
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    processor = VideoProcessor(str(args.video), default_geo=(55.086025, 38.149033, 750.0))
    tracker = load_aircraft_mask_tracker_for_video(args.anchors_dir, args.video)
    if tracker is None:
        raise SystemExit(
            "aircraft-mask anchors do not match --video; generate anchors for "
            "this video or pass the matching video"
        )
    matcher = NeuralMatcher(backend="auto")
    print(f"[verify] anchors={tracker.num_anchors()} backend={matcher.backend}")

    # Приём: сопоставляем кадр с самим собой. Так матчер гарантированно
    # находит много соответствий по всему кадру, поэтому любая оставшаяся
    # точка внутри маски — это настоящая утечка маски, а не следствие
    # низкой плотности признаков.
    panels: list[np.ndarray] = []
    violations_total = 0
    for frame_idx in args.samples:
        frame = processor.extract_frame(frame_idx)
        if frame is None:
            print(f"[verify] frame={frame_idx}: read failed")
            continue
        mask = tracker.mask_for_frame(frame_idx, frame.shape[:2], frame=frame)
        anchor = tracker.anchor_for_frame(frame_idx)
        diag = tracker.last_diagnostics

        # Самосопоставление: с обеих сторон одно и то же изображение. Вход со стороны
        # спутника не маскируется, поэтому матчер мог бы вернуть точки в любой
        # части кадра. Подавлять точки на фюзеляже должна только маска со
        # стороны самолёта.
        result = matcher.match(frame, frame.copy(), aircraft_mask=mask)
        mkpts0 = result["mkpts0"]
        raw = int(result.get("raw_matches", 0))
        after_mask = int(result.get("matches_after_mask", raw))
        inliers = len(mkpts0)

        # Разница: тот же вызов, но без маски. Считаем, сколько inlier-точек
        # попало бы на фюзеляж без фильтра — это размер ошибки, которую мы
        # предотвращаем.
        result_nomask = matcher.match(frame, frame.copy(), aircraft_mask=None)
        kp_nomask = result_nomask["mkpts0"]
        if len(kp_nomask) > 0:
            inside_nomask = int((~filter_keypoints_by_mask(kp_nomask, mask)).sum())
        else:
            inside_nomask = 0

        # Прямая проверка критерия:
        if len(mkpts0) > 0:
            inside = ~filter_keypoints_by_mask(mkpts0, mask)
            n_inside = int(inside.sum())
        else:
            n_inside = 0
        violations_total += n_inside

        panel_full = _draw_overlay(frame, mask, mkpts0)
        # Уменьшаем исходный кадр 1920x1080 для contact sheet.
        target_w = 960
        target_h = int(panel_full.shape[0] * (target_w / panel_full.shape[1]))
        panel = cv2.resize(panel_full, (target_w, target_h), interpolation=cv2.INTER_AREA)
        panel = _label(
            panel,
            f"f={frame_idx} anchor=f{anchor.frame_index} "
            f"mask={diag.method} conf={diag.confidence:.2f} raw={raw} after_mask={after_mask} inliers={inliers} "
            f"inside_mask={n_inside}  (without mask, would be inside={inside_nomask})",
        )
        panels.append(panel)
        print(
            f"[verify] f={frame_idx:>6}  anchor=f{anchor.frame_index:<6}  "
            f"mask={diag.method:<15} conf={diag.confidence:.2f} "
            f"raw={raw:<5} after_mask={after_mask:<5} inliers={inliers:<4} "
            f"inside_mask={n_inside}  inside_nomask={inside_nomask}"
        )

    if not panels:
        raise SystemExit("no frames sampled")

    sheet = np.concatenate(panels, axis=0)
    cv2.imwrite(str(args.output), sheet)
    print(f"\n[verify] composite -> {args.output}")
    print(f"[verify] TOTAL inside-mask keypoints across all samples: {violations_total}")
    if violations_total > 0:
        print("[verify] CRITERION FAILED: at least one inlier landed on aircraft body")
        sys.exit(2)
    print("[verify] CRITERION PASSED: zero inliers on aircraft body across samples")


if __name__ == "__main__":
    main()
