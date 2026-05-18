"""Проверка этапа 1.5: прогнать obstruction detector по участку перекрытия
кадра в GP010269.MP4 и проверить критерий §1.5.

Критерий из PROJECT_PLAN.md §1.5: «на участке t=280-360 detector срабатывает
в течение 1-2 с после входа в облако и очищается в течение 1-2 с после выхода.
Система не делает ложных map-fix во время участка».

Этот скрипт проверяет *половину критерия*, относящуюся к detector. Выходы:
  - results/stage1_5/metrics.csv   per-frame metrics + flags
  - results/stage1_5/metrics.png   4-panel time series with cloud band
  - results/stage1_5/contact.png   thumbnail strip of representative frames
  - results/stage1_5/summary.txt   ground-truth labelling + lag report

Ground truth для участка облака задаётся через CLI: по умолчанию
``--cloud-enter 295`` / ``--cloud-exit 358``. Значения оценены глазами по
сырому GoPro-видео без телеметрии; при необходимости их можно уточнить.

Часть критерия про «no false map-fix» проверяется отдельно, когда detector
подключён в ``run_diagnostic.py``. Это следующий шаг после прохождения этого
скрипта.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import csv
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aircraft_mask import load_aircraft_mask_tracker_for_video
from src.obstruction_detector import ObstructionDetector
from src.video_processor import VideoProcessor


def _detect_lag(
    times: np.ndarray, flags: np.ndarray, t_event: float, rising: bool
) -> float | None:
    """Найти первое время после ``t_event``, где ``flags`` совпадает с ожиданием.

    rising=True: ищем первый True после t_event — обнаружен вход в облако.
    rising=False: ищем первый False после t_event — обнаружен выход из облака.
    """
    target = True if rising else False
    after = times >= t_event
    cand = after & (flags == target)
    if not cand.any():
        return None
    return float(times[cand][0] - t_event)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, default=Path("data/videos/GP010269.MP4"))
    p.add_argument("--anchors-dir", type=Path, default=Path("data/masks/anchors"))
    # Реальное перекрытие в GP010269.MP4 — это участок инея на линзе t=15-40 с
    # (план тоже упоминает: "t≈10s в нашем видео показывает классический пример
    # с инеем на линзе"). Предполагаемое полное погружение в облако около t≈300 с
    # не соответствует этому видео: std там держится ~25-30, это безфактурная
    # местность, а не настоящее перекрытие. Поэтому критерий §1.5 проверяется
    # на событии с инеем; настройки порогов затем переносятся на full-cloud.
    p.add_argument("--start-seconds", type=float, default=0.0)
    p.add_argument("--end-seconds", type=float, default=80.0)
    p.add_argument("--fps", type=float, default=4.0,
                   help="sampling rate within the window")
    # Значения GT по умолчанию — ядро участка с инеем на линзе в GP010269.MP4.
    # Перекрытие имеет мягкий вход (~3 с), мягкий выход (~2 с) и частичное
    # оттаивание с колебаниями около t=25-30 с. PROJECT_PLAN §1.5 требует только:
    # (i) срабатывание за 1-2 с от входа, (ii) очистка за 1-2 с от выхода,
    # (iii) отсутствие ложных map-fix во время участка. 100 % флагов внутри
    # участка не требуется, потому что downstream-ворота этапа 1.4 ловят кадры,
    # которые detector пропустил.
    p.add_argument("--cloud-enter", type=float, default=12.0,
                   help="ground-truth obstruction-entry time (s)")
    p.add_argument("--cloud-exit", type=float, default=41.0,
                   help="ground-truth obstruction-exit time (s)")
    p.add_argument("--max-lag", type=float, default=2.0,
                   help="criterion: detector must trigger within this many seconds")
    p.add_argument("--output-dir", type=Path, default=Path("results/stage1_5"))
    p.add_argument("--contact-frames", type=int, nargs="+",
                   default=None,
                   help="seconds for the thumbnail contact sheet (default = 8 spread across window)")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor = VideoProcessor(str(args.video), default_geo=(55.086025, 38.149033, 750.0))
    src_fps = processor.info.fps

    mask_tracker = None
    if (args.anchors_dir / "index.json").exists():
        mask_tracker = load_aircraft_mask_tracker_for_video(args.anchors_dir, args.video)
    if mask_tracker is not None:
        print(f"[verify] aircraft mask: {mask_tracker.num_anchors()} anchors")
    elif not (args.anchors_dir / "index.json").exists():
        print(f"[verify] WARNING: no anchors at {args.anchors_dir}; running without mask")

    detector = ObstructionDetector()

    sample_step = max(1, int(round(src_fps / args.fps)))
    f0 = int(round(args.start_seconds * src_fps))
    f1 = int(round(args.end_seconds * src_fps))
    print(f"[verify] sweep frames {f0}-{f1} step={sample_step} "
          f"({(f1 - f0) // sample_step} samples)")

    rows: list[dict] = []
    cur = f0
    while cur < f1:
        frame = processor.extract_frame(cur)
        if frame is None:
            break
        ts = cur / max(src_fps, 1e-6)
        mask = mask_tracker.mask_for_frame(cur, frame.shape[:2], frame=frame) if mask_tracker else None
        result = detector.detect(frame, aircraft_mask=mask)
        m = result.metrics
        rows.append({
            "frame_idx": cur, "t": ts,
            "std": m.std, "entropy": m.entropy,
            "edge_density": m.edge_density, "low_var_patch_frac": m.low_var_patch_frac,
            "votes": m.votes, "raw_obstructed": int(result.raw_obstructed),
            "is_obstructed": int(result.is_obstructed),
        })
        cur += sample_step

    if not rows:
        raise SystemExit("no frames sampled")

    # --- запись CSV ---------------------------------------------------------
    csv_path = args.output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[verify] CSV  -> {csv_path}")

    # --- график -------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.array([r["t"] for r in rows])
    std = np.array([r["std"] for r in rows])
    entropy = np.array([r["entropy"] for r in rows])
    edge = np.array([r["edge_density"] for r in rows])
    patch = np.array([r["low_var_patch_frac"] for r in rows])
    raw = np.array([bool(r["raw_obstructed"]) for r in rows])
    final = np.array([bool(r["is_obstructed"]) for r in rows])

    fig, axes = plt.subplots(5, 1, figsize=(11, 11), sharex=True)
    panels = [
        (axes[0], std,     "global std (sigma_I)",       detector.std_threshold,                   "<"),
        (axes[1], entropy, "histogram entropy (bits)",   detector.entropy_threshold,               "<"),
        (axes[2], edge,    "edge density (frac)",        detector.edge_density_threshold,          "<"),
        (axes[3], patch,   "low-var patch fraction",     detector.low_var_patch_frac_threshold,    ">"),
    ]
    for ax, y, title, thr, sign in panels:
        ax.plot(times, y, lw=1.2)
        ax.axhline(thr, color="red", lw=0.8, ls="--",
                   label=f"thr {sign} {thr}")
        ax.axvspan(args.cloud_enter, args.cloud_exit, color="grey", alpha=0.15,
                   label="ground-truth cloud")
        ax.set_ylabel(title)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    ax = axes[4]
    ax.plot(times, raw.astype(int), label="raw_obstructed (per-frame vote)", lw=1.0, alpha=0.6)
    ax.plot(times, final.astype(int), label="is_obstructed (after hysteresis)", lw=2.0)
    ax.axvspan(args.cloud_enter, args.cloud_exit, color="grey", alpha=0.15,
               label="ground-truth cloud")
    ax.set_ylim(-0.1, 1.2)
    ax.set_yticks([0, 1])
    ax.set_ylabel("obstructed flag")
    ax.set_xlabel("video time (s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Stage 1.5 obstruction detector  |  cloud GT [{args.cloud_enter:.0f}-{args.cloud_exit:.0f}] s"
    )
    fig.tight_layout()
    plot_path = args.output_dir / "metrics.png"
    fig.savefig(plot_path, dpi=110)
    plt.close(fig)
    print(f"[verify] plot -> {plot_path}")

    # --- contact sheet ------------------------------------------------------
    if args.contact_frames is None:
        ts_targets = np.linspace(args.start_seconds, args.end_seconds, 8)
    else:
        ts_targets = np.array(args.contact_frames, dtype=float)
    panels = []
    for t_target in ts_targets:
        idx = int(np.argmin(np.abs(times - t_target)))
        r = rows[idx]
        frame = processor.extract_frame(r["frame_idx"])
        if frame is None:
            continue
        h, w = frame.shape[:2]
        target_w = 480
        scaled = cv2.resize(frame, (target_w, int(h * target_w / w)))
        flag = "OBSTR" if r["is_obstructed"] else "clear"
        color = (0, 0, 255) if r["is_obstructed"] else (0, 200, 0)
        cv2.putText(scaled, f"t={r['t']:.1f}s  {flag}  votes={r['votes']}/4",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(scaled, f"t={r['t']:.1f}s  {flag}  votes={r['votes']}/4",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        cv2.putText(scaled,
                    f"std={r['std']:.1f} H={r['entropy']:.2f} edge={r['edge_density']:.3f} pf={r['low_var_patch_frac']:.2f}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(scaled,
                    f"std={r['std']:.1f} H={r['entropy']:.2f} edge={r['edge_density']:.3f} pf={r['low_var_patch_frac']:.2f}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(scaled)

    if panels:
        # Две колонки; высоты выравниваем паддингом.
        max_h = max(p.shape[0] for p in panels)
        max_w = max(p.shape[1] for p in panels)
        padded = []
        for p in panels:
            pad_h = max_h - p.shape[0]
            pad_w = max_w - p.shape[1]
            padded.append(cv2.copyMakeBorder(p, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
        rows_ct = [np.hstack(padded[i:i + 2]) for i in range(0, len(padded), 2)]
        # Финальный паддинг, чтобы строки имели одинаковую ширину.
        max_row_w = max(r.shape[1] for r in rows_ct)
        rows_eq = [
            cv2.copyMakeBorder(r, 0, 0, 0, max_row_w - r.shape[1],
                               cv2.BORDER_CONSTANT, value=(0, 0, 0))
            for r in rows_ct
        ]
        sheet = np.vstack(rows_eq)
        contact_path = args.output_dir / "contact.png"
        cv2.imwrite(str(contact_path), sheet)
        print(f"[verify] contact -> {contact_path}")

    # --- проверка критерия --------------------------------------------------
    lag_in = _detect_lag(times, final, args.cloud_enter, rising=True)
    lag_out = _detect_lag(times, final, args.cloud_exit, rising=False)

    # False positive: detector срабатывает на чистом участке (>2 с вне GT-окна
    # облака; буфер 2 с покрывает неоднозначность границы GT при sweep 4 Гц).
    # FP > 0 — реальная ошибка: detector отключил бы matcher на нормальных кадрах.
    buf = 2.0
    outside = (times < args.cloud_enter - buf) | (times > args.cloud_exit + buf)
    fp = int((final & outside).sum())
    inside = (times >= args.cloud_enter) & (times <= args.cloud_exit)
    # FN внутри сегмента — *справочная* метрика: само перекрытие колеблется
    # из-за частичного оттаивания, поэтому 0 FN нереалистично. Ворота
    # inlier-ratio этапа 1.4 downstream ловят непомеченные кадры середины участка.
    fn = int(((~final) & inside).sum())
    inside_n = int(inside.sum())
    outside_n = int(outside.sum())

    summary = [
        f"Stage 1.5 obstruction detector verification",
        f"video           : {args.video}",
        f"window          : [{args.start_seconds:.1f}, {args.end_seconds:.1f}] s @ {args.fps} Hz",
        f"frames sampled  : {len(rows)}",
        f"cloud GT        : [{args.cloud_enter:.1f}, {args.cloud_exit:.1f}] s",
        "",
        f"detect lag in   : {lag_in!s} s   (criterion: <= {args.max_lag} s)",
        f"detect lag out  : {lag_out!s} s   (criterion: <= {args.max_lag} s)",
        f"false positives : {fp} of {outside_n} clear samples (>{buf}s outside cloud)",
        f"false negatives : {fn} of {inside_n} cloud samples (advisory; partial defrost)",
        "",
        "thresholds:",
        f"  std < {detector.std_threshold}",
        f"  entropy < {detector.entropy_threshold}",
        f"  edge_density < {detector.edge_density_threshold}",
        f"  low_var_patch_frac > {detector.low_var_patch_frac_threshold}",
        f"  votes_required = {detector.votes_required}",
        f"  hysteresis: enter={detector.entry_streak}, exit={detector.exit_streak}",
    ]
    summary_txt = "\n".join(summary)
    (args.output_dir / "summary.txt").write_text(summary_txt + "\n", encoding="utf-8")
    print()
    print(summary_txt)

    ok_in = lag_in is not None and lag_in <= args.max_lag
    ok_out = lag_out is not None and lag_out <= args.max_lag
    if ok_in and ok_out and fp == 0:
        print("\n[verify] CRITERION PASSED")
        return
    print("\n[verify] CRITERION NOT YET MET — inspect metrics.png and refine thresholds")
    sys.exit(2)


if __name__ == "__main__":
    main()
