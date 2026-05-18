"""Сквозной smoke-тест: проверить, что полный стек восприятия получает
фиксации на реальных крейсерских кадрах GP010269.MP4, когда retriever решает
задачу начального поиска.

Пайплайн на кадр:

    сырой кадр
       ├─> retriever (DINOv2-B) -> top-1 тайл z=14 -> грубо (lat, lon)
       │
       └─> выпрямление -> маска самолёта -> BEV(pitch=-30) -> matcher
                                                              vs
                                                окно MapManager z=17
                                                вокруг фиксации retriever
                                                              │
                                                              v
                                                  compute_map_measurement

Последовательные кадры здесь ещё НЕ связываются: нет временного доверия и нет
EKF. Это следующий шаг после доказательства, что сопоставление вообще возможно.
Вопрос максимально простой: **принимает ли матчер хоть один кадр, когда нужный
тайл уже попал в окно?** Если да, разблокированы следующие вехи: реальная
калибровка §3.2, EKF на реальном видео, измерение дрейфа траектории. Если нет,
сначала нужен §3.4 (XFeat/MatchAnything).

Метрики на сэмпл:
  * cosine-score top-1 у retriever, lat/lon и tile_id
  * matcher: raw_matches, num_inliers, inlier_ratio, reproj_px, accepted
  * остаток между фиксацией retriever и фиксацией matcher, км
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
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

from src.aircraft_mask import (
    AircraftMaskTracker,
    load_aircraft_mask_tracker_for_video,
)
from src.bev_rectifier import BevRectifier
from src.geo_segmentor import GeoSegmentor
from src.map_manager import MapManager
from src.map_measurement import compute_map_measurement
from src.neural_matching import NeuralMatcher
from src.online_occlusion import OnlineAircraftOcclusionMasker, SamAircraftSegmenter
from src.retriever import Retriever
from src.snow_mask import combine_with_aircraft_mask, detect_snow_mask
from src.undistort import Undistorter
from src.video_processor import VideoProcessor


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6_371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, default=Path("data/videos/GP010269.MP4"))
    p.add_argument("--camera-config", type=Path, default=Path("configs/camera_gopro_hx.yaml"))
    p.add_argument("--anchors-dir", type=Path, default=Path("data/masks/anchors"))
    p.add_argument(
        "--aircraft-mask-mode",
        choices=("off", "anchors", "online-sam", "hybrid"),
        default="anchors",
        help="off: без маски самолёта; anchors: только совместимый профиль "
        "камеры/носителя; online-sam: периодический SAM-refresh + optical flow; "
        "hybrid: SAM-refresh и совместимые anchors как резерв",
    )
    p.add_argument(
        "--allow-aircraft-mask-video-mismatch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="только для отладки: разрешить anchors, чей index.json указывает на другое видео",
    )
    p.add_argument("--sam-refresh-s", type=float, default=2.0)
    p.add_argument("--sam-model-id", type=str, default="facebook/sam3")
    p.add_argument(
        "--sam-prompts",
        type=str,
        default="aircraft|airplane wing|aircraft fuselage|landing gear|wing strut|cockpit frame",
    )
    p.add_argument("--sam-device", type=str, default="auto", choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--sam-score-threshold", type=float, default=0.20)
    p.add_argument("--sam-mask-threshold", type=float, default=0.50)
    p.add_argument("--sam-dilation-px", type=int, default=18)
    p.add_argument("--sam-close-radius-px", type=int, default=5)
    p.add_argument("--sam-min-coverage", type=float, default=0.002)
    p.add_argument("--sam-max-coverage", type=float, default=0.45)
    p.add_argument("--sam-left-only-frac", type=float, default=1.0)
    p.add_argument("--sam-max-box-right-frac", type=float, default=1.0)
    p.add_argument("--retriever-db", type=Path, default=Path("data/retrieval/db_z14"))
    p.add_argument("--output", type=Path, default=Path("results/end_to_end"))
    p.add_argument("--cruise-start-s", type=float, default=45.0)
    p.add_argument("--cruise-end-s", type=float, default=120.0)
    p.add_argument("--sample-period-s", type=float, default=2.0)
    p.add_argument("--zoom", type=int, default=17)
    p.add_argument("--window-radius", type=int, default=5,
                   help="радиус окна z=17 вокруг фиксации retriever, в тайлах")
    p.add_argument("--pitch-deg", type=float, default=-30.0)
    p.add_argument("--agl-m", type=float, default=620.0)
    p.add_argument("--ground-span-m", type=float, default=400.0)
    p.add_argument("--bev-out-size", type=int, default=800)
    p.add_argument("--backend", type=str, default="auto")
    p.add_argument("--snow-mask", action=argparse.BooleanOptionalAction, default=False,
                   help="этап 4-lite: маскировать снег на BEV перед matcher")
    p.add_argument("--snow-v-threshold", type=float, default=0.80)
    p.add_argument("--snow-s-threshold", type=float, default=0.15)
    p.add_argument("--retriever-top-k", type=int, default=1,
                   help="если >1, пробуем top-K тайлов retriever на кадр и "
                        "берём лучший результат matcher")
    p.add_argument("--apply-clahe", action=argparse.BooleanOptionalAction, default=False,
                   help="этап 3.6: CLAHE на grayscale перед matcher. На "
                        "межсезонном GP010269 ухудшило результат, поэтому "
                        "по умолчанию выключено.")
    p.add_argument("--semantic-mask", action=argparse.BooleanOptionalAction, default=False,
                   help="этап 4: фильтровать совпадения по семантическому классу")
    p.add_argument("--semantic-structural-match", action=argparse.BooleanOptionalAction, default=False,
                   help="этап 4b: независимый mask-to-mask NCC канал; "
                        "SegFormer обрабатывает drone и sat, NCC идёт по классам.")
    p.add_argument("--seg-config", type=Path,
                   default=Path("results/segformer_overture_b0_phase_c_osm_manualcw/segformer_overture_quick_cfg.py"))
    p.add_argument("--seg-checkpoint", type=Path,
                   default=Path("results/segformer_overture_b0_phase_c_osm_manualcw/best_mIoU_iter_1165.pth"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if not args.video.exists():
        raise SystemExit(f"video not found: {args.video}")

    print(f"[run] video         : {args.video}")
    print(f"[run] retriever DB  : {args.retriever_db}")
    print(f"[run] cruise window : t=[{args.cruise_start_s}, {args.cruise_end_s}] s "
          f"sample={args.sample_period_s}s")
    print()

    # Компоненты.
    retr = Retriever()
    retr.load_database(args.retriever_db)
    print(f"[run] retriever device : {retr.device}  N={len(retr.database)}")
    undistorter = Undistorter.from_yaml(args.camera_config)
    print(f"[run] K_rect[0,0]      : {undistorter.K_rect[0,0]:.2f}")
    proc = VideoProcessor(str(args.video),
                          default_geo=(55.086025, 38.149033, 750.0))
    src_fps = proc.info.fps
    platform_mask_tracker: AircraftMaskTracker | None = None
    if args.aircraft_mask_mode in ("anchors", "hybrid"):
        platform_mask_tracker = load_aircraft_mask_tracker_for_video(
            args.anchors_dir,
            args.video,
            allow_video_mismatch=args.allow_aircraft_mask_video_mismatch,
        )
    if args.aircraft_mask_mode == "off":
        mask_tracker = None
    elif args.aircraft_mask_mode in ("online-sam", "hybrid"):
        sam_prompts = [s.strip() for s in args.sam_prompts.split("|") if s.strip()]
        sam_segmenter = SamAircraftSegmenter(
            model_id=args.sam_model_id,
            prompts=sam_prompts,
            device=args.sam_device,
            score_threshold=args.sam_score_threshold,
            mask_threshold=args.sam_mask_threshold,
            dilation_px=args.sam_dilation_px,
            close_radius_px=args.sam_close_radius_px,
            min_coverage=args.sam_min_coverage,
            max_coverage=args.sam_max_coverage,
            left_only_frac=args.sam_left_only_frac,
            max_box_right_frac=args.sam_max_box_right_frac,
        )
        mask_tracker = OnlineAircraftOcclusionMasker(
            sam_segmenter=sam_segmenter,
            platform_tracker=platform_mask_tracker,
            sam_refresh_frames=max(1, int(round(args.sam_refresh_s * src_fps))),
        )
    else:
        mask_tracker = platform_mask_tracker
    print(f"[run] aircraft mask    : {args.aircraft_mask_mode}")
    if mask_tracker:
        print(f"[run] mask anchors     : {mask_tracker.num_anchors()}")
    bev = BevRectifier.build(
        K_rect=undistorter.K_rect, image_size=undistorter.image_size,
        pitch_deg=args.pitch_deg, agl_m=args.agl_m,
        ground_span_m=args.ground_span_m,
        out_size=(args.bev_out_size, args.bev_out_size),
    )
    print(f"[run] BEV pitch        : {args.pitch_deg}°")

    matcher = NeuralMatcher(backend=args.backend, apply_clahe=args.apply_clahe)

    # Семантическая маска этапа 4: SegFormer-B0, дообученный на Overture-RU.
    # Стабильные ID вручную ограничены до {3,4} (здания, дороги): растительность есть
    # в `stable` схемы, но неустойчива при разрыве зима-лето; вода всё равно
    # почти не даёт ключевых точек.
    seg = None
    need_seg = args.semantic_mask or args.semantic_structural_match
    if need_seg:
        import torch
        seg_device = "mps" if torch.backends.mps.is_available() else "cpu"
        # SegFormer-B0, дообученный на Overture-RU, надёжен на спутниковых
        # снимках: это его обучающий домен. На зимних BEV-warped кадрах с дрона
        # он сильно страдает от domain shift и предсказывает ~97 % background,
        # поэтому на стороне дрона используется эвристический текстурный
        # сегментер, а на спутниковой стороне — ML. Фильтр семантической маски
        # оставляет только совпадения, где точка дрона лежит в текстурной
        # наземной области, А спутниковая точка — на стабильном классе.
        seg = GeoSegmentor(
            backend="mmseg",
            mmseg_config=str(args.seg_config),
            mmseg_checkpoint=str(args.seg_checkpoint),
            mmseg_device=seg_device,
            # Включаем water (1): на сельхоз/лесных тайлах модель иногда
            # путает тёмный лес с водой, но это всё равно сезонно стабильные
            # поверхности. С water остаётся около 50 % допустимых пикселей.
            # bg (0) исключаем: это класс "нет объекта", такие совпадения надо отбрасывать.
            stable_class_ids={1, 2, 3, 4},
            top_crop_ratio=0.0,
        )
        seg_drone = GeoSegmentor(backend="heuristic", top_crop_ratio=0.0)
        if seg.backend != "mmseg":
            print(f"[run] ПРЕДУПРЕЖДЕНИЕ: запрошена semantic mask, но mmseg "
                  f"не инициализировался; запускаемся без class filter")
            seg = None
        else:
            print(f"[run] semantic mask  : ML(sat) + heuristic(drone)  "
                  f"sat stable={1, 2, 3, 4}")
    # Кэш MapManager по retriever tile_id (z=14).
    mm_cache: dict[str, tuple] = {}

    def _ensure_window(tile_id: str, lat: float, lon: float):
        if tile_id in mm_cache:
            return mm_cache[tile_id]
        mm = MapManager(zoom=args.zoom, window_radius=args.window_radius,
                        closer_threshold=1)
        map_cv2, bbox = mm.initialize(lat, lon)
        mm_cache[tile_id] = (map_cv2, bbox)
        return mm_cache[tile_id]

    sample_frames = []
    t = args.cruise_start_s
    while t <= args.cruise_end_s:
        sample_frames.append(int(round(t * src_fps)))
        t += args.sample_period_s
    print(f"[run] frames to test   : {len(sample_frames)}")
    print()

    rows: list[dict] = []
    accepted_count = 0
    last_inliers_log: list[int] = []
    t_start = time.time()
    for idx, fi in enumerate(sample_frames):
        frame = proc.extract_frame(fi)
        if frame is None:
            continue

        # Инициализация через retriever: top-K с объединением, если задано.
        tops = retr.query_image(frame, top_k=args.retriever_top_k)

        # Покадровый one-shot пайплайн, общий для всех кандидатов retriever:
        # выпрямление кадра -> маска -> BEV.
        rect = undistorter.undistort_image(frame)
        if mask_tracker is not None:
            mask_raw = mask_tracker.mask_for_frame(fi, frame.shape[:2], frame=frame)
            mask_rect = undistorter.undistort_image(mask_raw)
            mask_bev = bev.warp_mask(mask_rect)
        else:
            mask_bev = None
        frame_bev = bev.warp(rect)
        if args.snow_mask:
            snow = detect_snow_mask(
                frame_bev,
                v_threshold=args.snow_v_threshold,
                s_threshold=args.snow_s_threshold,
            )
            mask_bev = combine_with_aircraft_mask(snow, mask_bev)

        # Финальная форма этапа 4: ML-сегментация только на спутниковой стороне.
        # Зимний drone BEV обманывает эвристический SkySegmenter: снег
        # принимается за небо, текстура становится 0 %. ML-сегментер тоже
        # страдает от domain shift и даёт 97 % background. Фильтрация только
        # спутниковой стороны вместе с уже существующими масками самолёта и
        # снега даёт ту же логику защиты без ложных нулей со стороны дрона.
        drone_cmap = None  # явно выключено

        # Пробуем каждого top-K кандидата; оставляем результат с максимальным
        # числом inlier-точек. При равенстве предпочтение выше score retriever
        # (top-1 идёт первым).
        best_meas = None
        best_top = tops[0]
        best_match_ms = 0.0
        best_raw = 0
        for top in tops:
            map_cv2, bbox = _ensure_window(top["tile_id"], top["lat"], top["lon"])
            mt0 = time.time()
            r = matcher.match(frame_bev, map_cv2, aircraft_mask=mask_bev)
            m_ms = (time.time() - mt0) * 1000.0
            mkpts0_arr = r["mkpts0"]; mkpts1_arr = r["mkpts1"]

            # Семантический фильтр этапа 4, только спутниковая сторона.
            sat_cmap = None
            if seg is not None and args.semantic_mask and len(mkpts0_arr) > 0:
                sat_cmap = seg.stable_class_map(map_cv2)
                s_xy = mkpts1_arr.astype(np.int32)
                s_xy[:, 0] = np.clip(s_xy[:, 0], 0, sat_cmap.shape[1] - 1)
                s_xy[:, 1] = np.clip(s_xy[:, 1], 0, sat_cmap.shape[0] - 1)
                sat_keep = sat_cmap[s_xy[:, 1], s_xy[:, 0]] > 0
                mkpts0_arr = mkpts0_arr[sat_keep]
                mkpts1_arr = mkpts1_arr[sat_keep]

            n_kept = len(mkpts0_arr)
            n_raw = int(r.get("raw_matches", n_kept))
            m = compute_map_measurement(
                frame_shape=frame_bev.shape,
                mkpts_drone=mkpts0_arr, mkpts_map=mkpts1_arr,
                bbox=bbox, map_shape=map_cv2.shape,
                last_homography_scale=None,
            )
            if best_meas is None or m.num_inliers > best_meas.num_inliers:
                best_meas = m; best_top = top
                best_match_ms = m_ms; best_raw = n_raw
        meas = best_meas
        ret_lat = best_top["lat"]; ret_lon = best_top["lon"]
        ret_score = best_top["score"]; tile_id = best_top["tile_id"]
        match_ms = best_match_ms; raw_matches = best_raw
        # bbox/map_cv2 победившего кандидата нужны только для логирования.
        map_cv2, bbox = _ensure_window(tile_id, ret_lat, ret_lon)
        mkpts0 = np.empty((0, 2)); mkpts1 = np.empty((0, 2))  # дальше не используются

        residual_km = float("nan")
        if meas.accepted:
            accepted_count += 1
            residual_km = _haversine_km(ret_lat, ret_lon, meas.lat, meas.lon)
            last_inliers_log.append(meas.num_inliers)

        # --- Structural-канал: mask-to-mask NCC с SegFormer на обеих сторонах ---
        struct_accepted = False
        struct_peak = float("nan")
        struct_margin = float("nan")
        struct_lat = float("nan")
        struct_lon = float("nan")
        if args.semantic_structural_match and seg is not None:
            from src.semantic_mask_matcher import mask_to_mask_match
            # SegFormer на лучшем кандидате (map_cv2 уже доступен — это победитель).
            drone_class_map = seg.stable_class_map(frame_bev)
            if sat_cmap is None:
                sat_cmap = seg.stable_class_map(map_cv2)
            # Оценка m/px: frame_bev = ground_span_m / bev_out_size.
            drone_mppx = args.ground_span_m / frame_bev.shape[1]
            # Спутниковое окно покрывает тайлы window_radius на заданном zoom.
            # Упрощённо используем тот же m/px, что и у drone BEV.
            sat_mppx = drone_mppx
            sfix = mask_to_mask_match(
                drone_class_map=drone_class_map,
                sat_class_map=sat_cmap,
                drone_mppx=drone_mppx,
                sat_mppx=sat_mppx,
                sat_centre_latlon=(ret_lat, ret_lon),
            )
            struct_peak = sfix.peak_score
            struct_margin = sfix.peak_margin
            if sfix.accepted:
                struct_accepted = True
                struct_lat = sfix.lat
                struct_lon = sfix.lon

        # Кадр считается принятым, если хотя бы один канал дал валидное измерение.
        frame_any_accepted = meas.accepted or struct_accepted
        rows.append({
            "frame_idx": fi,
            "t_sec": fi / src_fps,
            "ret_score": ret_score,
            "ret_lat": ret_lat,
            "ret_lon": ret_lon,
            "ret_tile": tile_id,
            "raw_matches": raw_matches,
            "num_inliers": meas.num_inliers,
            "inlier_ratio": meas.inlier_ratio,
            "reproj_px": meas.reprojection_error_px,
            "sigma_xy_m": meas.sigma_xy_m,
            "accepted": int(meas.accepted),
            "reject_reason": meas.reject_reason,
            "match_ms": round(match_ms, 1),
            "fix_lat": meas.lat,
            "fix_lon": meas.lon,
            "residual_ret_km": residual_km,
            "struct_accepted": int(struct_accepted),
            "struct_peak": struct_peak,
            "struct_margin": struct_margin,
            "struct_lat": struct_lat,
            "struct_lon": struct_lon,
            "any_accepted": int(frame_any_accepted),
        })

        marker = "*" if frame_any_accepted else " "
        struct_str = (f" str_acc=Y peak={struct_peak:.2f}" if struct_accepted
                      else f" str_acc=N peak={struct_peak:.2f}" if args.semantic_structural_match
                      else "")
        print(f"  {marker} f={fi:>6}  t={fi/src_fps:5.1f}s  retr={ret_score:.3f} "
              f"({ret_lat:.4f},{ret_lon:.4f})  raw={raw_matches:<3} "
              f"inl={meas.num_inliers:<3} acc={int(meas.accepted)}{struct_str}")

    elapsed = time.time() - t_start
    print()
    print(f"[run] total runtime   : {elapsed:.1f}s")
    print(f"[run] mm windows used : {len(mm_cache)} (cached by retriever tile_id)")
    print(f"[run] frames tested   : {len(rows)}")
    n_struct_acc = sum(int(r.get("struct_accepted", 0)) for r in rows)
    n_any_acc = sum(int(r.get("any_accepted", 0) or r.get("accepted", 0)) for r in rows)
    print(f"[run] frames accepted (appearance) : {accepted_count}  ({100*accepted_count/max(1,len(rows)):.1f}%)")
    if args.semantic_structural_match:
        print(f"[run] frames accepted (structural) : {n_struct_acc}  ({100*n_struct_acc/max(1,len(rows)):.1f}%)")
        print(f"[run] frames accepted (any channel): {n_any_acc}  ({100*n_any_acc/max(1,len(rows)):.1f}%)")
    if last_inliers_log:
        arr = np.array(last_inliers_log)
        print(f"[run] inliers (acc'd) : median {float(np.median(arr)):.0f}  "
              f"max {int(arr.max())}")

    # Гистограмма причин отбраковки.
    rej_counts: dict[str, int] = {}
    for r in rows:
        if not r["accepted"]:
            rej_counts[r["reject_reason"]] = rej_counts.get(r["reject_reason"], 0) + 1
    if rej_counts:
        print("[run] rejects:")
        for k, v in sorted(rej_counts.items(), key=lambda kv: -kv[1]):
            print(f"    {k:30s} {v}")

    # CSV и короткая сводка.
    csv_path = args.output / "smoke.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[run] CSV -> {csv_path}")

    summary = [
        f"frames tested     : {len(rows)}",
        f"frames accepted   : {accepted_count}",
        f"acceptance rate   : {100*accepted_count/max(1,len(rows)):.1f}%",
        f"map windows used  : {len(mm_cache)}",
        f"runtime           : {elapsed:.1f}s",
    ]
    (args.output / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
