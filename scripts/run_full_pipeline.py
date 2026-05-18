"""Интеграционный сквозной запуск: EKF, оптический поток, поиск тайлов и сопоставление.

Собирает компоненты этапов 1-3 в единый покадровый цикл, который строит
настоящую *траекторию* по реальному видео GP010269.MP4. Это демонстрация
уже не отдельных покадровых фиксаций, а всей связки целиком.

На каждом кадре:

    1. Считается dt с прошлого кадра.
    2. ``predict``: EKF протягивает состояние на dt.
    3. Оптико-потоковая одометрия Лукаса-Канаде между текущим и прошлым кадром
       даёт скорость в связанной системе координат и угловую скорость курса.
       Это подаётся в EKF как измерение скорости с дисперсией, увеличенной
       по неопределённости поправки масштаба.
    4. **Приор высоты**: мягкая привязка z_u без барометра и телеметрии.
    5. **Приор курса**: включается только при ``|yaw_rate| < threshold``.

Каждые ``map_period`` кадров дополнительно запускается картографический канал:

    6. Поиск DINOv2-B возвращает центр лучшего тайла масштаба z=14.
    7. MapManager загружает окно масштаба z=17 с кэшем по идентификатору тайла.
    8. Выпрямление -> маска самолёта -> BEV при базовом pitch.
    9. Локальное сопоставление XFeat.
   10. ``compute_map_measurement`` формирует картографическое измерение.
   11. Если измерение принято, состояние обновляется по WGS84-точке.
   12. Опционально структурный сопоставитель работает как независимый канал
       на основе Overture для межсезонных сцен.

Выходы:
    results/full_pipeline/state.csv        покадровое состояние фильтра
    results/full_pipeline/trajectory.png   график траектории и принятых фиксаций
    results/full_pipeline/scale_history.png  сходимость поправки масштаба
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

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
from src.ekf import StateFilter
from src.frame_bridge import FrameBridge
from src.geo_segmentor import GeoSegmentor
from src.hud_drawer import HUDDrawer, HudState
from src.map_manager import MapManager
from src.map_measurement import compute_map_measurement
from src.neural_matching import NeuralMatcher
from src.obstruction_detector import ObstructionDetector
from src.online_occlusion import OnlineAircraftOcclusionMasker, SamAircraftSegmenter
from src.optical_flow_vo import OpticalFlowVO
from src.retriever import Retriever
from src.snow_mask import combine_with_aircraft_mask, detect_snow_mask
from src.track_state import ChannelSchedule, TrackMode, TrackState
from src.undistort import Undistorter
from src.video_processor import VideoProcessor


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, default=Path("data/videos/GP010269.MP4"))
    p.add_argument(
        "--camera-config", type=Path, default=Path("configs/camera_gopro_hx.yaml")
    )
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
        help="Текстовые prompt'ы для online-SAM через |",
    )
    p.add_argument("--sam-device", type=str, default="auto", choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--sam-score-threshold", type=float, default=0.20)
    p.add_argument("--sam-mask-threshold", type=float, default=0.50)
    p.add_argument("--sam-dilation-px", type=int, default=18)
    p.add_argument("--sam-close-radius-px", type=int, default=5)
    p.add_argument("--sam-min-coverage", type=float, default=0.002)
    p.add_argument("--sam-max-coverage", type=float, default=0.45)
    p.add_argument(
        "--sam-left-only-frac",
        type=float,
        default=1.0,
        help="Опциональный ROI-prior: принимать SAM boxes с центром левее этой доли кадра",
    )
    p.add_argument(
        "--sam-max-box-right-frac",
        type=float,
        default=1.0,
        help="Опциональный ROI-prior: отбрасывать SAM boxes, чей правый край дальше этой доли кадра",
    )
    p.add_argument("--retriever-db", type=Path, default=Path("data/retrieval/db_z14"))
    p.add_argument("--output", type=Path, default=Path("results/full_pipeline"))
    p.add_argument("--start-s", type=float, default=45.0)
    p.add_argument("--end-s", type=float, default=90.0)
    p.add_argument("--start-lat", type=float, default=55.086025)
    p.add_argument("--start-lon", type=float, default=38.149033)
    p.add_argument("--start-alt-msl", type=float, default=750.0)
    p.add_argument(
        "--ground-msl",
        type=float,
        default=130.0,
        help="примерная высота земли MSL в точке старта",
    )
    p.add_argument(
        "--dem",
        type=Path,
        default=Path("data/dem/srtm_n55_e038.tif"),
        help="DEM GeoTIFF для покадровой AGL; если недоступен, используется --agl-m",
    )
    p.add_argument("--map-period-s", type=float, default=2.0)
    p.add_argument("--zoom", type=int, default=17)
    p.add_argument("--window-radius", type=int, default=2)
    p.add_argument("--pitch-deg", type=float, default=-30.0)
    p.add_argument("--agl-m", type=float, default=620.0)
    p.add_argument("--ground-span-m", type=float, default=400.0)
    p.add_argument("--bev-out-size", type=int, default=800)
    p.add_argument("--backend", type=str, default="xfeat")
    p.add_argument(
        "--apply-clahe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="этап 3.6: CLAHE-препроцессинг в matcher. По умолчанию выключен "
        "(на межсезонном GP010269 дал -3 принятых кадра).",
    )
    p.add_argument(
        "--obstruction-detect", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument(
        "--snow-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="маскировать яркий малонасыщенный снег в drone BEV перед matching",
    )
    p.add_argument("--snow-v-threshold", type=float, default=0.80)
    p.add_argument("--snow-s-threshold", type=float, default=0.15)
    p.add_argument(
        "--semantic-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="фильтровать совпадения по стабильным семантическим классам на спутнике",
    )
    p.add_argument(
        "--semantic-structural-match",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="использовать class map от SegFormer внутри StructuralMatcher "
        "для class-wise mask-to-mask сопоставления геометрии",
    )
    p.add_argument(
        "--seg-config",
        type=Path,
        default=Path(
            "results/segformer_overture_b0_ade1400_nocw_best/segformer_overture_quick_cfg.py"
        ),
    )
    p.add_argument(
        "--seg-checkpoint",
        type=Path,
        default=Path(
            "results/segformer_overture_b0_ade1400_nocw_best/best_mIoU_iter_932.pth"
        ),
    )
    p.add_argument(
        "--vo-stride",
        type=int,
        default=2,
        help="запускать OF VO раз в N кадров видео, чтобы снизить общее время",
    )
    p.add_argument(
        "--vo-downsample-width",
        type=int,
        default=960,
        help="VO LK работает на пониженном разрешении: ~3× speedup "
        "при сохранённой точности; 0 → отключить",
    )
    p.add_argument(
        "--log-stride",
        type=int,
        default=2,
        help="писать состояние фильтра в CSV раз в N кадров",
    )
    p.add_argument(
        "--bootstrap-buffer",
        type=int,
        default=5,
        help="накопить N принятых измерений перед инициализацией EKF "
        "(защита от межсезонных ложных захватов)",
    )
    p.add_argument(
        "--bootstrap-cluster-radius-m",
        type=float,
        default=2000.0,
        help="измерения внутри этого радиуса образуют кластер; центроид "
        "крупнейшего кластера используется как начальная позиция EKF",
    )
    p.add_argument(
        "--structural-match",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="включить Overture structural matcher как независимый канал EKF",
    )
    p.add_argument(
        "--structural-vectors-root",
        type=Path,
        default=Path("data/overture_ru_dataset_starter/vectors"),
    )
    p.add_argument("--structural-region-id", type=str, default="kolomna_cruise")
    p.add_argument("--structural-period-s", type=float, default=2.0)
    p.add_argument("--structural-search-radius-m", type=float, default=600.0)
    # Адаптивное расписание: TRACK / WEAK / RELOCALIZE / BOOTSTRAP.
    p.add_argument(
        "--adaptive-schedule",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="адаптивное расписание каналов по σ_pos и gap "
        "(realtime-бюджет; --no-adaptive-schedule = старые фиксированные периоды)",
    )
    p.add_argument(
        "--track-sigma-m",
        type=float,
        default=60.0,
        help="σ_pos < этого порога → TRACK режим",
    )
    p.add_argument(
        "--weak-sigma-m",
        type=float,
        default=200.0,
        help="σ_pos < этого порога → как минимум WEAK",
    )
    p.add_argument(
        "--track-timeout-s",
        type=float,
        default=5.0,
        help="если gap с последнего fix меньше порога, остаёмся в TRACK",
    )
    p.add_argument(
        "--weak-timeout-s",
        type=float,
        default=30.0,
        help="если gap с последнего fix меньше порога, режим не хуже WEAK",
    )
    p.add_argument("--track-app-period-s", type=float, default=4.0)
    p.add_argument("--weak-app-period-s", type=float, default=1.0)
    p.add_argument("--reloc-app-period-s", type=float, default=0.5)
    p.add_argument(
        "--parallel-channels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="appearance-канал (GPU: retriever+XFeat) и structural-канал (CPU: NCC) "
        "запускаются в параллельных потоках; обе библиотеки релизят GIL",
    )
    p.add_argument(
        "--retriever-top-k",
        type=int,
        default=3,
        help="число кандидатных тайлов от retriever; matcher пробует "
        "каждого, выбирается лучший по inliers (план §3.5)",
    )
    p.add_argument(
        "--retriever-early-accept-inliers",
        type=int,
        default=30,
        help="если у топ-1 кандидата уже ≥ N inliers, пропускаем "
        "остальные — экономия XFeat-вызовов в TRACK/WEAK",
    )
    p.add_argument(
        "--hud-video",
        type=Path,
        default=None,
        help="путь .mp4 для HUD-видео; кадр + статус панель + "
        "мини-карта траектории. Если не указан — HUD не пишется.",
    )
    p.add_argument(
        "--hud-fps",
        type=float,
        default=0.0,
        help="FPS HUD-видео; 0 → используется FPS источника",
    )
    p.add_argument(
        "--hud-history-s",
        type=float,
        default=60.0,
        help="окно истории траектории на мини-карте, сек",
    )
    args = p.parse_args()
    if args.semantic_structural_match and not args.structural_match:
        args.structural_match = True
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"[run] video         : {args.video}")
    print(f"[run] window        : t=[{args.start_s}, {args.end_s}] s")
    print(f"[run] map period    : every {args.map_period_s} s")
    print()

    # ---- поднимаем компоненты ---------------------------------------------
    proc = VideoProcessor(
        str(args.video),
        default_geo=(args.start_lat, args.start_lon, args.start_alt_msl),
    )
    src_fps = proc.info.fps

    bridge = FrameBridge(
        lat0=args.start_lat,
        lon0=args.start_lon,
        alt0_msl=args.ground_msl,
    )
    filt = StateFilter(bridge)
    filt.initialize_from_wgs84(
        args.start_lat,
        args.start_lon,
        args.start_alt_msl,
        yaw=0.0,
        sigma_pos_m=5_000.0,
        sigma_yaw_rad=math.pi,
    )
    # Начальную скорость не зажимаем в ноль. Стартуем как «неизвестно»
    # (σ=50 м/с), чтобы первое OF-измерение имело нормальный вес.
    filt.P[StateFilter.IDX_VX_E, StateFilter.IDX_VX_E] = 50.0**2
    filt.P[StateFilter.IDX_VY_N, StateFilter.IDX_VY_N] = 50.0**2

    undistorter = Undistorter.from_yaml(args.camera_config)
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
    bev = BevRectifier.build(
        K_rect=undistorter.K_rect,
        image_size=undistorter.image_size,
        pitch_deg=args.pitch_deg,
        agl_m=args.agl_m,
        ground_span_m=args.ground_span_m,
        out_size=(args.bev_out_size, args.bev_out_size),
    )
    matcher = NeuralMatcher(backend=args.backend, apply_clahe=args.apply_clahe)
    retr = Retriever()
    retr.load_database(args.retriever_db)
    obstruction_detector = ObstructionDetector() if args.obstruction_detect else None
    dem_lookup = None
    dem_error = None
    if args.dem is not None and args.dem.exists():
        try:
            from src.dem_lookup import DemLookup

            dem_lookup = DemLookup(args.dem)
        except Exception as e:
            dem_error = e
    seg = None
    if args.semantic_mask or args.semantic_structural_match:
        import torch

        seg_device = "mps" if torch.backends.mps.is_available() else "cpu"
        seg = GeoSegmentor(
            backend="mmseg",
            mmseg_config=str(args.seg_config),
            mmseg_checkpoint=str(args.seg_checkpoint),
            mmseg_device=seg_device,
            stable_class_ids={1, 2, 3, 4},
            top_crop_ratio=0.0,
        )
        if seg.backend != "mmseg":
            seg = None
    structural = None
    if args.structural_match:
        from src.structural_matcher import StructuralMatcher

        structural = StructuralMatcher(
            vectors_root=args.structural_vectors_root,
            region_id=args.structural_region_id,
            search_radius_m=args.structural_search_radius_m,
        )

    vo = OpticalFlowVO(
        K=undistorter.K,
        D=undistorter.D,
        image_size=undistorter.image_size,
        pitch_deg=args.pitch_deg,
        downsample_to_width=args.vo_downsample_width,
    )

    print(f"[run] EKF initial pos σ : {filt.position_sigma_m():.0f} m")
    print(f"[run] retriever device  : {retr.device}")
    print(f"[run] matcher backend   : {matcher.backend}")
    print(
        f"[run] obstruction gate  : {'on' if obstruction_detector is not None else 'off'}"
    )
    print(f"[run] aircraft mask    : {args.aircraft_mask_mode}")
    if mask_tracker is not None:
        print(f"[run] mask anchors     : {mask_tracker.num_anchors()}")
    if args.aircraft_mask_mode in ("online-sam", "hybrid"):
        print(
            f"[run] SAM refresh      : every {args.sam_refresh_s:.1f}s "
            f"prompts={len([s for s in args.sam_prompts.split('|') if s.strip()])}"
        )
    print(f"[run] snow mask         : {'on' if args.snow_mask else 'off'}")
    print(
        f"[run] semantic mask     : {'on' if seg is not None and args.semantic_mask else 'off'}"
    )
    print(f"[run] structural match : {'on' if structural is not None else 'off'}")
    print(
        f"[run] semantic geometry: {'on' if structural is not None and seg is not None and args.semantic_structural_match else 'off'}"
    )
    if args.adaptive_schedule:
        print(
            f"[run] adaptive schedule: TRACK={args.track_app_period_s:.1f}s "
            f"WEAK={args.weak_app_period_s:.1f}s RELOC={args.reloc_app_period_s:.1f}s"
        )
    else:
        print(
            f"[run] schedule         : fixed map={args.map_period_s:.1f}s "
            f"struct={args.structural_period_s:.1f}s"
        )
    print(
        f"[run] DEM AGL source    : {args.dem if dem_lookup is not None else 'constant --agl-m'}"
    )
    if dem_error is not None:
        print(f"[run] DEM disabled      : {dem_error}")
    print()

    # ---- запись HUD-видео (опционально) -----------------------------------
    hud_writer = None
    if args.hud_video is not None:
        # Писатель видео создаём лениво при первом кадре, когда станет известен размер.
        args.hud_video.parent.mkdir(parents=True, exist_ok=True)
    hud_history_deque = []  # [(t_sec, x_e, y_n)], очищается по окну

    # ---- покадровый цикл --------------------------------------------------
    f0 = int(round(args.start_s * src_fps))
    f1 = int(round(args.end_s * src_fps))
    map_stride = max(1, int(round(args.map_period_s * src_fps)))

    # Автомат адаптивного расписания. Если --no-adaptive-schedule, режимам
    # выставляются одинаковые периоды (старый режим: --map-period-s / --structural-period-s),
    # и поведение совпадает с фиксированным расписанием.
    track_state = TrackState(
        sigma_track_m=args.track_sigma_m,
        sigma_weak_m=args.weak_sigma_m,
        timeout_track_s=args.track_timeout_s,
        timeout_weak_s=args.weak_timeout_s,
    )
    if args.adaptive_schedule:
        track_state.schedules = {
            TrackMode.BOOTSTRAP: ChannelSchedule(
                appearance_period_s=args.weak_app_period_s,
                structural_period_s=args.structural_period_s,
            ),
            TrackMode.TRACK: ChannelSchedule(
                appearance_period_s=args.track_app_period_s,
                structural_period_s=2.0 * args.track_app_period_s,
            ),
            TrackMode.WEAK: ChannelSchedule(
                appearance_period_s=args.weak_app_period_s,
                structural_period_s=2.0 * args.weak_app_period_s,
            ),
            TrackMode.RELOCALIZE: ChannelSchedule(
                appearance_period_s=args.reloc_app_period_s,
                structural_period_s=2.0 * args.reloc_app_period_s,
            ),
        }
    else:
        # Старый режим: одинаковые периоды для всех режимов.
        same = ChannelSchedule(
            appearance_period_s=args.map_period_s,
            structural_period_s=args.structural_period_s,
        )
        track_state.schedules = {m: same for m in TrackMode}

    mm_cache: dict[str, tuple] = {}
    sat_sem_cache: dict[str, np.ndarray] = {}

    def _ensure_window(tile_id: str, lat: float, lon: float):
        if tile_id in mm_cache:
            return mm_cache[tile_id]
        mm = MapManager(
            zoom=args.zoom, window_radius=args.window_radius, closer_threshold=1
        )
        m_cv2, m_bbox = mm.initialize(lat, lon)
        mm_cache[tile_id] = (m_cv2, m_bbox)
        return mm_cache[tile_id]

    def _ekf_tile_id(lat: float, lon: float) -> str:
        """ID slippy-тайла при zoom=args.zoom для (lat, lon). Стандартная WMTS-формула.

        В TRACK retriever пропускается — позиция уже хорошо известна, и
        идти в DINOv2 за тайл-кандидатом избыточно. Тайл вычисляется из
        EKF predicted позиции; ID используется как ключ в mm_cache.
        """
        n = 2**args.zoom
        x = int((lon + 180.0) / 360.0 * n)
        lat_rad = math.radians(lat)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return f"ekf_{args.zoom}/{x}/{y}"

    def _coordinated_turn_bank_deg() -> float:
        """Оценка крена в координированном повороте (упрощение для самолёта).

        bank ≈ atan(v · yaw_rate / g) — стандартная формула. В крейсере
        v ~ 50 м/с и yaw_rate 5°/с даёт ~24° крена, что совпадает с
        наблюдаемой геометрией самолёта в развороте.
        Без отдельного roll-сенсора это лучшая аппроксимация. На прямой
        |yaw_rate| мал → bank ~ 0, BEV не пересчитывается.
        """
        if not bootstrap_done:
            return 0.0
        v = filt.speed()
        yaw_rate = float(filt.x[StateFilter.IDX_YAW_RATE])
        if v < 5.0 or abs(yaw_rate) < math.radians(0.5):
            return 0.0
        return math.degrees(math.atan(v * yaw_rate / 9.80665))

    def _build_bev_frame(frame: np.ndarray, agl_m: float):
        rect = undistorter.undistort_image(frame)
        bev_frame = bev
        roll_deg = _coordinated_turn_bank_deg()
        # Пересчёт BEV-гомографии нужен если изменилась AGL или крен.
        # _compute_homography() занимает десятки μs, почти бесплатно по сравнению с warp.
        agl_changed = abs(agl_m - args.agl_m) > 1e-6
        roll_changed = abs(roll_deg) > 1.0  # порог: <1° игнорируем
        if agl_changed or roll_changed:
            bev_frame = BevRectifier.build(
                K_rect=undistorter.K_rect,
                image_size=undistorter.image_size,
                pitch_deg=args.pitch_deg,
                roll_deg=roll_deg,
                agl_m=agl_m,
                ground_span_m=args.ground_span_m,
                out_size=(args.bev_out_size, args.bev_out_size),
            )
        return bev_frame.warp(rect), bev_frame

    def _bev_mask(
        frame_bev: np.ndarray,
        bev_frame: BevRectifier,
        mask_raw: np.ndarray | None,
    ) -> np.ndarray | None:
        if mask_tracker is not None and mask_raw is not None:
            mask_rect = undistorter.undistort_image(mask_raw)
            mask_bev = bev_frame.warp_mask(mask_rect)
        else:
            mask_bev = None
        if args.snow_mask:
            snow = detect_snow_mask(
                frame_bev,
                v_threshold=args.snow_v_threshold,
                s_threshold=args.snow_s_threshold,
            )
            mask_bev = combine_with_aircraft_mask(snow, mask_bev)
        return mask_bev

    def _is_mask_unreliable() -> tuple[bool, str]:
        """Эвристика: перенос маски по LK ненадёжен → пропустить map-канал.

        Плохая маска опаснее отсутствия маски: ключевые точки на фюзеляже создают
        ложные фиксации. Лучше пропустить кадр и понадеяться на VO + следующий
        чистый кадр через 0.5-4 сек.
        """
        if mask_tracker is None:
            return False, ""
        diag = mask_tracker.last_diagnostics
        if diag.method in ("fallback_anchor", "uninitialized"):
            return True, f"method={diag.method}"
        yaw_rate_abs = abs(float(filt.x[StateFilter.IDX_YAW_RATE]))
        in_bank = yaw_rate_abs > math.radians(2.0)  # >2°/s — считаем банком
        far_from_anchor = abs(diag.frame_delta) > 60  # >2 сек @ 30 fps
        if diag.confidence < 0.3 and (in_bank or far_from_anchor):
            return True, (
                f"low_conf={diag.confidence:.2f} "
                f"bank={in_bank} delta={diag.frame_delta}"
            )
        return False, ""

    def _correct_view_centre_to_aircraft(
        lat: float,
        lon: float,
        bev_frame: BevRectifier,
    ) -> tuple[float, float]:
        """Перевод lat/lon из 'точка под центром BEV' в 'позиция самолёта'.

        Конфиг BEV по умолчанию центрирует кадр на оптической оси, которая для
        pitch=-30°, AGL=620м падает на землю ~1074м впереди самолёта.
        Без этой коррекции EKF получает систематический сдвиг на forward_m
        в направлении курса (особенно болезненно в разворотах: с yaw_rate
        дрейфует и предполагаемое «место под центром»).
        """
        forward_m, right_m = bev_frame.view_centre_body_m()
        if abs(forward_m) < 1e-6 and abs(right_m) < 1e-6:
            return lat, lon
        yaw_rad = float(filt.x[StateFilter.IDX_YAW])
        d_north_m = math.cos(yaw_rad) * forward_m - math.sin(yaw_rad) * right_m
        d_east_m = math.sin(yaw_rad) * forward_m + math.cos(yaw_rad) * right_m
        cos_lat = math.cos(math.radians(lat))
        if abs(cos_lat) < 1e-9:
            return lat, lon
        lat_aircraft = lat - d_north_m / 111320.0
        lon_aircraft = lon - d_east_m / (111320.0 * cos_lat)
        return float(lat_aircraft), float(lon_aircraft)

    rows: list[dict] = []
    map_fixes: list[tuple[float, float, float]] = []  # принятые (lat, lon, sigma_xy_m)
    n_map_attempt = 0
    n_map_accept = 0
    n_structural_attempt = 0
    n_structural_accept = 0
    n_vo_step = 0
    n_vo_valid = 0
    n_obstructed = 0
    n_map_skipped_obstruction = 0
    n_mask_skipped = 0
    n_consistency_rejects = 0
    n_retriever_skipped = 0
    n_dem_valid = 0
    n_dem_fallback = 0

    # Межсезонный bootstrap-буфер: пока не накоплено ``bootstrap_buffer``
    # принятых матчер-измерений, они не отправляются в EKF. Вместо этого
    # измерения кластеризуются по пространственной близости, а центроид
    # самого большого кластера используется для инициализации. Так режется
    # шум retriever порядка 25 %, который раньше давал ложный захват на
    # западном краю базы.
    bootstrap_buf: list[tuple[int, float, float, float, str]] = []
    bootstrap_done = False

    def _bootstrap_frame_count() -> int:
        return len({frame_idx for frame_idx, *_ in bootstrap_buf})

    def _cluster_centroid(
        points: list[tuple[int, float, float, float, str]], radius_m: float
    ) -> tuple[float, float, list[int]]:
        """Найти крупнейший кластер точек (frame, lat, lon, sigma, channel).
        Возвращает (centroid_lat, centroid_lon, member_frame_indices)."""
        n = len(points)
        best_count, best_indices = 0, []
        for i in range(n):
            members = []
            member_frames = set()
            for j in range(n):
                d_lat_m = (points[j][1] - points[i][1]) * 111320.0
                d_lon_m = (
                    (points[j][2] - points[i][2])
                    * 111320.0
                    * math.cos(math.radians(points[i][1]))
                )
                if math.hypot(d_lat_m, d_lon_m) <= radius_m:
                    members.append(j)
                    member_frames.add(points[j][0])
            if len(member_frames) > best_count:
                best_count = len(member_frames)
                best_indices = members
        if best_count == 0:
            return points[0][1], points[0][2], [points[0][0]]
        latest_by_frame = {}
        for i in best_indices:
            latest_by_frame[points[i][0]] = points[i]
        cl = list(latest_by_frame.values())
        return (
            float(np.mean([c[1] for c in cl])),
            float(np.mean([c[2] for c in cl])),
            list(latest_by_frame.keys()),
        )

    def _complete_bootstrap(
        c_lat: float,
        c_lon: float,
        member_frames: list[int],
        current_frame_idx: int,
    ) -> tuple[int, int, bool, bool]:
        filt.initialize_from_wgs84(
            c_lat,
            c_lon,
            args.start_alt_msl,
            yaw=filt.x[StateFilter.IDX_YAW],
            sigma_pos_m=200.0,
            sigma_yaw_rad=math.pi,
        )
        appearance_accept = 0
        structural_accept = 0
        appearance_flag = False
        structural_flag = False
        latest_by_frame = {}
        for point in bootstrap_buf:
            latest_by_frame[point[0]] = point
        for frame_idx in member_frames:
            _, la, lo, sg, channel = latest_by_frame[frame_idx]
            upd = filt.update_map_position_wgs84(la, lo, sigma_xy_m=max(sg, 20.0))
            if not upd.accepted:
                continue
            map_fixes.append((la, lo, sg))
            if channel == "structural":
                structural_accept += 1
                structural_flag |= frame_idx == current_frame_idx
            else:
                appearance_accept += 1
                appearance_flag |= frame_idx == current_frame_idx
        return appearance_accept, structural_accept, appearance_flag, structural_flag

    last_t = None
    t_start = time.time()
    print(
        f"[run] processing frames {f0}..{f1} ({f1 - f0} total, "
        f"VO stride {args.vo_stride}, map every {map_stride})"
    )
    print()

    for fi in range(f0, f1 + 1):
        ts = fi / src_fps
        if last_t is None:
            dt = 1.0 / src_fps
        else:
            dt = ts - last_t

        # Покадровые таймеры профилирования (мс). На бортовом компьютере смотреть
        # эти числа в CSV — главный способ найти узкое место.
        t_frame_start = time.perf_counter()
        timings_ms = {
            "t_decode_ms": 0.0,
            "t_mask_ms": 0.0,
            "t_obstr_ms": 0.0,
            "t_vo_ms": 0.0,
            "t_appearance_ms": 0.0,
            "t_structural_ms": 0.0,
            "t_frame_ms": 0.0,
        }

        _t = time.perf_counter()
        frame = proc.extract_frame(fi)
        if frame is None:
            continue
        timings_ms["t_decode_ms"] = (time.perf_counter() - _t) * 1000.0

        _t = time.perf_counter()
        mask_raw = (
            mask_tracker.mask_for_frame(fi, frame.shape[:2], frame=frame)
            if mask_tracker
            else None
        )
        timings_ms["t_mask_ms"] = (time.perf_counter() - _t) * 1000.0

        _t = time.perf_counter()
        obstr_result = (
            obstruction_detector.detect(frame, aircraft_mask=mask_raw)
            if obstruction_detector is not None
            else None
        )
        timings_ms["t_obstr_ms"] = (time.perf_counter() - _t) * 1000.0
        obstructed = (
            bool(obstr_result.is_obstructed) if obstr_result is not None else False
        )
        raw_obstructed = (
            bool(obstr_result.raw_obstructed) if obstr_result is not None else False
        )
        if obstructed:
            n_obstructed += 1
        if obstr_result is not None:
            obstr_metrics = obstr_result.metrics
            obstr_std = float(obstr_metrics.std)
            obstr_entropy = float(obstr_metrics.entropy)
            obstr_edge_density = float(obstr_metrics.edge_density)
            obstr_low_var_patch_frac = float(obstr_metrics.low_var_patch_frac)
            obstr_votes = int(obstr_metrics.votes)
        else:
            obstr_std = float("nan")
            obstr_entropy = float("nan")
            obstr_edge_density = float("nan")
            obstr_low_var_patch_frac = float("nan")
            obstr_votes = 0

        # 2. Прогноз EKF.
        filt.predict(dt)

        state_lat_for_agl, state_lon_for_agl, _ = filt.position_wgs84()
        agl_m = args.agl_m
        dem_agl = None
        if dem_lookup is not None:
            dem_agl = dem_lookup.height_agl(
                state_lat_for_agl,
                state_lon_for_agl,
                args.start_alt_msl,
            )
            if dem_agl is not None and math.isfinite(dem_agl) and dem_agl > 1.0:
                agl_m = float(dem_agl)
                n_dem_valid += 1
            else:
                n_dem_fallback += 1

        # 3. OF VO каждые vo_stride кадров.
        if (fi - f0) % args.vo_stride == 0:
            _t = time.perf_counter()
            step = vo.step(
                frame, dt=dt * args.vo_stride, agl_m=agl_m, aircraft_mask=mask_raw
            )
            timings_ms["t_vo_ms"] = (time.perf_counter() - _t) * 1000.0
            n_vo_step += 1
            if step.valid:
                n_vo_valid += 1
                filt.update_of_velocity(
                    np.array(step.velocity_body[:2]),
                    step.yaw_rate,
                    dt=dt * args.vo_stride,
                    sigma_v_mps=2.0,
                    sigma_yaw_rate_radps=math.radians(1.0),
                )

        # 4. Мягкое ограничение по высоте.
        filt.update_altitude(args.start_alt_msl, sigma_h_m=80.0)

        # 5. Мягкое ограничение по курсу, внутри ограничено по |yaw_rate|.
        filt.maybe_update_heading_prior(sigma_heading_rad=math.radians(20.0))

        # 6-11. Map measurement — адаптивное расписание через TrackState.
        accepted_this_frame = False
        appearance_accepted_this_frame = False
        structural_accepted_this_frame = False
        structural_seed_lat = None
        structural_seed_lon = None
        # После bootstrap оба канала складывают измерения сюда, а
        # межканальные ворота согласованности решают: принять оба или ни одного.
        app_pending: Optional[tuple[float, float, float, int]] = (
            None  # (lat, lon, sigma, inliers)
        )
        struct_pending: Optional[tuple[float, float, float, float]] = (
            None  # (lat, lon, sigma, score)
        )

        track_state.recompute_mode(
            t_sec=ts,
            sigma_pos_m=filt.position_sigma_m(),
            bootstrap_done=bootstrap_done,
        )
        mask_unreliable, mask_reason = _is_mask_unreliable()
        skip_map = obstructed or mask_unreliable
        run_appearance = track_state.should_run_appearance(ts, skip_map)
        run_structural = structural is not None and track_state.should_run_structural(
            ts, skip_map
        )
        if (
            mask_unreliable
            and not obstructed
            and (
                track_state.should_run_appearance(ts, obstructed=False)
                or (
                    structural is not None
                    and track_state.should_run_structural(ts, obstructed=False)
                )
            )
        ):
            n_mask_skipped += 1
            print(f"  f={fi:>5} t={ts:5.1f}s  MASK UNRELIABLE skip map ({mask_reason})")
        if run_structural:
            structural_seed_lat, structural_seed_lon, _ = filt.position_wgs84()

        # Учёт счётчиков и расписания.
        if run_appearance:
            track_state.mark_appearance_ran(ts)
            n_map_attempt += 1
            if obstructed:
                n_map_skipped_obstruction += 1
                print(
                    f"  f={fi:>5} t={ts:5.1f}s  OBSTRUCTION skip map "
                    f"votes={obstr_votes} std={obstr_std:.1f} H={obstr_entropy:.1f}"
                )
        if run_structural:
            track_state.mark_structural_ran(ts)
            n_structural_attempt += 1

        # BEV нужен и appearance-каналу (если кадр не перекрыт), и structural.
        # Считаем один раз и передаём в оба канала.
        run_app_real = run_appearance and not obstructed
        run_struct_real = run_structural
        frame_bev = bev_frame = mask_bev = None
        drone_class_map_bev = None
        if run_app_real or run_struct_real:
            frame_bev, bev_frame = _build_bev_frame(frame, agl_m)
            mask_bev = _bev_mask(frame_bev, bev_frame, mask_raw)
            if run_struct_real and args.semantic_structural_match and seg is not None:
                _t_sem_struct = time.perf_counter()
                drone_class_map_bev = seg.stable_class_map(frame_bev)
                timings_ms["t_structural_ms"] += (
                    time.perf_counter() - _t_sem_struct
                ) * 1000.0

        # В TRACK режиме (σ_pos < 60м) retriever избыточен: позиция уже
        # хорошо известна, тайл выводится из EKF без DINOv2 (~30-60мс
        # экономии на каждом appearance-вызове).
        skip_retriever = track_state.mode == TrackMode.TRACK and bootstrap_done
        if skip_retriever and run_app_real:
            n_retriever_skipped += 1

        # Тяжёлые вычислительные замыкания без побочных эффектов на общее состояние.
        # `sat_sem_cache` пишется только из appearance, structural к нему
        # не обращается, поэтому гонки за состояние здесь нет.
        def _match_against_candidate(cand: dict):
            """Прогон matcher против одного retriever-кандидата."""
            assert frame_bev is not None
            map_cv2_local, mbbox_local = _ensure_window(
                cand["tile_id"],
                cand["lat"],
                cand["lon"],
            )
            res_local = matcher.match(
                frame_bev,
                map_cv2_local,
                aircraft_mask=mask_bev,
            )
            mk_d = res_local["mkpts0"]
            mk_m = res_local["mkpts1"]
            if seg is not None and len(mk_d) > 0:
                if cand["tile_id"] not in sat_sem_cache:
                    sat_sem_cache[cand["tile_id"]] = seg.stable_class_map(map_cv2_local)
                sat_cmap = sat_sem_cache[cand["tile_id"]]
                s_xy = mk_m.astype(np.int32)
                s_xy[:, 0] = np.clip(s_xy[:, 0], 0, sat_cmap.shape[1] - 1)
                s_xy[:, 1] = np.clip(s_xy[:, 1], 0, sat_cmap.shape[0] - 1)
                sat_keep = sat_cmap[s_xy[:, 1], s_xy[:, 0]] > 0
                mk_d = mk_d[sat_keep]
                mk_m = mk_m[sat_keep]
            meas_local = compute_map_measurement(
                frame_shape=frame_bev.shape,
                mkpts_drone=mk_d,
                mkpts_map=mk_m,
                bbox=mbbox_local,
                map_shape=map_cv2_local.shape,
                last_homography_scale=None,
            )
            return meas_local

        def _appearance_compute():
            assert frame is not None
            assert frame_bev is not None
            if skip_retriever:
                pred_lat, pred_lon, _ = filt.position_wgs84()
                candidates = [
                    {
                        "tile_id": _ekf_tile_id(pred_lat, pred_lon),
                        "lat": pred_lat,
                        "lon": pred_lon,
                        "score": float("nan"),
                        "index": -1,
                    }
                ]
            else:
                k = max(1, int(args.retriever_top_k))
                candidates = retr.query_image(frame, top_k=k)
            # Перебираем top-k, выбираем по числу inliers.
            # Межсезонный шум retriever: top-1 не всегда правильный,
            # но среди top-3 правильный почти всегда есть (план §3.5).
            best_meas = None
            best_cand = candidates[0]
            best_inliers = -1
            early = int(args.retriever_early_accept_inliers)
            for cand in candidates:
                meas = _match_against_candidate(cand)
                inl = meas.num_inliers if meas.accepted else 0
                if inl > best_inliers:
                    best_inliers = inl
                    best_meas = meas
                    best_cand = cand
                # Раннее принятие: первый достаточно хороший кандидат — не тратим
                # XFeat на остальные.
                if meas.accepted and inl >= early:
                    break
            return best_meas, best_cand

        # Адаптивный радиус structural-поиска: в TRACK позиция точная →
        # ищем в узком окне (быстрее NCC); в RELOC σ большая → ищем
        # шире чтобы не промахнуться. Для масштабирования используем
        # 3·σ_pos с потолком в args.structural_search_radius_m.
        if bootstrap_done:
            adaptive_radius_m = float(
                np.clip(
                    3.0 * filt.position_sigma_m(),
                    100.0,  # нижний пол: ниже становится дороже rasterize чем NCC
                    args.structural_search_radius_m,
                )
            )
        else:
            adaptive_radius_m = args.structural_search_radius_m

        def _structural_compute():
            assert structural is not None
            assert frame_bev is not None
            assert structural_seed_lat is not None
            assert structural_seed_lon is not None
            return structural.match(
                frame_bev=frame_bev,
                seed_lat=structural_seed_lat,
                seed_lon=structural_seed_lon,
                bev_ground_span_m=args.ground_span_m,
                bev_mask=mask_bev,
                search_radius_m=adaptive_radius_m,
                drone_class_map=drone_class_map_bev,
            )

        app_compute = None
        struct_compute = None
        if args.parallel_channels and run_app_real and run_struct_real:
            # PyTorch (XFeat / DINOv2) и cv2.matchTemplate релизят GIL,
            # поэтому два потока действительно идут параллельно: GPU + CPU.
            _t_par = time.perf_counter()
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_app = ex.submit(_appearance_compute)
                fut_struct = ex.submit(_structural_compute)
                app_compute = fut_app.result()
                struct_compute = fut_struct.result()
            par_ms = (time.perf_counter() - _t_par) * 1000.0
            # Реальное время параллельной фазы записываем в обе графы: отдельных
            # таймингов на канал в этом режиме нет (см. --no-parallel-channels).
            timings_ms["t_appearance_ms"] = par_ms
            timings_ms["t_structural_ms"] += par_ms
        else:
            if run_app_real:
                _t = time.perf_counter()
                app_compute = _appearance_compute()
                timings_ms["t_appearance_ms"] = (time.perf_counter() - _t) * 1000.0
            if run_struct_real:
                _t = time.perf_counter()
                struct_compute = _structural_compute()
                timings_ms["t_structural_ms"] += (time.perf_counter() - _t) * 1000.0

        # ---- Постобработка appearance-канала ----
        if app_compute is not None:
            meas, _top = app_compute
            if meas is not None and meas.accepted:
                assert bev_frame is not None
                aircraft_lat, aircraft_lon = _correct_view_centre_to_aircraft(
                    meas.lat,
                    meas.lon,
                    bev_frame,
                )
                if not bootstrap_done:
                    bootstrap_buf.append(
                        (fi, aircraft_lat, aircraft_lon, meas.sigma_xy_m, "appearance")
                    )
                    print(
                        f"  f={fi:>5} t={ts:5.1f}s  bootstrap buffer "
                        f"[{_bootstrap_frame_count()}/{args.bootstrap_buffer} frames]: "
                        f"({aircraft_lat:.4f},{aircraft_lon:.4f}) inl={meas.num_inliers}"
                    )
                    if _bootstrap_frame_count() >= args.bootstrap_buffer:
                        c_lat, c_lon, member_frames = _cluster_centroid(
                            bootstrap_buf,
                            args.bootstrap_cluster_radius_m,
                        )
                        print(
                            f"  [bootstrap] largest cluster "
                            f"{len(member_frames)}/{_bootstrap_frame_count()} frames; "
                            f"centroid=({c_lat:.4f}, {c_lon:.4f}) — initialising EKF"
                        )
                        dn_map, dn_struct, app_ok, struct_ok = _complete_bootstrap(
                            c_lat,
                            c_lon,
                            member_frames,
                            fi,
                        )
                        n_map_accept += dn_map
                        n_structural_accept += dn_struct
                        accepted_this_frame = app_ok or struct_ok
                        appearance_accepted_this_frame = app_ok
                        structural_accepted_this_frame = struct_ok
                        bootstrap_done = True
                else:
                    app_pending = (
                        aircraft_lat,
                        aircraft_lon,
                        max(meas.sigma_xy_m, 10.0),
                        meas.num_inliers,
                    )

        # ---- Постобработка structural-канала ----
        if struct_compute is not None:
            sfix = struct_compute
            if sfix.accepted:
                assert bev_frame is not None
                sigma_xy_m = max(sfix.sigma_xy_m, 25.0)
                s_aircraft_lat, s_aircraft_lon = _correct_view_centre_to_aircraft(
                    sfix.lat,
                    sfix.lon,
                    bev_frame,
                )
                if not bootstrap_done:
                    bootstrap_buf.append(
                        (fi, s_aircraft_lat, s_aircraft_lon, sigma_xy_m, "structural")
                    )
                    struct_kind = (
                        "semantic-structural" if sfix.semantic_mode else "structural"
                    )
                    print(
                        f"  f={fi:>5} t={ts:5.1f}s  {struct_kind} bootstrap "
                        f"[{_bootstrap_frame_count()}/{args.bootstrap_buffer} frames]: "
                        f"({s_aircraft_lat:.4f},{s_aircraft_lon:.4f}) score={sfix.peak_score:.3f}"
                    )
                    if _bootstrap_frame_count() >= args.bootstrap_buffer:
                        c_lat, c_lon, member_frames = _cluster_centroid(
                            bootstrap_buf,
                            args.bootstrap_cluster_radius_m,
                        )
                        print(
                            f"  [bootstrap] largest cluster "
                            f"{len(member_frames)}/{_bootstrap_frame_count()} frames; "
                            f"centroid=({c_lat:.4f}, {c_lon:.4f}) — initialising EKF"
                        )
                        dn_map, dn_struct, app_ok, struct_ok = _complete_bootstrap(
                            c_lat,
                            c_lon,
                            member_frames,
                            fi,
                        )
                        n_map_accept += dn_map
                        n_structural_accept += dn_struct
                        accepted_this_frame = app_ok or struct_ok
                        appearance_accepted_this_frame = app_ok
                        structural_accepted_this_frame = struct_ok
                        bootstrap_done = True
                else:
                    struct_pending = (
                        s_aircraft_lat,
                        s_aircraft_lon,
                        sigma_xy_m,
                        sfix.peak_score,
                    )
            else:
                struct_kind = (
                    "semantic-structural" if sfix.semantic_mode else "structural"
                )
                print(
                    f"  f={fi:>5} t={ts:5.1f}s  {struct_kind} reject "
                    f"{sfix.reject_reason} score={sfix.peak_score:.3f} "
                    f"edges={sfix.n_drone_edges} sat={sfix.n_sat_features}"
                )

        # 12. Межканальные ворота согласованности (план §2.4).
        # Если оба канала вернули accept в одном кадре, но разошлись > 3σ
        # совместной ковариации — оба отбрасываются, фильтр продолжает
        # протяжку по VO до следующего согласованного обновления.
        if app_pending is not None and struct_pending is not None:
            d_lat_m = (app_pending[0] - struct_pending[0]) * 111320.0
            cos_lat = math.cos(math.radians(app_pending[0]))
            d_lon_m = (app_pending[1] - struct_pending[1]) * 111320.0 * cos_lat
            disagree_m = math.hypot(d_lat_m, d_lon_m)
            joint_sigma_m = math.hypot(app_pending[2], struct_pending[2])
            if disagree_m > 3.0 * joint_sigma_m:
                n_consistency_rejects += 1
                print(
                    f"  f={fi:>5} t={ts:5.1f}s  CONSISTENCY REJECT  "
                    f"app=({app_pending[0]:.4f},{app_pending[1]:.4f}) "
                    f"struct=({struct_pending[0]:.4f},{struct_pending[1]:.4f}) "
                    f"Δ={disagree_m:.0f}m gate=3·{joint_sigma_m:.0f}m"
                )
                app_pending = None
                struct_pending = None

        # 13. Применение pending в EKF.
        if app_pending is not None:
            a_lat, a_lon, a_sigma, a_inl = app_pending
            upd = filt.update_map_position_wgs84(a_lat, a_lon, sigma_xy_m=a_sigma)
            state_lat, state_lon, _ = filt.position_wgs84()
            print(
                f"  f={fi:>5} t={ts:5.1f}s  meas=({a_lat:.4f},{a_lon:.4f}) "
                f"σ={a_sigma:.0f}m inl={a_inl}  "
                f"state→({state_lat:.4f},{state_lon:.4f}) σ_pos={filt.position_sigma_m():.0f}m  "
                f"upd_acc={upd.accepted} d²={upd.mahalanobis2:.1f}"
            )
            if upd.accepted:
                n_map_accept += 1
                map_fixes.append((a_lat, a_lon, a_sigma))
                accepted_this_frame = True
                appearance_accepted_this_frame = True
                track_state.record_accept(ts)
        if struct_pending is not None:
            s_lat, s_lon, s_sigma, s_score = struct_pending
            upd = filt.update_map_position_wgs84(s_lat, s_lon, sigma_xy_m=s_sigma)
            state_lat, state_lon, _ = filt.position_wgs84()
            struct_kind = (
                "semantic-structural"
                if (args.semantic_structural_match and seg is not None)
                else "structural"
            )
            print(
                f"  f={fi:>5} t={ts:5.1f}s  {struct_kind}=({s_lat:.4f},{s_lon:.4f}) "
                f"score={s_score:.3f} σ={s_sigma:.0f}m  "
                f"state→({state_lat:.4f},{state_lon:.4f}) σ_pos={filt.position_sigma_m():.0f}m  "
                f"upd_acc={upd.accepted} d²={upd.mahalanobis2:.1f}"
            )
            if upd.accepted:
                n_structural_accept += 1
                map_fixes.append((s_lat, s_lon, s_sigma))
                accepted_this_frame = True
                structural_accepted_this_frame = True
                track_state.record_accept(ts)

        timings_ms["t_frame_ms"] = (time.perf_counter() - t_frame_start) * 1000.0

        # Логируем каждые log_stride кадров.
        if (fi - f0) % args.log_stride == 0:
            lat, lon, alt = filt.position_wgs84()
            x_e, y_n, _ = filt.position_enu()
            rows.append(
                {
                    "frame_idx": fi,
                    "t_sec": ts,
                    "lat": lat,
                    "lon": lon,
                    "alt_msl": alt,
                    "x_e": x_e,
                    "y_n": y_n,
                    "sigma_pos_m": filt.position_sigma_m(),
                    "speed_mps": filt.speed(),
                    "agl_m": agl_m,
                    "dem_agl_m": dem_agl if dem_agl is not None else float("nan"),
                    "heading_deg": filt.heading_deg(),
                    "scale_bias": filt.scale_bias(),
                    "scale_sigma": filt.scale_bias_sigma(),
                    "map_accepted": int(accepted_this_frame),
                    "appearance_accepted": int(appearance_accepted_this_frame),
                    "structural_accepted": int(structural_accepted_this_frame),
                    "track_mode": track_state.mode.value,
                    **timings_ms,
                    "obstructed": int(obstructed),
                    "raw_obstructed": int(raw_obstructed),
                    "obstr_votes": obstr_votes,
                    "obstr_std": obstr_std,
                    "obstr_entropy": obstr_entropy,
                    "obstr_edge_density": obstr_edge_density,
                    "obstr_low_var_patch_frac": obstr_low_var_patch_frac,
                }
            )

        # Кадр HUD-видео.
        if args.hud_video is not None:
            x_e_now, y_n_now, _ = filt.position_enu()
            hud_history_deque.append((ts, x_e_now, y_n_now))
            window_min_t = ts - args.hud_history_s
            while hud_history_deque and hud_history_deque[0][0] < window_min_t:
                hud_history_deque.pop(0)
            lat_now, lon_now, _ = filt.position_wgs84()
            hud_state = HudState(
                lat=lat_now,
                lon=lon_now,
                sigma_pos_m=filt.position_sigma_m(),
                speed_mps=filt.speed(),
                heading_deg=filt.heading_deg(),
                bank_deg=_coordinated_turn_bank_deg(),
                agl_m=agl_m,
                scale_bias=filt.scale_bias(),
                scale_sigma=filt.scale_bias_sigma(),
                track_mode=track_state.mode.value,
                obstructed=obstructed,
                mask_method=(
                    mask_tracker.last_diagnostics.method
                    if mask_tracker is not None
                    else ""
                ),
                mask_confidence=(
                    mask_tracker.last_diagnostics.confidence
                    if mask_tracker is not None
                    else 1.0
                ),
                app_attempts=n_map_attempt,
                app_accepts=n_map_accept,
                struct_attempts=n_structural_attempt,
                struct_accepts=n_structural_accept,
                map_accepted_now=appearance_accepted_this_frame,
                structural_accepted_now=structural_accepted_this_frame,
                consistency_rejected_now=False,  # выставляется выше при срабатывании
                t_frame_ms=timings_ms["t_frame_ms"],
                t_vo_ms=timings_ms["t_vo_ms"],
                t_app_ms=timings_ms["t_appearance_ms"],
                t_struct_ms=timings_ms["t_structural_ms"],
                enu_xy_history=[(x, y) for (_t, x, y) in hud_history_deque],
            )
            hud_frame = HUDDrawer.draw_pipeline_state(frame.copy(), hud_state)
            if hud_writer is None:
                hud_h, hud_w = hud_frame.shape[:2]
                fps_out = args.hud_fps if args.hud_fps > 0 else float(src_fps)
                hud_writer = cv2.VideoWriter(
                    str(args.hud_video),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps_out,
                    (hud_w, hud_h),
                )
                if not hud_writer.isOpened():
                    print(f"[run] HUD writer FAILED for {args.hud_video}; skipping HUD")
                    hud_writer = None
            if hud_writer is not None:
                hud_writer.write(hud_frame)

        last_t = ts

    if hud_writer is not None:
        hud_writer.release()
        print(f"[run] HUD video       : {args.hud_video}")

    elapsed = time.time() - t_start
    print(f"[run] elapsed         : {elapsed:.1f}s")
    print(f"[run] frames logged   : {len(rows)}")
    print(f"[run] OF VO steps     : {n_vo_step}  valid {n_vo_valid}")
    print(
        f"[run] map attempts    : {n_map_attempt}  accepted {n_map_accept}  "
        f"({100 * n_map_accept / max(1, n_map_attempt):.1f}%)"
    )
    if structural is not None:
        print(
            f"[run] structural      : attempts {n_structural_attempt}  "
            f"accepted {n_structural_accept}  "
            f"({100 * n_structural_accept / max(1, n_structural_attempt):.1f}%)"
        )
    print(
        f"[run] obstruction     : frames {n_obstructed}  map skips {n_map_skipped_obstruction}"
    )
    print(f"[run] mask reliability: skips {n_mask_skipped}")
    print(f"[run] consistency gate: rejects {n_consistency_rejects}")
    print(f"[run] retriever skip  : {n_retriever_skipped} appearance calls (TRACK)")
    print(f"[run] DEM AGL         : valid {n_dem_valid}  fallback {n_dem_fallback}")
    print(f"[run] map windows     : {len(mm_cache)}")
    print(
        f"[run] final scale_bias: {filt.scale_bias():.3f} ± {filt.scale_bias_sigma():.3f}"
    )
    print(f"[run] final pos σ     : {filt.position_sigma_m():.1f} m")
    if rows:
        # Сводка покадровых таймингов: медиана и 95-й перцентиль показывают
        # реальную бортовую нагрузку. P95 важнее среднего из-за тяжёлых кадров в хвосте.
        def _stat(key: str) -> tuple[float, float]:
            vals = np.array([r.get(key, 0.0) for r in rows], dtype=np.float64)
            vals = vals[vals > 0.0]
            if len(vals) == 0:
                return 0.0, 0.0
            return float(np.median(vals)), float(np.percentile(vals, 95))

        print("[run] timings (ms)    : median / p95 over frames")
        for key in (
            "t_decode_ms",
            "t_mask_ms",
            "t_obstr_ms",
            "t_vo_ms",
            "t_appearance_ms",
            "t_structural_ms",
            "t_frame_ms",
        ):
            med, p95 = _stat(key)
            print(f"          {key:<20s}  med={med:6.1f}  p95={p95:6.1f}")

    # ---- CSV --------------------------------------------------------------
    csv_path = args.output / "state.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[run] CSV -> {csv_path}")

    # ---- plots ------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lats = np.array([r["lat"] for r in rows])
    lons = np.array([r["lon"] for r in rows])
    sigs = np.array([r["sigma_pos_m"] for r in rows])
    times = np.array([r["t_sec"] for r in rows])
    scales = np.array([r["scale_bias"] for r in rows])
    scale_sigs = np.array([r["scale_sigma"] for r in rows])

    fix_lats = np.array([f[0] for f in map_fixes]) if map_fixes else np.array([])
    fix_lons = np.array([f[1] for f in map_fixes]) if map_fixes else np.array([])

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.plot(lons, lats, color="tab:blue", lw=1.0, label="EKF trajectory", alpha=0.8)
    if len(fix_lats):
        ax.scatter(
            fix_lons,
            fix_lats,
            s=80,
            color="tab:red",
            marker="x",
            label=f"map fixes ({len(map_fixes)})",
            zorder=5,
        )
    ax.scatter(
        lons[0], lats[0], s=120, color="tab:green", marker="o", label="start", zorder=6
    )
    ax.scatter(
        lons[-1], lats[-1], s=120, color="tab:purple", marker="s", label="end", zorder=6
    )
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_title(
        f"Trajectory  ({args.start_s:.0f}-{args.end_s:.0f}s, "
        f"{n_map_accept} accepted fixes)"
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(args.output / "trajectory.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(times, sigs, color="tab:blue")
    axes[0].set_ylabel("σ_pos (m)")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.3)
    axes[0].set_title("Filter state evolution")

    axes[1].plot(times, scales, color="tab:purple", label="scale_bias")
    axes[1].fill_between(
        times, scales - scale_sigs, scales + scale_sigs, alpha=0.2, color="tab:purple"
    )
    axes[1].axhline(1.0, color="grey", ls="--", lw=0.6)
    axes[1].set_ylabel("scale_bias")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    speeds = np.array([r["speed_mps"] for r in rows])
    axes[2].plot(times, speeds, color="tab:green")
    axes[2].set_ylabel("speed (m/s)")
    axes[2].set_xlabel("time (s)")
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output / "scale_history.png", dpi=120)
    plt.close(fig)

    print(f"[run] plots -> {args.output}/")


if __name__ == "__main__":
    main()
