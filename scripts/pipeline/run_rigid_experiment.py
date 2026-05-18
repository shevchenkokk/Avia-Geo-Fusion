import json
import matplotlib.pyplot as plt
import folium
import cv2
import numpy as np
import logging
import os
import shutil
import subprocess
from tqdm import tqdm
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.video_processor import VideoProcessor
from src.map_loader import MapDownloader
from src.map_manager import MapManager
from src.neural_matching import NeuralMatcher
from src.geolocator_rigid import GeolocatorRigid
from src.hud_drawer import HUDDrawer
from src.geo_segmentor import GeoSegmentor
from src.match_filters import filter_matches_by_class, format_class_stats
from src.video_discovery import pick_default_video

logging.basicConfig(level=logging.WARNING)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _estimate_track_confidence(match_stats: dict, match_count: int, is_keyframe: bool, backend: str | None) -> float:
    if not match_stats:
        return 0.0

    if not bool(match_stats.get("accepted", False)):
        return 0.0

    inlier_ratio = float(match_stats.get("inlier_ratio", 0.0))
    reproj_error = float(match_stats.get("reproj_error", 50.0))
    geom_matches = float(match_stats.get("matches", match_count))

    score_ratio = _clip01((inlier_ratio - 0.20) / 0.55)
    score_reproj = _clip01((18.0 - reproj_error) / 18.0)
    score_count = _clip01(geom_matches / 60.0)

    score = 0.50 * score_ratio + 0.30 * score_reproj + 0.20 * score_count
    if not is_keyframe:
        score *= 0.95
    if backend == "orb":
        score *= 0.90
    return _clip01(score)


def _update_tracking_state(
    state: str,
    high_conf_streak: int,
    low_conf_streak: int,
    confidence: float,
    success: bool,
) -> tuple[str, int, int]:
    if not success:
        low_conf_streak += 1
        high_conf_streak = 0
    elif confidence >= 0.62:
        high_conf_streak += 1
        low_conf_streak = max(0, low_conf_streak - 1)
    elif confidence >= 0.40:
        high_conf_streak = 0
        low_conf_streak = max(0, low_conf_streak - 1)
    else:
        low_conf_streak += 1
        high_conf_streak = 0

    if state == "TRACK":
        if low_conf_streak >= 2:
            state = "WEAK"
    elif state == "WEAK":
        if low_conf_streak >= 4 and confidence < 0.40:
            state = "RELOCALIZE"
        elif high_conf_streak >= 2:
            state = "TRACK"
    else:  # RELOCALIZE
        if high_conf_streak >= 2:
            state = "WEAK"

    return state, high_conf_streak, low_conf_streak


def _draw_tracking_state(img: np.ndarray, state: str, confidence: float, x: int = 20, y: int = 34) -> np.ndarray:
    color = (40, 220, 40)
    if state == "WEAK":
        color = (0, 200, 255)
    elif state == "RELOCALIZE":
        color = (40, 40, 255)

    label = f"STATE: {state}  conf={confidence:.2f}"
    cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2, cv2.LINE_AA)
    return img


def _select_visual_matches(pts_drone: np.ndarray, pts_map: np.ndarray, max_points: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic point subset to avoid frame-to-frame flicker from random sampling."""
    if len(pts_drone) <= max_points:
        return pts_drone, pts_map

    # Stable ordering by screen coordinates.
    order = np.lexsort((pts_drone[:, 1], pts_drone[:, 0]))
    step = max(1, int(np.ceil(len(order) / float(max_points))))
    keep = order[::step][:max_points]
    return pts_drone[keep], pts_map[keep]


def _overlay_ground_mask(drone_img: np.ndarray, stable_mask: np.ndarray | None, alpha: float = 0.22) -> np.ndarray:
    if stable_mask is None:
        return drone_img
    if len(drone_img.shape) == 2:
        drone_img = cv2.cvtColor(drone_img, cv2.COLOR_GRAY2BGR)
    mask_colored = np.zeros_like(drone_img)
    mask_colored[:, :, 1] = stable_mask
    return cv2.addWeighted(drone_img, 1.0, mask_colored, alpha, 0)


def _draw_geometry_debug(
    img: np.ndarray,
    map_x_offset: int,
    projection_polygon: np.ndarray | None,
    match_stats: dict,
    map_scale_x: float = 1.0,
    map_scale_y: float = 1.0,
) -> np.ndarray:
    out = img

    if projection_polygon is not None and len(projection_polygon) >= 4:
        poly = np.asarray(projection_polygon, dtype=np.float32).copy()
        poly[:, 0] *= float(map_scale_x)
        poly[:, 1] *= float(map_scale_y)
        poly[:, 0] += float(map_x_offset)
        poly_i = poly.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [poly_i], True, (255, 210, 0), 2, cv2.LINE_AA)

    model = str(match_stats.get("model", "none")).upper()
    inlier_ratio = float(match_stats.get("inlier_ratio", 0.0))
    reproj_error = float(match_stats.get("reproj_error", float("inf")))
    accepted = bool(match_stats.get("accepted", False))
    reject_reason = str(match_stats.get("reject_reason", ""))

    status = "OK" if accepted else f"REJ:{reject_reason or 'n/a'}"
    text = f"GEOM {model} inlier={inlier_ratio:.2f} err={reproj_error:.1f} {status}"
    text_x = int(map_x_offset + 20)
    cv2.putText(out, text, (text_x, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (text_x, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 230, 80), 1, cv2.LINE_AA)
    return out


def _make_unique_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    idx = 2
    while True:
        candidate = parent / f"{stem}_v{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def _finalize_video_h264(raw_path: Path, final_path: Path) -> Path:
    """Transcode OpenCV mp4v output to H.264 for better VS Code compatibility."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raw_path.replace(final_path)
        print("[WARN] ffmpeg not found, keeping raw mp4v output.")
        return final_path

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(raw_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(final_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            raw_path.unlink(missing_ok=True)
        except Exception:
            pass
        return final_path
    except Exception:
        raw_path.replace(final_path)
        print("[WARN] H.264 transcode failed, keeping raw mp4v output.")
        return final_path

def generate_final_video(video_path: str, output_path: str, duration_sec: int = 20, fps_out: int = 15):
    output_path_obj = _make_unique_path(output_path)
    raw_output_path_obj = _make_unique_path(output_path_obj.with_name(f"{output_path_obj.stem}_raw{output_path_obj.suffix}"))
    print(f"Генерация HQ HUD видео: {output_path_obj}")
    
    processor = VideoProcessor(video_path, default_geo=(55.086025, 38.149033, 750.0))
    video_fps = processor.info.fps
    total_frames = duration_sec * fps_out
    frame_step = int(video_fps / fps_out)

    map_zoom = int(os.environ.get("MAP_ZOOM", "17"))
    map_window_radius = int(os.environ.get("MAP_WINDOW_RADIUS", "3"))
    map_closer_threshold = int(os.environ.get("MAP_CLOSER_THRESHOLD", "1"))
    vis_layout = os.environ.get("VIS_LAYOUT", "split").lower()
    
    # Инициализация MapManager — скользящее окно карты вокруг стартовой позиции
    pos_t0 = processor.telemetry.get_position_at_frame(0.0)
    map_manager = MapManager(zoom=map_zoom, window_radius=map_window_radius, closer_threshold=map_closer_threshold)
    map_cv2, bbox = map_manager.initialize(pos_t0.latitude, pos_t0.longitude)
    
    matcher_backend = os.environ.get("MATCHER_BACKEND", "auto")
    matcher_features = os.environ.get("MATCHER_FEATURES", "auto")
    min_ground_points = int(os.environ.get("MIN_GROUND_POINTS", "8"))
    auto_fallback_to_loftr = os.environ.get("MATCHER_AUTO_FALLBACK", "1") != "0"
    seg_backend = os.environ.get("SEG_BACKEND", "heuristic").lower()
    seg_cfg = os.environ.get("SEG_MODEL_CFG", "").strip() or None
    seg_ckpt = os.environ.get("SEG_MODEL_CKPT", "").strip() or None
    seg_device = os.environ.get("SEG_DEVICE", "cuda:0")
    seg_classes_raw = os.environ.get("SEG_STABLE_CLASSES", "2,3,4")
    try:
        seg_stable_classes = {int(v.strip()) for v in seg_classes_raw.split(",") if v.strip()}
    except Exception:
        seg_stable_classes = {2, 3, 4}
    matcher = NeuralMatcher(backend=matcher_backend, lightglue_features=matcher_features)
    geolocator = GeolocatorRigid(bbox=bbox, map_shape=map_cv2.shape, smoothing_factor=0.2)
    geo_segmentor = GeoSegmentor(
        top_crop_ratio=0.32,
        texture_percentile=55.0,
        backend=seg_backend,
        mmseg_config=seg_cfg,
        mmseg_checkpoint=seg_ckpt,
        mmseg_device=seg_device,
        stable_class_ids=seg_stable_classes,
    )
    map_geo_segmentor = GeoSegmentor(
        top_crop_ratio=0.0,
        texture_percentile=55.0,
        backend=seg_backend,
        mmseg_config=seg_cfg,
        mmseg_checkpoint=seg_ckpt,
        mmseg_device=seg_device,
        stable_class_ids=seg_stable_classes,
    )
    map_class_map = map_geo_segmentor.stable_class_map(map_cv2)
    map_stable_mask = np.where(map_class_map > 0, 255, 0).astype(np.uint8)
    
    # Узнаем размеры для сохранения видео
    first_frame = processor.extract_frame(0)
    if first_frame is None: return
        
    test_match = matcher.match(first_frame, map_cv2)
    print(
        "Matcher backend:",
        test_match.get("backend", matcher_backend),
        "features:",
        test_match.get("features", matcher_features),
        "min_ground_points:",
        min_ground_points,
        "zoom:",
        map_zoom,
        "layout:",
        vis_layout,
        "seg_backend:",
        geo_segmentor.backend,
    )
    display_drone_h = int(first_frame.shape[0])
    display_drone_w = int(first_frame.shape[1])
    display_map_h = display_drone_h
    display_map_w = max(
        1,
        int(round(map_cv2.shape[1] * (display_map_h / float(max(1, map_cv2.shape[0]))))),
    )
    map_scale_x = display_map_w / float(max(1, map_cv2.shape[1]))
    map_scale_y = display_map_h / float(max(1, map_cv2.shape[0]))

    debug_w = display_drone_w + display_map_w
    debug_h = display_drone_h * (2 if vis_layout == "split" else 1)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(raw_output_path_obj), fourcc, fps_out, (debug_w, debug_h))
    
    print("Рендер кадров: ключевые сопоставления + оптический поток...")
    
    # Переменные трекинга Лукаса-Канаде.
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    
    prev_gray_drone = None
    tracked_pts_drone = None
    tracked_pts_map = None
    keyframe_interval = 10  # Раз в секунду обновляем тяжёлое сопоставление.
    min_tracked_points = 20
    tracking_state = "TRACK"
    high_conf_streak = 0
    low_conf_streak = 0
    last_confidence = 0.0
    sparse_match_streak = 0
    
    # Метрики для графиков
    kalman_trajectory = []
    raw_trajectory = []
    
    def _mask_panel(mask: np.ndarray | None, target_h: int, target_w: int, title: str) -> np.ndarray:
        panel = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        if mask is not None:
            if mask.shape[:2] != (target_h, target_w):
                mask_resized = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            else:
                mask_resized = mask
            panel[:, :, 1] = mask_resized

        cv2.putText(panel, title, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2, cv2.LINE_AA)
        return panel

    def render_side_by_side(
        drone_img,
        map_img,
        pts_drone,
        pts_map,
        stable_mask_drone=None,
        stable_mask_map=None,
        max_points=100,
    ):
        """В split-режиме рисует raw+matches сверху и отдельные маски снизу."""
        if len(drone_img.shape) == 2:
            drone_img = cv2.cvtColor(drone_img, cv2.COLOR_GRAY2BGR)

        drone_disp = cv2.resize(drone_img, (display_drone_w, display_drone_h), interpolation=cv2.INTER_AREA)
        map_disp = cv2.resize(map_img, (display_map_w, display_map_h), interpolation=cv2.INTER_AREA)

        mask_drone_disp = None
        if stable_mask_drone is not None:
            mask_drone_disp = cv2.resize(stable_mask_drone, (display_drone_w, display_drone_h), interpolation=cv2.INTER_NEAREST)

        mask_map_disp = None
        if stable_mask_map is not None:
            mask_map_disp = cv2.resize(stable_mask_map, (display_map_w, display_map_h), interpolation=cv2.INTER_NEAREST)

        if vis_layout == "overlay":
            drone_top = _overlay_ground_mask(drone_disp, mask_drone_disp)
            map_top = _overlay_ground_mask(map_disp, mask_map_disp, alpha=0.18)
        else:
            drone_top = drone_disp.copy()
            map_top = map_disp.copy()

        top = np.zeros((display_drone_h, display_drone_w + display_map_w, 3), dtype=np.uint8)
        top[:, :display_drone_w] = drone_top
        top[:, display_drone_w:display_drone_w + display_map_w] = map_top

        pts_drone_draw, pts_map_draw = _select_visual_matches(
            np.asarray(pts_drone, dtype=np.float32),
            np.asarray(pts_map, dtype=np.float32),
            max_points=max_points,
        )

        if len(pts_drone_draw) > 0:
            pts_drone_draw = pts_drone_draw.copy()
            pts_drone_draw[:, 0] *= display_drone_w / float(max(1, drone_img.shape[1]))
            pts_drone_draw[:, 1] *= display_drone_h / float(max(1, drone_img.shape[0]))

        if len(pts_map_draw) > 0:
            pts_map_draw = pts_map_draw.copy()
            pts_map_draw[:, 0] *= display_map_w / float(max(1, map_img.shape[1]))
            pts_map_draw[:, 1] *= display_map_h / float(max(1, map_img.shape[0]))

        for p1, p2 in zip(pts_drone_draw, pts_map_draw):
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]) + display_drone_w, int(p2[1]))
            cv2.line(top, pt1, pt2, (0, 255, 120), 1, cv2.LINE_AA)
            cv2.circle(top, pt1, 3, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(top, pt2, 3, (255, 0, 0), -1, cv2.LINE_AA)

        if vis_layout != "split":
            return top

        bottom = np.zeros_like(top)
        drone_mask_panel = _mask_panel(mask_drone_disp, display_drone_h, display_drone_w, "DRONE SEG MASK")
        map_mask_panel = _mask_panel(mask_map_disp, display_map_h, display_map_w, "MAP SEG MASK")
        bottom[:, :display_drone_w] = drone_mask_panel
        bottom[:, display_drone_w:display_drone_w + display_map_w] = map_mask_panel

        return np.vstack([top, bottom])

    def filter_matches_by_ground(stable_mask_drone, stable_mask_map, pts_drone, pts_map):
        """Оставляет только точки, попадающие в маску и на дроне, и на карте."""
        if len(pts_drone) == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        h0, w0 = stable_mask_drone.shape[:2]
        h1, w1 = stable_mask_map.shape[:2]

        valid_indices = []
        for point_idx, (p0, p1) in enumerate(zip(pts_drone, pts_map)):
            x0, y0 = int(round(p0[0])), int(round(p0[1]))
            x1, y1 = int(round(p1[0])), int(round(p1[1]))
            ok_drone = 0 <= x0 < w0 and 0 <= y0 < h0 and stable_mask_drone[y0, x0] > 0
            ok_map = 0 <= x1 < w1 and 0 <= y1 < h1 and stable_mask_map[y1, x1] > 0
            if ok_drone and ok_map:
                valid_indices.append(point_idx)

        if not valid_indices:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        return (
            np.asarray([pts_drone[idx] for idx in valid_indices], dtype=np.float32),
            np.asarray([pts_map[idx] for idx in valid_indices], dtype=np.float32),
        )

    
    for i in tqdm(range(total_frames)):
        frame_idx = i * frame_step
        frame = processor.extract_frame(frame_idx)
        if frame is None: break
            
        drone_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        drone_class_map = geo_segmentor.stable_class_map(frame)
        stable_mask = np.where(drone_class_map > 0, 255, 0).astype(np.uint8)
        seg_filter_applied = False
        raw_match_count = 0
        last_class_stats: dict = {}
        
        if tracking_state == "TRACK":
            dynamic_keyframe_interval = keyframe_interval
        elif tracking_state == "WEAK":
            dynamic_keyframe_interval = max(2, keyframe_interval // 2)
        else:
            dynamic_keyframe_interval = 1

        needs_keyframe = False
        if tracking_state == "RELOCALIZE":
            tracked_pts_drone = None
            tracked_pts_map = None
            needs_keyframe = True
        elif tracked_pts_drone is None or i % dynamic_keyframe_interval == 0:
            needs_keyframe = True
        elif len(tracked_pts_drone) < min_tracked_points:
            needs_keyframe = True
            
        if needs_keyframe:
            # === Полный цикл (Neural Network: LoFTR) ===
            match_result = matcher.match(frame, map_cv2)
            
            raw_mkpts_drone = match_result['mkpts0']
            raw_mkpts_map = match_result['mkpts1']
            raw_match_count = int(len(raw_mkpts_drone))

            if matcher.backend == "lightglue":
                if raw_match_count < min_ground_points:
                    sparse_match_streak += 1
                else:
                    sparse_match_streak = 0

                if auto_fallback_to_loftr and sparse_match_streak >= 3:
                    print("\n[Matcher] LightGlue sparse matches -> fallback to LoFTR")
                    matcher = NeuralMatcher(backend="loftr")
                    sparse_match_streak = 0
                    match_result = matcher.match(frame, map_cv2)
                    raw_mkpts_drone = match_result['mkpts0']
                    raw_mkpts_map = match_result['mkpts1']
                    raw_match_count = int(len(raw_mkpts_drone))
            else:
                sparse_match_streak = 0

            # --- СЕМАНТИКА: class-aware гейт (road↔road, bldg↔bldg, veg↔veg). ---
            # Отсекает cross-class ложняки от LoFTR/LightGlue до того, как они
            # попадут в RANSAC и поведут гомографию в сторону.
            mkpts_drone_seg, mkpts_map_seg, class_stats = filter_matches_by_class(
                raw_mkpts_drone,
                raw_mkpts_map,
                drone_class_map,
                map_class_map,
                schema=geo_segmentor.schema,
            )
            last_class_stats = class_stats
            if len(mkpts_drone_seg) >= min_ground_points:
                mkpts_drone, mkpts_map = mkpts_drone_seg, mkpts_map_seg
                seg_filter_applied = True
            else:
                # Слишком мало совместимых по классу — падаем обратно на бинарный
                # гейт, иначе геолокатор вообще лишится точек на плохом кадре.
                mkpts_drone, mkpts_map = filter_matches_by_ground(
                    stable_mask, map_stable_mask, raw_mkpts_drone, raw_mkpts_map,
                )

            # Визуально показываем только надежные (seg-filtered) точки.
            if seg_filter_applied:
                viz_mkpts_drone, viz_mkpts_map = mkpts_drone, mkpts_map
            else:
                viz_mkpts_drone = np.empty((0, 2), dtype=np.float32)
                viz_mkpts_map = np.empty((0, 2), dtype=np.float32)
            
            # Унифицированная отрисовка для Keyframe (без рывков цвета)
            base_img = render_side_by_side(
                frame,
                map_cv2,
                viz_mkpts_drone,
                viz_mkpts_map,
                stable_mask_drone=stable_mask,
                stable_mask_map=map_stable_mask,
            )
            
            # Сохраняем стейт для OF
            if len(mkpts_drone) > 0:
                tracked_pts_drone = np.array(mkpts_drone, dtype=np.float32).reshape(-1, 1, 2)
                tracked_pts_map = np.array(mkpts_map, dtype=np.float32).reshape(-1, 1, 2)
                prev_gray_drone = drone_gray.copy()
            else:
                tracked_pts_drone = None
        else:
            # === Сверхбыстрый трекинг Оптическим потоком (Optical Flow) ===
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray_drone, drone_gray, tracked_pts_drone, None, **lk_params)

            if p1 is None or st is None:
                tracked_pts_drone = None
                tracked_pts_map = None
                mkpts_drone = np.empty((0, 2), dtype=np.float32)
                mkpts_map = np.empty((0, 2), dtype=np.float32)
                base_img = render_side_by_side(
                    frame,
                    map_cv2,
                    mkpts_drone,
                    mkpts_map,
                    stable_mask_drone=stable_mask,
                    stable_mask_map=map_stable_mask,
                )
                prev_gray_drone = drone_gray.copy()
                result_rigid = geolocator.estimate_drone_position(frame.shape, mkpts_drone, mkpts_map, is_keyframe=needs_keyframe)
                if result_rigid is not None:
                    gps_coord = (result_rigid[0], result_rigid[1])
                else:
                    gps_coord = None
                last_confidence = 0.0
                prev_state = tracking_state
                tracking_state, high_conf_streak, low_conf_streak = _update_tracking_state(
                    tracking_state,
                    high_conf_streak,
                    low_conf_streak,
                    last_confidence,
                    success=False,
                )
                if tracking_state != prev_state:
                    print(f"[Tracker] state {prev_state} -> {tracking_state} (conf={last_confidence:.2f})")
                gps_display = gps_coord if gps_coord is not None else getattr(geolocator, "last_candidate_gps", None)
                final_img = HUDDrawer.draw_telemetry(
                    base_img,
                    gps_display,
                    len(mkpts_drone),
                    is_ai=True,
                    gps_locked=gps_coord is not None,
                    reject_reason=str(getattr(geolocator, "last_match_stats", {}).get("reject_reason", "")),
                    map_zoom=map_zoom,
                    class_stats_line=format_class_stats(last_class_stats) if last_class_stats else None,
                )
                final_img = _draw_tracking_state(final_img, tracking_state, last_confidence, x=display_drone_w + 20, y=34)
                if final_img.shape[:2] != (debug_h, debug_w):
                    final_img = cv2.resize(final_img, (debug_w, debug_h))
                out.write(final_img)
                continue

            # Проверка вперёд-назад для устойчивости трекинга
            p0r, st_back, _ = cv2.calcOpticalFlowPyrLK(drone_gray, prev_gray_drone, p1, None, **lk_params)
            if p0r is None or st_back is None:
                tracked_pts_drone = None
                tracked_pts_map = None
                mkpts_drone = np.empty((0, 2), dtype=np.float32)
                mkpts_map = np.empty((0, 2), dtype=np.float32)
                base_img = render_side_by_side(
                    frame,
                    map_cv2,
                    mkpts_drone,
                    mkpts_map,
                    stable_mask_drone=stable_mask,
                    stable_mask_map=map_stable_mask,
                )
                prev_gray_drone = drone_gray.copy()
                result_rigid = geolocator.estimate_drone_position(frame.shape, mkpts_drone, mkpts_map, is_keyframe=needs_keyframe)
                if result_rigid is not None:
                    gps_coord = (result_rigid[0], result_rigid[1])
                else:
                    gps_coord = None
                last_confidence = 0.0
                prev_state = tracking_state
                tracking_state, high_conf_streak, low_conf_streak = _update_tracking_state(
                    tracking_state,
                    high_conf_streak,
                    low_conf_streak,
                    last_confidence,
                    success=False,
                )
                if tracking_state != prev_state:
                    print(f"[Tracker] state {prev_state} -> {tracking_state} (conf={last_confidence:.2f})")
                gps_display = gps_coord if gps_coord is not None else getattr(geolocator, "last_candidate_gps", None)
                final_img = HUDDrawer.draw_telemetry(
                    base_img,
                    gps_display,
                    len(mkpts_drone),
                    is_ai=True,
                    gps_locked=gps_coord is not None,
                    reject_reason=str(getattr(geolocator, "last_match_stats", {}).get("reject_reason", "")),
                    map_zoom=map_zoom,
                    class_stats_line=format_class_stats(last_class_stats) if last_class_stats else None,
                )
                final_img = _draw_tracking_state(final_img, tracking_state, last_confidence, x=display_drone_w + 20, y=34)
                if final_img.shape[:2] != (debug_h, debug_w):
                    final_img = cv2.resize(final_img, (debug_w, debug_h))
                out.write(final_img)
                continue

            st_flat = st.reshape(-1) == 1
            st_back_flat = st_back.reshape(-1) == 1

            prev_pts_flat = tracked_pts_drone.reshape(-1, 2)
            p1_flat = p1.reshape(-1, 2)
            p0r_flat = p0r.reshape(-1, 2)

            fb_error = np.linalg.norm(prev_pts_flat - p0r_flat, axis=1)
            fb_ok = fb_error < 1.5
            valid_flow = st_flat & st_back_flat & fb_ok

            # Отфильтровываем успешные точки (status + forward-backward)
            good_new = p1_flat[valid_flow]
            good_map = tracked_pts_map.reshape(-1, 2)[valid_flow]
            raw_match_count = int(len(good_new))
            
            # Отсекаем точки, улетевшие за границу кадра
            valid_idx = []
            h, w = frame.shape[:2]
            for idx, pt in enumerate(good_new):
                if 0 <= pt[0] < w and 0 <= pt[1] < h:
                    valid_idx.append(idx)
                    
            good_new = good_new[valid_idx]
            good_map = good_map[valid_idx]
            
            mkpts_drone = good_new.reshape(-1, 2)
            mkpts_map = good_map.reshape(-1, 2)

            # Дополнительно фильтруем OF-точки сегментацией по текущему кадру
            mkpts_drone_seg, mkpts_map_seg = filter_matches_by_ground(stable_mask, map_stable_mask, mkpts_drone, mkpts_map)
            min_track_points = max(4, min_ground_points // 2)
            if len(mkpts_drone_seg) >= min_track_points:
                mkpts_drone, mkpts_map = mkpts_drone_seg, mkpts_map_seg
                seg_filter_applied = True

            if seg_filter_applied:
                viz_mkpts_drone, viz_mkpts_map = mkpts_drone, mkpts_map
            else:
                viz_mkpts_drone = np.empty((0, 2), dtype=np.float32)
                viz_mkpts_map = np.empty((0, 2), dtype=np.float32)
            
            # Визуализация 
            base_img = render_side_by_side(
                frame,
                map_cv2,
                viz_mkpts_drone,
                viz_mkpts_map,
                stable_mask_drone=stable_mask,
                stable_mask_map=map_stable_mask,
            )
            
            # Обновляем стейт на следующий кадр
            tracked_pts_drone = mkpts_drone.reshape(-1, 1, 2) if len(mkpts_drone) > 0 else None
            tracked_pts_map = mkpts_map.reshape(-1, 1, 2) if len(mkpts_map) > 0 else None
            prev_gray_drone = drone_gray.copy()
        
        # Геолокация
        result_rigid = geolocator.estimate_drone_position(frame.shape, mkpts_drone, mkpts_map, is_keyframe=needs_keyframe)
        if result_rigid is not None:
            gps_coord = (result_rigid[0], result_rigid[1])
        else:
            gps_coord = None

        match_stats = getattr(geolocator, "last_match_stats", {})
        projection_polygon = getattr(geolocator, "last_projection_polygon", None)
        base_img = _draw_geometry_debug(
            base_img,
            display_drone_w,
            projection_polygon,
            match_stats,
            map_scale_x=map_scale_x,
            map_scale_y=map_scale_y,
        )
        seg_status = "ON" if seg_filter_applied else "BYPASS"
        match_line = f"MATCH used={len(mkpts_drone)} raw={raw_match_count} seg={seg_status}"
        match_x = int(display_drone_w + 20)
        cv2.putText(base_img, match_line, (match_x, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(base_img, match_line, (match_x, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 210), 1, cv2.LINE_AA)
        last_confidence = _estimate_track_confidence(match_stats, len(mkpts_drone), needs_keyframe, matcher.backend)
        prev_state = tracking_state
        tracking_state, high_conf_streak, low_conf_streak = _update_tracking_state(
            tracking_state,
            high_conf_streak,
            low_conf_streak,
            last_confidence,
            success=gps_coord is not None,
        )
        if tracking_state != prev_state:
            print(f"[Tracker] state {prev_state} -> {tracking_state} (conf={last_confidence:.2f})")

        if gps_coord is None or tracking_state == "RELOCALIZE":
            tracked_pts_drone = None
            tracked_pts_map = None
        
        # -- СКОЛЬЗЯЩЕЕ ОКНО: обновляем карту если дрон приблизился к краю --
        if gps_coord is not None:
            new_map_cv2, new_bbox = map_manager.update(gps_coord[0], gps_coord[1])
            # Если MapManager сдвинул окно — обновляем карту и geolocator (без потери Калмана)
            if new_bbox != bbox:
                map_cv2 = new_map_cv2
                bbox = new_bbox
                geolocator.update_bbox(new_bbox, map_cv2.shape)
                map_class_map = map_geo_segmentor.stable_class_map(map_cv2)
                map_stable_mask = np.where(map_class_map > 0, 255, 0).astype(np.uint8)
                # Сбрасываем трекинг точек, т.к. карта изменилась
                tracked_pts_drone = None
                tracked_pts_map = None
                print(f"\n[MapManager] Карта сдвинута → новый bbox: {new_bbox}")
        
        # Собираем метрики для графиков
        if gps_coord is not None and hasattr(geolocator, 'last_raw_gps'):
            kalman_trajectory.append(gps_coord)
            raw_trajectory.append(geolocator.last_raw_gps)
        
        # Рисуем HUD (Телеметрию поверх видео)
        gps_display = gps_coord if gps_coord is not None else getattr(geolocator, "last_candidate_gps", None)
        final_img = HUDDrawer.draw_telemetry(
            base_img,
            gps_display,
            len(mkpts_drone),
            is_ai=True,
            gps_locked=gps_coord is not None,
            reject_reason=str(match_stats.get("reject_reason", "")),
            map_zoom=map_zoom,
            seg_status=seg_status,
            match_used=len(mkpts_drone),
            match_raw=raw_match_count,
            class_stats_line=format_class_stats(last_class_stats) if last_class_stats else None,
        )
        final_img = _draw_tracking_state(final_img, tracking_state, last_confidence, x=display_drone_w + 20, y=34)
        
        # Если размер съехал из-за четностей в PyTorch, выравниваем:
        if final_img.shape[:2] != (debug_h, debug_w):
            final_img = cv2.resize(final_img, (debug_w, debug_h))
            
        out.write(final_img)
        
    out.release()
    final_output_path = _finalize_video_h264(raw_output_path_obj, output_path_obj)
    print(f"\n[УСПЕХ] Видео сохранено в {final_output_path}")
    
    # Генерация графиков и метрик
    print("Генерация графиков анализа...")
    if len(kalman_trajectory) > 0 and len(raw_trajectory) > 0:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        sns.set_theme(style="darkgrid")
        plt.figure(figsize=(10, 8))
        
        raw_lats = [pt[0] for pt in raw_trajectory]
        raw_lons = [pt[1] for pt in raw_trajectory]
        kalman_lats = [pt[0] for pt in kalman_trajectory]
        kalman_lons = [pt[1] for pt in kalman_trajectory]
        
        plt.plot(raw_lons, raw_lats, 'r--', alpha=0.5, label='Original CNN Detections (Jittery)')
        plt.plot(kalman_lons, kalman_lats, 'b-', linewidth=2, label='Kalman + Optical Flow (Target Track)')
        plt.scatter(kalman_lons[0], kalman_lats[0], color='green', s=100, label='Start Point', zorder=5)
        plt.scatter(kalman_lons[-1], kalman_lats[-1], color='orange', s=100, label='End Point', zorder=5)
        
        plt.title('Траектория локализации: Сырые данные LoFTR vs Гибридный Фильтр')
        plt.xlabel('Longitude (°)')
        plt.ylabel('Latitude (°)')
        plt.legend()
        plt.tight_layout()
        
        plot_path = _make_unique_path("results/trajectory_metrics.png")
        plt.savefig(plot_path, dpi=300)
        print(f"[УСПЕХ] График сохранен в {plot_path}")
        
    # Генерация интерактивной карты
    print("Генерация интерактивной карты (Folium)...")
    if len(kalman_trajectory) > 0:
        kalman_lats = [pt[0] for pt in kalman_trajectory]
        kalman_lons = [pt[1] for pt in kalman_trajectory]
        center_lat = sum(kalman_lats) / len(kalman_lats)
        center_lon = sum(kalman_lons) / len(kalman_lons)
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=18, tiles='OpenStreetMap')
        
        # Добавляем спутниковый слой Google
        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr='Google Satellite',
            name='Google Satellite',
            overlay=False
        ).add_to(m)
        
        # Рисуем сырую и сглаженную линии
        if len(raw_trajectory) > 0:
            folium.PolyLine(raw_trajectory, color='red', weight=2, opacity=0.4, dash_array='5, 5', tooltip="Raw LoFTR").add_to(m)
        folium.PolyLine(kalman_trajectory, color='blue', weight=4, opacity=0.8, tooltip="Kalman + Optical Flow").add_to(m)
        
        map_path = _make_unique_path("results/interactive_map_rigid.html")
        m.save(str(map_path))
        print(f"[УСПЕХ] Карта сохранена в {map_path}")

if __name__ == "__main__":
    video_path = pick_default_video()
    if video_path is None:
        print("Видео не найдено! Поместите .MP4 в data/videos или в корень проекта.")
    else:
        Path("results").mkdir(exist_ok=True)
        generate_final_video(str(video_path), "results/final_demo_hud_rigid.mp4", duration_sec=15, fps_out=15)
