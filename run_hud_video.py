import json
import matplotlib.pyplot as plt
import folium
import cv2
import numpy as np
import logging
from tqdm import tqdm
from pathlib import Path

from src.video_processor import VideoProcessor
from src.map_loader import MapDownloader
from src.map_manager import MapManager
from src.neural_matching import NeuralMatcher
from src.geolocator import Geolocator
from src.hud_drawer import HUDDrawer

logging.basicConfig(level=logging.WARNING)

def generate_final_video(video_path: str, output_path: str, duration_sec: int = 20, fps_out: int = 15):
    print(f"Генерация HQ HUD видео: {output_path}")
    
    processor = VideoProcessor(video_path, default_geo=(55.086025, 38.149033, 750.0))
    video_fps = processor.info.fps
    total_frames = duration_sec * fps_out
    frame_step = int(video_fps / fps_out)
    
    # Инициализация MapManager — скользящее окно карты вокруг стартовой позиции
    pos_t0 = processor.telemetry.get_position_at_frame(0.0)
    map_manager = MapManager(zoom=17, window_radius=3, closer_threshold=1)
    map_cv2, bbox = map_manager.initialize(pos_t0.latitude, pos_t0.longitude)
    
    matcher = NeuralMatcher()
    geolocator = Geolocator(bbox=bbox, map_shape=map_cv2.shape, smoothing_factor=0.2)
    
    # Узнаем размеры для сохранения видео
    first_frame = processor.extract_frame(0)
    if first_frame is None: return
        
    test_match = matcher.match(first_frame, map_cv2)
    debug_h, debug_w = test_match['debug_img'].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps_out, (debug_w, debug_h))
    
    print("Рендер кадров (Высокое качество, AI + Оптический поток)...")
    
    # --- Лукас-Канаде Трекинг Переменные ---
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    
    prev_gray_drone = None
    tracked_pts_drone = None
    tracked_pts_map = None
    keyframe_interval = fps_out  # Раз в 1 секунду пересчитываем нейронкой
    min_tracked_points = 20
    
    # Метрики для графиков
    kalman_trajectory = []
    raw_trajectory = []
    
    def render_side_by_side(drone_img, map_img, pts_drone, pts_map, max_points=100):
        """Рисует склеенное изображение с линиями, унифицировано для всех кадров"""
        if len(drone_img.shape) == 2: drone_img = cv2.cvtColor(drone_img, cv2.COLOR_GRAY2BGR)
        h1, w1 = drone_img.shape[:2]
        h2, w2 = map_img.shape[:2]
        out_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        out_img[:h1, :w1] = drone_img
        out_img[:h2, w1:w1+w2] = map_img
        
        # Обрезаем количество точек, чтобы линии не превращали экран в зеленую кашу
        if len(pts_drone) > max_points:
            np.random.seed(42) # Фиксируем сид, чтобы мерцание выборки было меньше
            idx = np.random.choice(len(pts_drone), max_points, replace=False)
            pts_drone_draw = np.array(pts_drone)[idx]
            pts_map_draw = np.array(pts_map)[idx]
        else:
            pts_drone_draw = pts_drone
            pts_map_draw = pts_map
            
        for p1, p2 in zip(pts_drone_draw, pts_map_draw):
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]) + w1, int(p2[1]))
            # Более красивые линии (Semi-transparent)
            cv2.line(out_img, pt1, pt2, (0, 255, 120), 1, cv2.LINE_AA)
            cv2.circle(out_img, pt1, 3, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(out_img, pt2, 3, (255, 0, 0), -1, cv2.LINE_AA)
        return out_img

    
    for i in tqdm(range(total_frames)):
        frame_idx = i * frame_step
        frame = processor.extract_frame(frame_idx)
        if frame is None: break
            
        drone_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        needs_keyframe = False
        if tracked_pts_drone is None or i % keyframe_interval == 0:
            needs_keyframe = True
        elif len(tracked_pts_drone) < min_tracked_points:
            needs_keyframe = True
            
        if needs_keyframe:
            # === Полный цикл (Neural Network: LoFTR) ===
            match_result = matcher.match(frame, map_cv2)
            
            mkpts_drone = match_result['mkpts0']
            mkpts_map = match_result['mkpts1']
            
            # Унифицированная отрисовка для Keyframe (без рывков цвета)
            base_img = render_side_by_side(frame, map_cv2, mkpts_drone, mkpts_map)
            
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
            
            # Отфильтровываем успешные точки (status == 1)
            good_new = p1[st == 1]
            good_map = tracked_pts_map[st == 1]
            
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
            
            # Визуализация 
            base_img = render_side_by_side(frame, map_cv2, mkpts_drone, mkpts_map)
            
            # Обновляем стейт на следующий кадр
            tracked_pts_drone = good_new.reshape(-1, 1, 2)
            tracked_pts_map = good_map.reshape(-1, 1, 2)
            prev_gray_drone = drone_gray.copy()
        
        # Геолокация
        gps_coord = geolocator.estimate_center_gps(frame.shape, mkpts_drone, mkpts_map)
        
        # -- СКОЛЬЗЯЩЕЕ ОКНО: обновляем карту если дрон приблизился к краю --
        if gps_coord is not None:
            new_map_cv2, new_bbox = map_manager.update(gps_coord[0], gps_coord[1])
            # Если MapManager сдвинул окно — обновляем карту и geolocator (без потери Калмана)
            if new_bbox != bbox:
                map_cv2 = new_map_cv2
                bbox = new_bbox
                geolocator.update_bbox(new_bbox, map_cv2.shape)
                # Сбрасываем трекинг точек, т.к. карта изменилась
                tracked_pts_drone = None
                tracked_pts_map = None
                print(f"\n[MapManager] Карта сдвинута → новый bbox: {new_bbox}")
        
        # Собираем метрики для графиков
        if gps_coord is not None and hasattr(geolocator, 'last_raw_gps'):
            kalman_trajectory.append(gps_coord)
            raw_trajectory.append(geolocator.last_raw_gps)
        
        # Рисуем HUD (Телеметрию поверх видео)
        final_img = HUDDrawer.draw_telemetry(base_img, gps_coord, len(mkpts_drone), is_ai=True)
        
        # Если размер съехал из-за четностей в PyTorch, выравниваем:
        if final_img.shape[:2] != (debug_h, debug_w):
            final_img = cv2.resize(final_img, (debug_w, debug_h))
            
        out.write(final_img)
        
    out.release()
    print(f"\n[УСПЕХ] Видео сохранено в {output_path}")
    
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
        
        plot_path = Path("results/trajectory_metrics.png")
        plt.savefig(plot_path, dpi=300)
        print(f"[УСПЕХ] График сохранен в {plot_path}")
        
    # Генерация интерактивной карты
    print("Генерация интерактивной карты (Folium)...")
    if len(kalman_trajectory) > 0:
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
        folium.PolyLine(raw_trajectory, color='red', weight=2, opacity=0.4, dash_array='5, 5', tooltip="Raw LoFTR").add_to(m)
        folium.PolyLine(kalman_trajectory, color='blue', weight=4, opacity=0.8, tooltip="Kalman + Optical Flow").add_to(m)
        
        map_path = "results/interactive_map_hq.html"
        m.save(map_path)
        print(f"[УСПЕХ] Карта сохранена в {map_path}")

if __name__ == "__main__":
    video_files = list(Path('.').glob('*.MP4'))
    if video_files:
        Path("results").mkdir(exist_ok=True)
        generate_final_video(str(video_files[0]), "results/final_demo_hud_hq.mp4", duration_sec=15, fps_out=15)