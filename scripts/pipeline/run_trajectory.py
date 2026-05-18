import cv2
import numpy as np
import logging
import folium
from tqdm import tqdm
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.video_processor import VideoProcessor
from src.map_loader import MapDownloader
from src.neural_matching import NeuralMatcher
from src.geolocator import Geolocator
from src.video_discovery import pick_default_video

logging.basicConfig(level=logging.WARNING)

def run_trajectory_tracking(video_path: str, duration_sec: int = 5, fps_out: int = 5):
    print(f"Запуск отслеживания траектории для {video_path}...")
    
    processor = VideoProcessor(video_path, default_geo=(55.086025, 38.149033, 750.0))
    video_fps = processor.info.fps
    total_frames = duration_sec * fps_out
    frame_step = int(video_fps / fps_out)
    
    # 1. Загрузка карты
    pos_t0 = processor.telemetry.get_position_at_frame(0.0)
    map_loader = MapDownloader(zoom=16)
    map_img_pil, bbox = map_loader.get_basemap_for_location(pos_t0.latitude, pos_t0.longitude, radius_tiles=1)
    map_cv2 = cv2.cvtColor(np.array(map_img_pil), cv2.COLOR_RGB2BGR)
    
    # 2. Инициализация сопоставителя и геолокатора
    matcher = NeuralMatcher()
    geolocator = Geolocator(bbox=bbox, map_shape=map_cv2.shape, smoothing_factor=0.2) # 0.2 - сильное сглаживание скачков
    
    # Подготовка визуала - будем рисовать трек прямо на карте
    output_map = map_cv2.copy()
    
    print("\n[matcher] Обработка кадров и расчет GPS...")
    for i in tqdm(range(total_frames)):
        frame_idx = i * frame_step
        frame = processor.extract_frame(frame_idx)
        if frame is None: break
            
        # Находим точки
        match_result = matcher.match(frame, map_cv2)
        mkpts_drone = match_result['mkpts0']
        mkpts_map = match_result['mkpts1']
        
        # Считаем GPS и сглаживаем
        gps_coord = geolocator.estimate_center_gps(frame.shape, mkpts_drone, mkpts_map)
        
        if gps_coord:
            lat, lon = gps_coord
            
            # Переведем обратно GPS в пиксели только для того, чтобы нарисовать точку на карте
            map_x = int((lon - geolocator.west) / (geolocator.east - geolocator.west) * geolocator.map_w)
            map_y = int((geolocator.north - lat) / (geolocator.north - geolocator.south) * geolocator.map_h)
            
            # Рисуем красный круг (где камера видит центр)
            cv2.circle(output_map, (map_x, map_y), 4, (0, 0, 255), -1)
            
            # Соединяем линией с прошлой точкой для отрисовки траектории
            if len(geolocator.trajectory_gps) > 1:
                prev_lat, prev_lon = geolocator.trajectory_gps[-2]
                prev_x = int((prev_lon - geolocator.west) / (geolocator.east - geolocator.west) * geolocator.map_w)
                prev_y = int((geolocator.north - prev_lat) / (geolocator.north - geolocator.south) * geolocator.map_h)
                cv2.line(output_map, (prev_x, prev_y), (map_x, map_y), (0, 255, 255), 2)
            
    # Сохраняем итоговую карту с траекторией (растровую)
    Path("results").mkdir(exist_ok=True)
    cv2.imwrite("results/flight_trajectory.jpg", output_map)
    print(f"\n[УСПЕХ] Растровая карта сохранена в results/flight_trajectory.jpg")
    
    # Генерация интерактивной карты Folium (Web GIS)
    if len(geolocator.trajectory_gps) > 0:
        start_coord = geolocator.trajectory_gps[0]
        f_map = folium.Map(location=start_coord, zoom_start=15)
        
        # Добавляем спутниковый слой Esri
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Esri Satellite',
            overlay=False,
            control=True
        ).add_to(f_map)
        
        # Отрисовка траектории
        folium.PolyLine(
            geolocator.trajectory_gps,
            color='red',
            weight=3,
            opacity=0.8,
            tooltip="Траектория БПЛА"
        ).add_to(f_map)
        
        # Маркер старта и конца
        folium.Marker(start_coord, popup="Начало сегмента", icon=folium.Icon(color="green")).add_to(f_map)
        folium.Marker(geolocator.trajectory_gps[-1], popup="Конец сегмента", icon=folium.Icon(color="red")).add_to(f_map)
        
        html_path = "results/interactive_map.html"
        f_map.save(html_path)
        print(f"[УСПЕХ] Интерактивная веб-карта сохранена в {html_path}")

if __name__ == "__main__":
    video_path = pick_default_video()
    if video_path is None:
        print("Видео не найдено! Поместите .MP4 в data/videos или в корень проекта.")
    else:
        run_trajectory_tracking(str(video_path), duration_sec=10, fps_out=3) # Берем 10 сек по 3 FPS
