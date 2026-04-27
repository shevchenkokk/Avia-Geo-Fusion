import cv2
import logging
import numpy as np
from pathlib import Path
from src.video_processor import VideoProcessor
from src.map_loader import MapDownloader
from src.neural_matching import NeuralMatcher
from src.video_discovery import pick_default_video

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def match_and_save(frame_drone, map_cv2, backend, out_path, lg_features="auto"):
    print(f"\n--- Запуск сопоставления фичей ({backend.upper()}) ---")
    if backend == 'lightglue':
        matcher = NeuralMatcher(backend=backend, lightglue_features=lg_features, use_segmentation=False)
    else:
        matcher = NeuralMatcher(backend=backend, use_segmentation=False)
        
    result = matcher.match(frame_drone, map_cv2)
    
    if result['debug_img'] is not None:
        cv2.imwrite(out_path, result['debug_img'])
        print(f"Результат сопоставления сохранен в: {out_path}")
    else:
        print(f"Ошибка мэтчинга {backend.upper()}!")

def build_pics():
    video_path = pick_default_video()
    if video_path is None:
        print("Видео не найдено! Поместите .MP4 в data/videos или в корень проекта.")
        return
        
    default_geo = (55.086025, 38.149033, 750.0)
    v_proc = VideoProcessor(str(video_path), default_geo=default_geo)
    frame_drone = v_proc.extract_frame(0)
    
    if frame_drone is None:
        print("Не удалось извлечь кадр.")
        return
        
    pos_t0 = v_proc.telemetry.get_position_at_frame(0.0)
    
    print("\n--- Загрузка спутниковой подложки ---")
    map_loader = MapDownloader(zoom=17) # зум 17!
    map_img_pil, bbox = map_loader.get_basemap_for_location(
        lat=pos_t0.latitude, 
        lon=pos_t0.longitude, 
        radius_tiles=1
    )
    
    map_cv2 = cv2.cvtColor(np.array(map_img_pil), cv2.COLOR_RGB2BGR)
    
    # Сделаем высоту map_cv2 равной высоте frame_drone
    h_drone, w_drone = frame_drone.shape[:2]
    h_map, w_map = map_cv2.shape[:2]
    
    if h_drone != h_map:
        scale = h_drone / float(h_map)
        new_w = int(w_map * scale)
        map_cv2 = cv2.resize(map_cv2, (new_w, h_drone))

    Path("results").mkdir(exist_ok=True)
    match_and_save(frame_drone, map_cv2, 'loftr', "results/loftr_matches_v2.jpg")
    match_and_save(frame_drone, map_cv2, 'lightglue', "results/lightglue_matches_v2.jpg")

if __name__ == '__main__':
    build_pics()
