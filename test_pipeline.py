import cv2
import logging
from pathlib import Path
from src.video_processor import VideoProcessor
from src.map_loader import MapDownloader
from src.feature_matching import FeatureMatcher

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_pipeline():
    # 1. Загружаем кадр с БПЛА
    video_files = list(Path('.').glob('*.MP4'))
    if not video_files:
        print("Видео не найдено!")
        return
        
    default_geo = (55.086025, 38.149033, 750.0)
    v_proc = VideoProcessor(str(video_files[0]), default_geo=default_geo)
    frame_drone = v_proc.extract_frame(0) # Берем самый ПЕРВЫЙ кадр
    
    if frame_drone is None:
        print("Не удалось извлечь кадр.")
        return
        
    # CV2 загружает в BGR, переведем в RGB для совместимости с картой
    # frame_drone_rgb = cv2.cvtColor(frame_drone, cv2.COLOR_BGR2RGB)
    
    # 2. Получаем координаты (фоллбэк или телеметрия) и грузим карту
    pos_t0 = v_proc.telemetry.get_position_at_frame(0.0)
    
    print("\n--- Загрузка спутниковой подложки ---")
    map_loader = MapDownloader(zoom=16) # зум 16, чтобы объектов было побольше
    map_img_pil, bbox = map_loader.get_basemap_for_location(
        lat=pos_t0.latitude, 
        lon=pos_t0.longitude, 
        radius_tiles=1  # Сетка 3х3 тайла для скорости
    )
    
    # Конвертируем PIL Image (RGB) в формат OpenCV (BGR)
    import numpy as np
    map_cv2 = cv2.cvtColor(np.array(map_img_pil), cv2.COLOR_RGB2BGR)
    
    # 3. Мэтчинг (baseline: SIFT)
    print("\n--- Запуск сопоставления фичей (SIFT) ---")
    matcher = FeatureMatcher(method='sift')
    result = matcher.match(frame_drone, map_cv2)
    
    # Сохраним результат
    Path("results").mkdir(exist_ok=True)
    out_path = "results/sift_matches_baseline.jpg"
    
    if result['debug_img'] is not None:
        cv2.imwrite(out_path, result['debug_img'])
        print(f"Результат сопоставления сохранен в: {out_path}")
    else:
        print("Ошибка мэтчинга!")

if __name__ == '__main__':
    test_pipeline()