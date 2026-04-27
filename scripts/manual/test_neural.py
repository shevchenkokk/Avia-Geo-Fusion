import cv2
import time
import numpy as np
from pathlib import Path
from src.video_processor import VideoProcessor
from src.map_loader import MapDownloader
from src.neural_matching import NeuralMatcher
from src.video_discovery import pick_default_video

def test_loftr():
    video_path = pick_default_video()
    if video_path is None:
        print("Видео не найдено! Поместите .MP4 в data/videos или в корень проекта.")
        return

    v_proc = VideoProcessor(str(video_path), default_geo=(55.086025, 38.149033, 750.0))
    frame_drone = v_proc.extract_frame(0)
    
    # Получаем карту (зум 16)
    map_loader = MapDownloader(zoom=16)
    pos_t0 = v_proc.telemetry.get_position_at_frame(0.0)
    map_img_pil, _ = map_loader.get_basemap_for_location(pos_t0.latitude, pos_t0.longitude, radius_tiles=1)
    map_cv2 = cv2.cvtColor(np.array(map_img_pil), cv2.COLOR_RGB2BGR)
    
    # Инициализация нейросети
    print("\n[AI] Инициализация LoFTR...")
    matcher = NeuralMatcher()
    
    print("\n[AI] Прогрев и поиск (первый запуск может занять несколько секунд)...")
    t0 = time.time()
    result = matcher.match(frame_drone, map_cv2)
    t1 = time.time()
    
    print(f"Ответ получен за {t1-t0:.2f} сек.")
    
    # Сохраним
    Path("results").mkdir(exist_ok=True)
    out_path = "results/loftr_matches.jpg"
    cv2.imwrite(out_path, result['debug_img'])
    print(f"[УСПЕХ] Результат LoFTR сохранен в {out_path}")

if __name__ == "__main__":
    test_loftr()