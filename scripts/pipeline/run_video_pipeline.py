import cv2
import numpy as np
import logging
from tqdm import tqdm
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.video_processor import VideoProcessor
from src.map_loader import MapDownloader
from src.feature_matching import FeatureMatcher
from src.video_discovery import pick_default_video

logging.basicConfig(level=logging.INFO)

def process_video_segment(video_path: str, output_path: str, duration_sec: int = 5, fps_out: int = 2):
    """
    Обрабатывает короткий сегмент видео, сопоставляя его с картой, и сохраняет результат в новое видео.
    
    :param video_path: Путь к исходному видео.
    :param output_path: Путь для сохранения результата (mp4).
    :param duration_sec: Сколько секунд видео обработать.
    :param fps_out: С какой частотой (кадров в секунду) брать кадры из исходника.
    """
    processor = VideoProcessor(video_path, default_geo=(55.086025, 38.149033, 750.0))
    video_fps = processor.info.fps
    total_frames_to_process = duration_sec * fps_out
    frame_step = int(video_fps / fps_out)
    
    # 1. Сразу скачиваем карту для стартовой точки (предполагаем, что за 5 сек дрон далеко не улетел)
    print("Загрузка опорной карты для сегмента...")
    pos_t0 = processor.telemetry.get_position_at_frame(0.0)
    map_loader = MapDownloader(zoom=16)
    map_img_pil, _ = map_loader.get_basemap_for_location(pos_t0.latitude, pos_t0.longitude, radius_tiles=1)
    map_cv2 = cv2.cvtColor(np.array(map_img_pil), cv2.COLOR_RGB2BGR)
    
    # Мэтчер выбираем на основе аргументов (в данном случае используем LoFTR)
    print("Инициализация NeuralMatcher (LoFTR)...")
    from src.neural_matching import NeuralMatcher
    matcher = NeuralMatcher()
    
    # Настраиваем VideoWriter (узнаем размер по первому кадру)
    first_frame = processor.extract_frame(0)
    if first_frame is None:
        print("Ошибка чтения видео")
        return
        
    test_match = matcher.match(first_frame, map_cv2)
    debug_h, debug_w = test_match['debug_img'].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps_out, (debug_w, debug_h))
    
    print(f"Начинаем рендер видео: {output_path} (длина {duration_sec} сек, {fps_out} FPS)")
    
    for i in tqdm(range(total_frames_to_process)):
        frame_idx = i * frame_step
        frame = processor.extract_frame(frame_idx)
        
        if frame is None:
            break
            
        match_result = matcher.match(frame, map_cv2)
        debug_img = match_result['debug_img']
        
        # Если размер не совпадает (бывает редко, но на всякий случай)
        if debug_img.shape[:2] != (debug_h, debug_w):
            debug_img = cv2.resize(debug_img, (debug_w, debug_h))
            
        out.write(debug_img)
        
    out.release()
    print(f"Видео успешно сохранено в {output_path}!")

if __name__ == "__main__":
    video_path = pick_default_video()
    if video_path is None:
        print("Видео не найдено! Поместите .MP4 в data/videos или в корень проекта.")
    else:
        Path("results").mkdir(exist_ok=True)
        # Обработаем 5 первых секунд видео со скоростью 5 кадров в секунду (всего 25 кадров)
        process_video_segment(str(video_path), "results/demo_loftr_matching.mp4", duration_sec=5, fps_out=5)
