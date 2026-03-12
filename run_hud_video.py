import cv2
import numpy as np
import logging
from tqdm import tqdm
from pathlib import Path

from src.video_processor import VideoProcessor
from src.map_loader import MapDownloader
from src.neural_matching import NeuralMatcher
from src.geolocator import Geolocator
from src.hud_drawer import HUDDrawer

logging.basicConfig(level=logging.WARNING)

def generate_final_video(video_path: str, output_path: str, duration_sec: int = 10, fps_out: int = 5):
    print(f"Генерация HUD видео: {output_path}")
    
    processor = VideoProcessor(video_path, default_geo=(55.086025, 38.149033, 750.0))
    video_fps = processor.info.fps
    total_frames = duration_sec * fps_out
    frame_step = int(video_fps / fps_out)
    
    # Загрузка карты
    pos_t0 = processor.telemetry.get_position_at_frame(0.0)
    map_loader = MapDownloader(zoom=16)
    map_img_pil, bbox = map_loader.get_basemap_for_location(pos_t0.latitude, pos_t0.longitude, radius_tiles=1)
    map_cv2 = cv2.cvtColor(np.array(map_img_pil), cv2.COLOR_RGB2BGR)
    
    matcher = NeuralMatcher()
    geolocator = Geolocator(bbox=bbox, map_shape=map_cv2.shape, smoothing_factor=0.2)
    
    # Узнаем размеры для сохранения видео
    first_frame = processor.extract_frame(0)
    if first_frame is None: return
        
    test_match = matcher.match(first_frame, map_cv2)
    debug_h, debug_w = test_match['debug_img'].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps_out, (debug_w, debug_h))
    
    print("Рендер кадров...")
    for i in tqdm(range(total_frames)):
        frame_idx = i * frame_step
        frame = processor.extract_frame(frame_idx)
        if frame is None: break
            
        # Мэтчинг
        match_result = matcher.match(frame, map_cv2)
        base_img = match_result['debug_img']
        
        # Геолокация
        mkpts_drone = match_result['mkpts0']
        mkpts_map = match_result['mkpts1']
        gps_coord = geolocator.estimate_center_gps(frame.shape, mkpts_drone, mkpts_map)
        
        # Рисуем HUD (Телеметрию поверх видео)
        final_img = HUDDrawer.draw_telemetry(base_img, gps_coord, len(mkpts_drone), is_ai=True)
        
        # Если размер съехал из-за четностей в PyTorch, выравниваем:
        if final_img.shape[:2] != (debug_h, debug_w):
            final_img = cv2.resize(final_img, (debug_w, debug_h))
            
        out.write(final_img)
        
    out.release()
    print(f"\n[УСПЕХ] Видео сохранено в {output_path}")

if __name__ == "__main__":
    video_files = list(Path('.').glob('*.MP4'))
    if video_files:
        Path("results").mkdir(exist_ok=True)
        generate_final_video(str(video_files[0]), "results/final_demo_hud.mp4")