import sys
from pathlib import Path
from src.video_processor import VideoProcessor
from src.video_discovery import pick_default_video

def test_telemetry():
    video_path = pick_default_video()
    if video_path is None:
        print("Видеофайлы не найдены!")
        return

    video_file = str(video_path)
    print(f"Тестируем: {video_file}")
    
    # 1. Проверяем парсинг видео и фоллбэк телеметрии
    print("\n--- Инициализация VideoProcessor ---")
    default_geo = (55.086025, 38.149033, 750.0)
    processor = VideoProcessor(video_file, default_geo=default_geo)
    
    # 2. Проверяем инфо
    print(f"Разрешение: {processor.info.resolution}")
    print(f"FPS: {processor.info.fps}")
    print(f"Длительность (сек): {processor.info.duration_sec}")
    
    # 3. Извлекаем телеметрию для первого кадра (0 сек)
    print("\n--- Получение телеметрии для t=0 ---")
    pos_t0 = processor.telemetry.get_position_at_frame(0.0)
    print(f"t=0.0 -> Lat: {pos_t0.latitude}, Lon: {pos_t0.longitude}, Alt: {pos_t0.altitude}")
    
    # 4. Проверяем извлечение кадра
    print("\n--- Извлечение кадра ---")
    frame = processor.extract_frame(0)
    if frame is not None:
        print(f"Успешно извлечен кадр: {frame.shape}")
    else:
        print("Ошибка: не удалось извлечь кадр!")

if __name__ == '__main__':
    test_telemetry()
