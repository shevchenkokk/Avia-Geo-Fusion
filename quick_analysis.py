#!/usr/bin/env python3
"""
Быстрый скрипт для анализа видео GoPro

Использование:
    python quick_analysis.py [видеофайл]
    
Пример:
    python quick_analysis.py GOPR0269.MP4
"""

import sys
from pathlib import Path
from src.video_processor import VideoProcessor, ImageSegmenter
import json

# Параметры проекта
PROJECT_CONFIG = {
    'start_lat': 55.086025,
    'start_lon': 38.149033,
    'start_height': 750
}

def main():
    # Загружаем конфигурацию проекта
    config_path = Path('config.json')
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"📋 Проект: {config.get('project_name', 'Unknown')}")
        print(f"   {config.get('project_description', '')}\n")
    
    # Получаем видеофайл для анализа
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
    else:
        # Используем первый найденный видеофайл
        video_files = list(Path('.').glob('*.MP4'))
        if not video_files:
            print("❌ Видеофайлы не найдены!")
            print("   Поместите файлы .MP4 в корневую папку проекта")
            sys.exit(1)
        video_file = str(video_files[0])
    
    print(f"🎬 Анализ видеофайла: {video_file}\n")
    
    try:
        # Инициализируем процессор видео
        processor = VideoProcessor(video_file)
        
        # Выводим информацию о видео
        print("=" * 60)
        print("📹 Информация о видео:")
        print("=" * 60)
        print(f"  Файл: {processor.info.filename}")
        print(f"  Разрешение: {processor.info.resolution}")
        print(f"  FPS: {processor.info.fps:.2f}")
        print(f"  Всего кадров: {processor.info.frame_count}")
        print(f"  Длительность: {processor.info.duration_min:.2f} мин ({processor.info.duration_sec:.1f} сек)")
        
        # Извлекаем метаданные
        print("\n" + "=" * 60)
        print("🏷️  Метаданные видео:")
        print("=" * 60)
        metadata = processor.extract_metadata()
        
        if metadata:
            if 'format' in metadata and 'tags' in metadata['format']:
                tags = metadata['format']['tags']
                print("  Теги:")
                for key, value in list(tags.items())[:10]:  # Показываем первые 10
                    if value:
                        print(f"    {key}: {value}")
            
            # Проверяем наличие GPS данных
            if 'streams' in metadata:
                for stream in metadata['streams']:
                    if 'side_data_list' in stream:
                        print(f"\n  Side Data найдены: {len(stream['side_data_list'])} записей")
                        for side_data in stream['side_data_list'][:5]:
                            print(f"    - {side_data.get('side_data_type', 'Unknown')}")
        else:
            print("  ⚠️  Метаданные не извлечены")
        
        # Извлекаем и анализируем первый кадр
        print("\n" + "=" * 60)
        print("🖼️  Анализ первого кадра:")
        print("=" * 60)
        first_frame = processor.extract_frame(0)
        
        if first_frame is not None:
            print(f"  Размер кадра: {first_frame.shape}")
            
            # Сегментация неба/земли
            segmenter = ImageSegmenter()
            sky_mask, ground_mask = segmenter.segment_sky_ground(first_frame)
            
            # Анализ объектов
            objects, mask_cleaned, contours = segmenter.analyze_objects(
                first_frame, ground_mask, min_area=100
            )
            
            print(f"  Обнаружено объектов: {len(objects)}")
            print(f"  Найдено контуров: {len(contours)}")
            
            if objects:
                print("\n  Первые 5 объектов:")
                for i, obj in enumerate(objects[:5], 1):
                    print(f"    {i}. Позиция: ({obj['x']}, {obj['y']}), "
                          f"Размер: {obj['width']}x{obj['height']}, "
                          f"Площадь: {obj['area']:.0f} px²")
        else:
            print("  ❌ Не удается извлечь первый кадр")
        
        # Геореференцирование
        print("\n" + "=" * 60)
        print("🗺️  Геореференцирование:")
        print("=" * 60)
        frames_df = processor.georeference_frames(
            start_lat=PROJECT_CONFIG['start_lat'],
            start_lon=PROJECT_CONFIG['start_lon'],
            start_height=PROJECT_CONFIG['start_height'],
            sample_rate=30
        )
        
        print(f"  Начальные координаты: {PROJECT_CONFIG['start_lat']}, {PROJECT_CONFIG['start_lon']}")
        print(f"  Высота при отцепе: {PROJECT_CONFIG['start_height']} м")
        print(f"  Обработано кадров: {len(frames_df)}")
        print(f"  Диапазон времени: {frames_df['time_sec'].min():.2f} - {frames_df['time_sec'].max():.2f} сек")
        
        print("\n✓ Анализ завершен успешно!")
        print(f"\n💡 Для подробного анализа откройте Jupyter notebook:")
        print(f"   jupyter notebook notebooks/video_analysis.ipynb")
        
    except Exception as e:
        print(f"\n❌ Ошибка при анализе: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
