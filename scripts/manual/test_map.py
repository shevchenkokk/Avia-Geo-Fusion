import sys
import logging
from pathlib import Path
from src.map_loader import MapDownloader

# Настроим логирование для наглядности
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_map_loader():
    # Стартовые координаты из конфига: 55.086025, 38.149033
    target_lat = 55.086025
    target_lon = 38.149033
    
    print(f"Запрашиваем карту для: Lat {target_lat}, Lon {target_lon}")
    
    # Инициализируем загрузчик (Google Satellite, zoom 17 чтобы не качать слишком много, радиус 2)
    # Зум 17 - хороший компромисс между детализацией и охватом. 
    # Радиус 2 = 5х5 тайлов = сетка 1280x1280 пикселей.
    downloader = MapDownloader(provider='google_satellite', zoom=17)
    
    # Скачиваем и склеиваем
    img, bbox = downloader.get_basemap_for_location(target_lat, target_lon, radius_tiles=2)
    
    print("\n--- Результат ---")
    print(f"Размер карты: {img.width}x{img.height} пикселей")
    print(f"Границы (West, South, East, North): {bbox}")
    
    # Сохраняем в файл, чтобы Вы могли открыть и посмотреть
    output_path = "results/test_basemap.jpg"
    Path("results").mkdir(exist_ok=True)
    img.save(output_path)
    print(f"Карта сохранена в: {output_path}")

if __name__ == '__main__':
    test_map_loader()
