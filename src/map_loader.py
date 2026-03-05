"""
Модуль для работы со спутниковыми картами (Basemap).
Отвечает за скачивание тайлов спутниковых снимков для заданных координат, 
их сшивку в единое ортофото и расчет его географических границ.
"""

import os
import requests
import math
import logging
import mercantile
from PIL import Image
from io import BytesIO
from typing import Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Провайдеры спутниковых карт (XYZ tiles)
PROVIDERS = {
    'google_satellite': "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    'esri_satellite': "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
}

class MapDownloader:
    """Загрузчик и сшиватель спутниковых тайлов."""

    def __init__(self, provider: str = 'google_satellite', zoom: int = 18, cache_dir: str = './data/basemaps'):
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Available: {list(PROVIDERS.keys())}")
        
        self.provider_url = PROVIDERS[provider]
        self.zoom = zoom
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Заголовок, чтобы не блокировали по User-Agent
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

    def download_tile(self, x: int, y: int, z: int) -> Optional[Image.Image]:
        """Скачивает один тайл (256x256)"""
        url = self.provider_url.format(x=x, y=y, z=z)
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            logger.error(f"Ошибка загрузки тайла ({x}, {y}, {z}): {e}")
            # Возвращаем черный квадрат в случае ошибки
            return Image.new('RGB', (256, 256), color='black')

    def get_basemap_for_location(self, lat: float, lon: float, radius_tiles: int = 2) -> Tuple[Image.Image, Tuple[float, float, float, float]]:
        """
        Скачивает блок тайлов вокруг заданной точки и склеивает их.
        
        :param lat: Широта центра
        :param lon: Долгота центра
        :param radius_tiles: Радиус загрузки в тайлах вокруг центра (2 = сетка 5x5 тайлов)
        :return: (Картинка, (west, south, east, north) bounding box)
        """
        # 1. Находим центральный тайл для (lat, lon) на текущем zoom
        center_tile = mercantile.tile(lon, lat, self.zoom)
        logger.info(f"Центральный тайл: {center_tile}")

        # 2. Определяем сетку тайлов
        min_x = center_tile.x - radius_tiles
        max_x = center_tile.x + radius_tiles
        min_y = center_tile.y - radius_tiles
        max_y = center_tile.y + radius_tiles

        width_tiles = max_x - min_x + 1
        height_tiles = max_y - min_y + 1

        pixel_width = width_tiles * 256
        pixel_height = height_tiles * 256

        # Итоговое изображение
        merged_image = Image.new('RGB', (pixel_width, pixel_height))

        # 3. Скачиваем и вставляем
        logger.info(f"Начинаю загрузку {width_tiles * height_tiles} тайлов...")
        for curr_x in range(min_x, max_x + 1):
            for curr_y in range(min_y, max_y + 1):
                tile_img = self.download_tile(curr_x, curr_y, self.zoom)
                if tile_img:
                    paste_x = (curr_x - min_x) * 256
                    paste_y = (curr_y - min_y) * 256
                    merged_image.paste(tile_img, (paste_x, paste_y))

        # 4. Вычисляем Bounding Box (левый верхний и правый нижний угол)
        # mercantile.bounds возвращает (west, south, east, north)
        top_left_bounds = mercantile.bounds(mercantile.Tile(x=min_x, y=min_y, z=self.zoom))
        bottom_right_bounds = mercantile.bounds(mercantile.Tile(x=max_x, y=max_y, z=self.zoom))

        west = top_left_bounds.west
        north = top_left_bounds.north
        east = bottom_right_bounds.east
        south = bottom_right_bounds.south

        bbox = (west, south, east, north)
        
        logger.info(f"Карта успешно склеена. Размер: {pixel_width}x{pixel_height}. Bounding Box: {bbox}")
        return merged_image, bbox
