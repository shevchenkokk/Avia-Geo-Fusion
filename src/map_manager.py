"""
Модуль управления динамической картой — MapManager.

Реализует концепцию «скользящего окна» (Sliding Window Map), описанную в работах
по навигации самолёта (Chen et al., Yao et al., MMVL):

    - Память хранит только тайлы вокруг текущей позиции самолета (±window_radius тайлов).
    - Когда самолет приближается к границе окна (closer_threshold тайлов до края),
      центр окна сдвигается к новой позиции, новые тайлы подгружаются,
      а вышедшие из зоны видимости тайлы выгружаются (eviction) из кэша.
    - Весь глобальный набор тайлов (вся страна) никогда не загружается в RAM.

Это позволяет работать на маршрутах произвольной длины, ограничивая потребление
памяти константой (window_radius^2 * 256 * 256 * 3 байт ≈ ~50 МБ для radius=3).
"""

import cv2
import logging
import mercantile
import numpy as np
from PIL import Image
from typing import Optional, Tuple

from src.map_loader import MapDownloader

logger = logging.getLogger(__name__)


class MapManager:
    """
    Менеджер карты со скользящим окном.
    Принимает обновление GPS-позиции самолета и автоматически поддерживает
    актуальную склеенную спутниковую карту в памяти.
    """

    def __init__(
        self,
        zoom: int = 17,
        window_radius: int = 3,
        closer_threshold: int = 1,
        loader: Optional[MapDownloader] = None,
    ):
        """
        :param zoom:              Уровень зума спутниковых тайлов (16-18).
        :param window_radius:     Радиус окна в тайлах вокруг центра. radius=3 → сетка 7×7 тайлов.
        :param closer_threshold:  При каком расстоянии (в тайлах) от края окна запускать сдвиг.
        :param loader:            Экземпляр MapDownloader (создаётся автоматически если None).
        """
        self.zoom = zoom
        self.window_radius = window_radius
        self.closer_threshold = closer_threshold
        self.loader = loader or MapDownloader(zoom=zoom)

        # Координаты центра текущего окна в тайловом пространстве
        self.center_tx: Optional[int] = None
        self.center_ty: Optional[int] = None

        # LRU-кэш тайлов: {(tile_x, tile_y): PIL.Image}
        self._tile_cache: dict = {}

        # Текущая склеенная карта (cv2 BGR) и её географические границы
        self.current_map_cv2: Optional[np.ndarray] = None
        self.current_bbox: Optional[Tuple[float, float, float, float]] = None

        self._initialized = False

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def initialize(self, lat: float, lon: float) -> Tuple[np.ndarray, Tuple]:
        """
        Первоначальная загрузка окна карты вокруг стартовой позиции самолета.
        Вызывать один раз до входа в основной цикл обработки видео.
        """
        center = mercantile.tile(lon, lat, self.zoom)
        self.center_tx = center.x
        self.center_ty = center.y
        logger.info(f"[MapManager] Инициализация окна {2*self.window_radius+1}×{2*self.window_radius+1} тайлов"
                    f" вокруг ({lat:.5f}, {lon:.5f}) → tile({center.x}, {center.y})")
        self._load_window()
        self._stitch()
        self._initialized = True
        return self.current_map_cv2, self.current_bbox

    def update(self, lat: float, lon: float) -> Tuple[np.ndarray, Tuple]:
        """
        Обновление карты по текущей GPS-оценке самолета (из Фильтра Калмана).
        Если самолет приближается к краю окна — сдвигаем центр и подгружаем новые тайлы.
        Вызывать каждый кадр / раз в n кадров.

        :return: (current_map_cv2, current_bbox) — актуальная карта и её bbox
        """
        if not self._initialized:
            return self.initialize(lat, lon)

        plane_tile = mercantile.tile(lon, lat, self.zoom)
        dx = abs(plane_tile.x - self.center_tx)
        dy = abs(plane_tile.y - self.center_ty)

        # Порог сдвига: если самолет оказался ближе к краю чем closer_threshold тайлов
        trigger = self.window_radius - self.closer_threshold
        if dx > trigger or dy > trigger:
            logger.info(
                f"[MapManager] Самолет у края окна (dx={dx}, dy={dy}) → сдвиг центра "
                f"к tile({plane_tile.x}, {plane_tile.y})"
            )
            self.center_tx = plane_tile.x
            self.center_ty = plane_tile.y
            self._load_window()
            self._stitch()

        return self.current_map_cv2, self.current_bbox

    @property
    def window_size_px(self) -> Tuple[int, int]:
        """Возвращает размер текущей карты в пикселях (w, h)."""
        n = 2 * self.window_radius + 1
        return n * 256, n * 256

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    def _load_window(self):
        """Загружает недостающие тайлы и выгружает ненужные из кэша."""
        r = self.window_radius
        needed = {
            (self.center_tx + dx, self.center_ty + dy)
            for dx in range(-r, r + 1)
            for dy in range(-r, r + 1)
        }

        # 1. Evict (выгрузить) тайлы вышедшие за пределы нового окна
        evicted = 0
        for key in list(self._tile_cache.keys()):
            if key not in needed:
                del self._tile_cache[key]
                evicted += 1

        # 2. Собираем список только тех тайлов, которых нет в кэше
        tiles_to_download = []
        for (tx, ty) in needed:
            if (tx, ty) not in self._tile_cache:
                tiles_to_download.append((tx, ty, self.zoom))

        # 3. Загружаем отсутствующие новые тайлы
        loaded = 0
        if tiles_to_download:
            logger.info(f"[MapManager] Требуется загрузить {len(tiles_to_download)} новых тайлов.")
            # Параллельная загрузка тайлов
            downloaded_tiles = self.loader.download_tiles_parallel(tiles_to_download)
            # 4. Добавляем скачанное в кэш
            for (tx, ty, z), img in downloaded_tiles.items():
                self._tile_cache[(tx, ty)] = img
                loaded += 1

        logger.info(f"[MapManager] Загружено новых тайлов: {loaded}, выгружено: {evicted},"
                    f" в кэше: {len(self._tile_cache)}")

    def _stitch(self):
        """Склеивает тайлы из кэша в единое изображение и пересчитывает bbox."""
        r = self.window_radius
        min_x = self.center_tx - r
        max_x = self.center_tx + r
        min_y = self.center_ty - r
        max_y = self.center_ty + r

        n = 2 * r + 1
        merged = Image.new('RGB', (n * 256, n * 256), color=(128, 128, 128))

        for tx in range(min_x, max_x + 1):
            for ty in range(min_y, max_y + 1):
                tile_img = self._tile_cache.get((tx, ty))
                if tile_img is not None:
                    paste_x = (tx - min_x) * 256
                    paste_y = (ty - min_y) * 256
                    merged.paste(tile_img, (paste_x, paste_y))

        # Географические границы склеенного изображения
        tl = mercantile.bounds(mercantile.Tile(x=min_x, y=min_y, z=self.zoom))
        br = mercantile.bounds(mercantile.Tile(x=max_x, y=max_y, z=self.zoom))
        self.current_bbox = (tl.west, br.south, br.east, tl.north)

        self.current_map_cv2 = cv2.cvtColor(np.array(merged), cv2.COLOR_RGB2BGR)
        logger.info(f"[MapManager] Карта пересшита. Bbox: {self.current_bbox}")
