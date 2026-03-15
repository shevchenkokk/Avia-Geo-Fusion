"""
Модуль для расчета физических GPS-координат (WGS84) из пиксельных совпадений 
между кадром с дрона и спутниковой картой, с учетом временного сглаживания (фильтрации скачков).
"""

import cv2
import logging
import numpy as np
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)

class Geolocator:
    def __init__(self, bbox: Tuple[float, float, float, float], map_shape: Tuple[int, int], smoothing_factor: float = 0.3):
        """
        :param bbox: Границы карты (west, south, east, north)
        :param map_shape: Разрешение карты (height, width)
        :param smoothing_factor: Коэффициент сглаживания (EMA). От 0.0 до 1.0. 
                                 Меньше значение = сильнее сглаживание (меньше дергается, но больше задержка).
        """
        self.west, self.south, self.east, self.north = bbox
        self.map_h, self.map_w = map_shape[:2]
        
        self.smoothing_factor = smoothing_factor
        
        # Состояние (память) для сглаживания
        self.last_gps: Optional[Tuple[float, float]] = None
        self.trajectory_gps: List[Tuple[float, float]] = []

        # -- KALMAN FILTER SETUP --
        # 4 состояния: [lat, lon, v_lat, v_lon]
        # 2 измерения: [lat, lon]
        self.kalman = cv2.KalmanFilter(4, 2)
        
        # Переходная матрица (x_t = x_{t-1} + v_{t-1})
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        
        # Матрица измерений
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        
        # Ковариации
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-5
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-4
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)
        
        self.is_kalman_inited = False

    def update_bbox(self, bbox: Tuple[float, float, float, float], map_shape: Tuple[int, int]):
        """
        Обновляет географические границы карты при сдвиге окна MapManager.
        Состояние Фильтра Калмана и история траектории при этом НЕ сбрасываются.
        """
        self.west, self.south, self.east, self.north = bbox
        self.map_h, self.map_w = map_shape[:2]
        logger.debug(f"[Geolocator] bbox обновлён: {bbox}")

    def pixel_to_gps(self, x: float, y: float) -> Tuple[float, float]:
        """
        Переводит пиксели (x, y) на скачанной карте в (Latitude, Longitude).
        """
        # Долгота: линейно от запада к востоку
        lon = self.west + (self.east - self.west) * (x / self.map_w)
        # Широта: y=0 это север, y=map_h это юг
        lat = self.north - (self.north - self.south) * (y / self.map_h)
        return lat, lon

    def estimate_center_gps(self, frame_shape: Tuple[int, int], mkpts_drone: np.ndarray, mkpts_map: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Рассчитывает GPS-координаты, в которую смотрит центр камеры БПЛА.
        
        :param frame_shape: (height, width) кадра с дрона
        :param mkpts_drone: Пиксели на кадре
        :param mkpts_map: Пиксели на карте
        :return: (lat, lon) сглаженные
        """
        if len(mkpts_drone) < 4:
            return None # Недостаточно точек для гомографии
            
        # 1. Считаем матрицу гомографии (преобразования перспективы)
        H, mask = cv2.findHomography(mkpts_drone, mkpts_map, cv2.RANSAC, 5.0)
        
        if H is None:
            return None
            
        frame_h, frame_w = frame_shape[:2]
        
        # 2. Берем центральную точку кадра БПЛА
        center_pt = np.array([[[frame_w / 2.0, frame_h / 2.0]]], dtype=np.float32)
        
        # 3. Проецируем центр кадра на спутниковую карту (где этот центр находится на карте)
        try:
            center_map_pt = cv2.perspectiveTransform(center_pt, H)
        except Exception:
            return None
            
        map_x, map_y = center_map_pt[0][0]
        
        # Проверка, что точка не улетела вообще за пределы карты
        if map_x < -self.map_w or map_x > self.map_w * 2 or map_y < -self.map_h or map_y > self.map_h * 2:
            return None
            
        # 4. Переводим пиксель карты в GPS
        raw_lat, raw_lon = self.pixel_to_gps(map_x, map_y)
        self.last_raw_gps = (raw_lat, raw_lon) # Сохраняем сырые данные для метрик
        
        # 5. Инициализация и применение Фильтра Калмана
        if not self.is_kalman_inited:
            # Инициализация состояния (x, y, dx=0, dy=0)
            self.kalman.statePre = np.array([[raw_lat], [raw_lon], [0.0], [0.0]], np.float32)
            self.kalman.statePost = np.array([[raw_lat], [raw_lon], [0.0], [0.0]], np.float32)
            self.is_kalman_inited = True
            kalman_lat, kalman_lon = raw_lat, raw_lon
        else:
            # Шаг 1: Предсказание следующей позиции на основе скорости
            self.kalman.predict()
            
            # Шаг 2: Коррекция предсказания на основе новых "шумных" измерений LoFTR
            measurement = np.array([[np.float32(raw_lat)], [np.float32(raw_lon)]])
            estimated = self.kalman.correct(measurement)
            kalman_lat, kalman_lon = float(estimated[0]), float(estimated[1])
            
        # 6. Вторичное сглаживание скачков (Exponential Moving Average) после Калмана для плавности графики
        if self.last_gps is None:
            smoothed_lat, smoothed_lon = kalman_lat, kalman_lon
        else:
            smoothed_lat = self.last_gps[0] * (1 - self.smoothing_factor) + kalman_lat * self.smoothing_factor
            smoothed_lon = self.last_gps[1] * (1 - self.smoothing_factor) + kalman_lon * self.smoothing_factor
            
        self.last_gps = (smoothed_lat, smoothed_lon)
        self.trajectory_gps.append(self.last_gps)
        
        return self.last_gps
