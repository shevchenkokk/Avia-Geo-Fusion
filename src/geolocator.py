"""
Модуль для расчета физических GPS-координат (WGS84) из пиксельных совпадений 
между кадром с дрона и спутниковой картой, с учетом временного сглаживания (фильтрации скачков).
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional

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
        
        # 5. Временное сглаживание скачков (Exponential Moving Average)
        if self.last_gps is None:
            smoothed_lat, smoothed_lon = raw_lat, raw_lon
        else:
            smoothed_lat = self.last_gps[0] * (1 - self.smoothing_factor) + raw_lat * self.smoothing_factor
            smoothed_lon = self.last_gps[1] * (1 - self.smoothing_factor) + raw_lon * self.smoothing_factor
            
        self.last_gps = (smoothed_lat, smoothed_lon)
        self.trajectory_gps.append(self.last_gps)
        
        return self.last_gps
