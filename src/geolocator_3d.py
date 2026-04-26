"""
Модуль расчета полной 6-DoF (6 степеней свободы) позы камеры в пространстве.
Использует классический алгоритм Perspective-n-Point (PnP) для определения 
координат (Lat, Lon) и высоты (Altitude) самолета относительно земли.

В отличие от 2D-гомографии, этот метод требует знания внутренних параметров камеры,
но позволяет оценивать высоту полета без использования внешних датчиков (барометр/GPS).
"""

import cv2
import numpy as np
import logging
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)

class Geolocator3D:
    def __init__(
        self,
        bbox: Tuple[float, float, float, float],
        map_shape: Tuple[int, int],
        smoothing_factor: float = 0.3,
        fov_degrees: float = 90.0
    ):
        """
        :param bbox: (west, south, east, north) - географические границы карты.
        :param map_shape: (height, width) - разрешение карты в пикселях.
        :param smoothing_factor: Коэффициент EMA-сглаживания траектории.
        :param fov_degrees: Угол обзора камеры по горизонтали в градусах. 
                            Критически важный параметр для точности высоты.
        """
        self.west, self.south, self.east, self.north = bbox
        self.map_h, self.map_w = map_shape[:2]
        
        # EMA Smoothing
        self.smoothing_factor = smoothing_factor
        self.last_gps: Optional[Tuple[float, float]] = None
        self.trajectory_gps: List[Tuple[float, float]] = []
        self.last_raw_gps: Optional[Tuple[float, float]] = None
        self.fov_degrees = fov_degrees

        # Кэшируем масштаб, чтобы не считать каждый раз
        self._cached_mpp: Optional[Tuple[float, float]] = None

    def update_bbox(self, bbox: Tuple[float, float, float, float], map_shape: Tuple[int, int]):
        """Обновляет географические границы и сбрасывает кэш масштаба."""
        self.west, self.south, self.east, self.north = bbox
        self.map_h, self.map_w = map_shape[:2]
        self._cached_mpp = None
        
    def _get_meters_per_pixel(self) -> Tuple[float, float]:
        """Расчет и кэширование масштаба карты в метрах на пиксель."""
        if self._cached_mpp:
            return self._cached_mpp

        center_lat = (self.north + self.south) / 2.0
        # Длина дуги 1 градуса долготы зависит от широты: cos(lat) * 111320 м
        lon_dist_m = (self.east - self.west) * np.cos(np.radians(center_lat)) * 111320.0
        # Длина дуги 1 градуса широты примерно постоянна: 111000 м
        lat_dist_m = (self.north - self.south) * 111000.0
        
        m_per_px_x = lon_dist_m / self.map_w
        m_per_px_y = lat_dist_m / self.map_h
        self._cached_mpp = (m_per_px_x, m_per_px_y)
        return self._cached_mpp

    def _build_camera_matrix(self, frame_shape: Tuple[int, int]) -> np.ndarray:
        """Собирает матрицу внутренних параметров камеры (K) на основе FOV."""
        h, w = frame_shape[:2]
        fov_rad = np.radians(self.fov_degrees)
        # Убедимся, что fov_rad не равен 0 или Pi, чтобы избежать деления на ноль
        if np.tan(fov_rad / 2.0) == 0:
            fx = float('inf')
        else:
            fx = (w / 2.0) / np.tan(fov_rad / 2.0)
        fy = fx
        cx, cy = w / 2.0, h / 2.0
        
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    def _map_pixels_to_local_3d_vectorized(self, mkpts_map: np.ndarray) -> np.ndarray:
        """
        Векторизованное преобразование пиксельных координат карты в локальную 3D-систему (метры).
        Z-координата для всех точек устанавливается в 0 (уровень земли).
        """
        m_per_px_x, m_per_px_y = self._get_meters_per_pixel()
        # Создаем пустой массив (N, 3) для XYZ
        obj_points_3d = np.zeros((mkpts_map.shape[0], 3), dtype=np.float32)
        # Рассчитываем X и Y сразу для всех точек
        obj_points_3d[:, 0] = mkpts_map[:, 0] * m_per_px_x  # X = x_px * scale_x
        obj_points_3d[:, 1] = -mkpts_map[:, 1] * m_per_px_y # Y = -y_px * scale_y
        return obj_points_3d
        
    def _local_3d_to_gps(self, X: float, Y: float) -> Tuple[float, float]:
        """Обратный перевод из локальных 3D метров в GPS."""
        m_per_px_x, m_per_px_y = self._get_meters_per_pixel()
        x_px = X / m_per_px_x
        y_px = -Y / m_per_px_y
        
        lon = self.west + (self.east - self.west) * (x_px / self.map_w)
        lat = self.north - (self.north - self.south) * (y_px / self.map_h)
        return lat, lon

    def estimate_plane_position(
        self,
        frame_shape: Tuple[int, int],
        mkpts_plane: np.ndarray,
        mkpts_map: np.ndarray
    ) -> Optional[Tuple[float, float, float]]:
        """
        Рассчитывает 3D позицию и ориентацию камеры (6-DoF) через solvePnP.
        
        :param frame_shape: (height, width) кадра с самолета.
        :param mkpts_plane: (N, 2) массив 2D-точек на кадре.
        :param mkpts_map: (N, 2) массив соответствующих 2D-точек на карте.
        :return: (Latitude, Longitude, Altitude_meters) или None при ошибке.
        """
        if len(mkpts_plane) < 4:
            return None

        # 1. Преобразование 2D точек карты в 3D точки на плоскости Z=0
        obj_points_3d = self._map_pixels_to_local_3d_vectorized(mkpts_map)
        img_points_2d = mkpts_plane.astype(np.float32)
        
        # 2. Получение матрицы камеры
        K = self._build_camera_matrix(frame_shape)
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        
        # 3. Решение PnP задачи с фильтрацией выбросов через RANSAC
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_points_3d, img_points_2d, K, dist_coeffs, 
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        except Exception as e:
            logger.error(f"Ошибка solvePnPRansac: {e}")
            return None
            
        if not success or inliers is None or len(inliers) < 4:
            return None

        # 4. Вычисление мировых координат камеры из векторов вращения (rvec) и сдвига (tvec)
        R, _ = cv2.Rodrigues(rvec)
        # Позиция камеры в мировых координатах C = -R_transpose * T
        camera_position_world = -np.matrix(R).T * np.matrix(tvec)
        
        X_cam, Y_cam, Z_cam = camera_position_world.A1

        # 5. Фильтрация абсурдных значений высоты
        if not (0 < Z_cam < 10000.0): # Высота должна быть положительной
            return None

        # 6. Конвертация XY в GPS и временное сглаживание
        raw_lat, raw_lon = self._local_3d_to_gps(X_cam, Y_cam)
            
        # Проверка выхода за пределы карты
        lat_range = self.north - self.south
        lon_range = self.east - self.west
        if not (self.south - lat_range <= raw_lat <= self.north + lat_range):
            return None
        if not (self.west - lon_range <= raw_lon <= self.east + lon_range):
            return None

        self.last_raw_gps = (raw_lat, raw_lon)

        # EMA сглаживание
        if self.last_gps is None:
            smoothed_lat, smoothed_lon = raw_lat, raw_lon
        else:
            smoothed_lat = self.last_gps[0] * (1 - self.smoothing_factor) + raw_lat * self.smoothing_factor
            smoothed_lon = self.last_gps[1] * (1 - self.smoothing_factor) + raw_lon * self.smoothing_factor
            
        self.last_gps = (smoothed_lat, smoothed_lon)
        self.trajectory_gps.append(self.last_gps)
        
        return smoothed_lat, smoothed_lon, Z_cam

    def get_trajectory(self):
        """Возвращает историю успешно рассчитанных координат."""
        return self.trajectory_gps
