"""
Модуль расчета координат самолета на основе жестких геометрических преобразований.
Поддерживает концепцию "Keyframe + Tracking" (ИИ-матчинг + Оптический поток).
обеспечивает защиту от физически невозможных скачков (ограничение скорости) 
и резервное копирование вычислений (Homography -> Affine).
"""

import cv2
import numpy as np
import logging
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)

class GeolocatorRigid:
    def __init__(
        self,
        bbox: Tuple[float, float, float, float],
        map_shape: Tuple[int, int],
        smoothing_factor: float = 0.2
    ):
        """
        :param bbox: (west, south, east, north) - географические границы текущей спутниковой карты.
        :param map_shape: (height, width) - разрешение скачанной карты в пикселях.
        :param smoothing_factor: Коэффициент EMA-сглаживания (0.0 - 1.0). 
                                 Меньше = сильнее сглаживание траектории.
        """
        self.west, self.south, self.east, self.north = bbox
        self.map_h, self.map_w = map_shape[:2]
        
        self.smoothing_factor = smoothing_factor
        self.last_gps = None
        self.trajectory_gps = []

        self.last_raw_gps = None
        self.last_candidate_gps = None
        self.last_match_stats = {}
        self.last_projection_candidate = None
        self.last_projection_polygon = None

    def update_bbox(self, bbox: Tuple[float, float, float, float], map_shape: Tuple[int, int]):
        """Обновляет географические границы при сдвиге локальной карты."""
        self.west, self.south, self.east, self.north = bbox
        self.map_h, self.map_w = map_shape[:2]

    def _pixel_to_gps(self, mx: float, my: float) -> Tuple[float, float]:
        """Линейная интерполяция пиксельных координат карты в WGS84 (Широта, Долгота)."""
        lon = self.west + (self.east - self.west) * (mx / self.map_w)
        lat = self.north - (self.north - self.south) * (my / self.map_h)
        return lat, lon

    def _estimate_with_homography(self, frame_shape: Tuple[int, int], mkpts_plane: np.ndarray, mkpts_map: np.ndarray):
        """
        Пытается вычислить 3D-проекцию (гомографию) кадра на карту.
        :return: (map_x, map_y, inlier_ratio, reproj_error, "homography", polygon) или None
        """
        if len(mkpts_plane) < 4:
            return None

        H, inliers = cv2.findHomography(mkpts_plane, mkpts_map, cv2.RANSAC, 5.0)
        if H is None or inliers is None:
            return None

        inlier_mask = inliers.ravel().astype(bool)
        inlier_count = int(inlier_mask.sum())
        # Требуем минимум 6 надежных точек для 3D проекции во избежание сильных искажений
        if inlier_count < 6:
            return None

        h, w = frame_shape[:2]

        # Проецируем оптический центр камеры на карту
        plane_center = np.array([[[w / 2.0, h / 2.0]]], dtype=np.float32)
        map_center = cv2.perspectiveTransform(plane_center, H)[0][0]
        mx, my = float(map_center[0]), float(map_center[1])

        # Вычисляем ошибку репроекции (насколько математика не совпала с реальностью)
        projected = cv2.perspectiveTransform(mkpts_plane.reshape(-1, 1, 2), H).reshape(-1, 2)
        err = np.linalg.norm(projected[inlier_mask] - mkpts_map[inlier_mask], axis=1)
        reproj_error = float(err.mean()) if err.size else float("inf")
        inlier_ratio = float(inlier_count / max(1, len(mkpts_plane)))

        frame_corners = np.array(
            [[[0.0, 0.0]], [[w - 1.0, 0.0]], [[w - 1.0, h - 1.0]], [[0.0, h - 1.0]]],
            dtype=np.float32,
        )
        polygon = cv2.perspectiveTransform(frame_corners, H).reshape(-1, 2)

        return mx, my, inlier_ratio, reproj_error, "homography", polygon

    def _estimate_with_affine(self, frame_shape: Tuple[int, int], mkpts_plane: np.ndarray, mkpts_map: np.ndarray):
        """
        Fallback-метод: вычисляет 2D аффинное преобразование (масштаб + поворот + сдвиг).
        Используется, если гомография не смогла сойтись (например, при строгом виде сверху).
        """
        if len(mkpts_plane) < 4:
            return None
        
        M, inliers = cv2.estimateAffinePartial2D(
            mkpts_plane,
            mkpts_map,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
        )
        if M is None:
            return None

        inlier_mask = np.ones(len(mkpts_plane), dtype=bool)
        if inliers is not None:
            inlier_mask = inliers.ravel().astype(bool)
        
        inlier_count = int(inlier_mask.sum())
        if inlier_count < 4:
            return None

        h, w = frame_shape[:2]
        plane_center = np.array([[[w / 2.0, h / 2.0]]], dtype=np.float32)
        map_center = cv2.transform(plane_center, M)[0][0]
        mx, my = float(map_center[0]), float(map_center[1])

        projected = cv2.transform(mkpts_plane.reshape(-1, 1, 2), M).reshape(-1, 2)
        err = np.linalg.norm(projected[inlier_mask] - mkpts_map[inlier_mask], axis=1)
        reproj_error = float(err.mean()) if err.size else float("inf")
        inlier_ratio = float(inlier_count / max(1, len(mkpts_plane)))

        frame_corners = np.array(
            [[[0.0, 0.0]], [[w - 1.0, 0.0]], [[w - 1.0, h - 1.0]], [[0.0, h - 1.0]]],
            dtype=np.float32,
        )
        polygon = cv2.transform(frame_corners, M).reshape(-1, 2)

        return mx, my, inlier_ratio, reproj_error, "affine", polygon

    def estimate_plane_position(self, frame_shape: Tuple[int, int], mkpts_plane: np.ndarray, mkpts_map: np.ndarray, is_keyframe: bool = False) -> Optional[Tuple[float, float]]:
        """
        Главный пайплайн расчета координат с многоуровневой защитой от выбросов.
        
        :param is_keyframe: Флаг True означает, что совпадения получены от тяжелой нейросети (LoFTR/LightGlue).
                            False означает быстрый межкадровый оптический поток.
        :return: (Latitude, Longitude, Altitude=0.0) или None, если расчет провален.
        """
        if len(mkpts_plane) < 4:
            self._reset_stats("too_few_matches", is_keyframe, len(mkpts_plane))
            return None

        try:
            est = None
            # Для Keyframe всегда пытаемся сначала найти 3D-гомографию
            if is_keyframe:
                est = self._estimate_with_homography(frame_shape, mkpts_plane, mkpts_map)
            # Если это оптический поток ИЛИ гомография провалилась - используем стабильный Affine
            if est is None:
                est = self._estimate_with_affine(frame_shape, mkpts_plane, mkpts_map)

            if est is None:
                self._reset_stats("no_model_fit", is_keyframe, len(mkpts_plane))
                return None
        except Exception as e:
            logger.error(f"Geometric estimation error: {e}")
            self._reset_stats("geometry_exception", is_keyframe, len(mkpts_plane))
            return None

        mx, my, inlier_ratio, reproj_error, model_type, polygon = est

        self.last_projection_candidate = polygon
        self.last_projection_polygon = None
        self.last_match_stats = {
            "model": model_type,
            "inlier_ratio": float(inlier_ratio),
            "reproj_error": float(reproj_error),
            "is_keyframe": bool(is_keyframe),
            "matches": int(len(mkpts_plane)),
            "accepted": False,
            "reject_reason": "",
        }

        if is_keyframe:
            if inlier_ratio < 0.22 or reproj_error > 18.0:
                self.last_match_stats["reject_reason"] = "weak_keyframe_geometry"
                return None
        else:
            if inlier_ratio < 0.30 or reproj_error > 10.0:
                self.last_match_stats["reject_reason"] = "weak_track_geometry"
                return None

        lat, lon = self._pixel_to_gps(mx, my)
        self.last_candidate_gps = (lat, lon)
        
        # Защита от выхода за границы карты
        lat_range = self.north - self.south
        lon_range = self.east - self.west
        if not (self.south - lat_range*2 <= lat <= self.north + lat_range*2):
            self.last_match_stats["reject_reason"] = "lat_out_of_bounds"
            return None
        if not (self.west - lon_range*2 <= lon <= self.east + lon_range*2):
            self.last_match_stats["reject_reason"] = "lon_out_of_bounds"
            return None

        # ЖЕСТКАЯ ФИЗИЧЕСКАЯ ЗАЩИТА: ограничение скорости
        if self.last_gps is not None and not is_keyframe:
            # 1 градус широты = ~111.1 км
            d_lat_m = (lat - self.last_gps[0]) * 111111.0
            d_lon_m = (lon - self.last_gps[1]) * 111111.0 * np.cos(np.radians(self.last_gps[0]))
            dist_meters = np.sqrt(d_lat_m**2 + d_lon_m**2)
            
            # Если между кадрами прыжок > 70 метров (скорость 1000 км/ч), это ошибка проекции
            # Но если это Keyframe (LoFTR), мы разрешаем прыжок, чтобы восстановиться после Target Lost!
            if dist_meters > 70.0:
                self.last_match_stats["reject_reason"] = "motion_jump_guard"
                return None

        # Только подтвержденная геометрия должна попадать в визуальный контур.
        self.last_projection_polygon = self.last_projection_candidate
        self.last_match_stats["accepted"] = True
        self.last_raw_gps = (lat, lon)
        
        # EMA сглаживание:
        if self.last_gps is None or is_keyframe:
            smoothed_lat, smoothed_lon = lat, lon
        else:
            smoothed_lat = self.last_gps[0] * (1 - self.smoothing_factor) + lat * self.smoothing_factor
            smoothed_lon = self.last_gps[1] * (1 - self.smoothing_factor) + lon * self.smoothing_factor
            
        self.last_gps = (smoothed_lat, smoothed_lon)
        self.trajectory_gps.append(self.last_gps)
        
        # Возвращаем (lat, lon, 0.0) для совместимости с интерфейсами, требующими Altitude
        return smoothed_lat, smoothed_lon, 0.0

    def _reset_stats(self, reason: str, is_keyframe: bool, matches_count: int):
        """Вспомогательный метод для очистки состояния при неудачном матчинге."""
        self.last_projection_candidate = None
        self.last_projection_polygon = None
        self.last_candidate_gps = None
        self.last_match_stats = {
            "model": "none",
            "inlier_ratio": 0.0,
            "reproj_error": float("inf"),
            "is_keyframe": bool(is_keyframe),
            "matches": matches_count,
            "accepted": False,
            "reject_reason": reason,
        }

    def get_trajectory(self):
        """Возвращает историю успешно рассчитанных координат."""
        return self.trajectory_gps

    # Алиас для обратной совместимости со старыми скриптами запуска.
    def estimate_drone_position(self, frame_shape, mkpts_drone, mkpts_map, is_keyframe: bool = False):
        return self.estimate_plane_position(frame_shape, mkpts_drone, mkpts_map, is_keyframe=is_keyframe)
