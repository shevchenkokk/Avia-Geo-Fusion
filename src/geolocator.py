"""
Модуль для расчета физических GPS-координат из пиксельных совпадений
между кадром с самолета и спутниковой картой.
Использует Фильтр Калмана для учета физики движения и сглаживания траектории.

Stage 1.1: каждый вызов ``estimate_center_gps`` заполняет ``self.last_diagnostics``
вне зависимости от того, вернула ли функция координаты или None. Это нужно,
чтобы внешний DiagnosticLogger мог писать одну строку CSV на кадр и видеть
причину провала (reject_reason), даже когда геолокатор молча возвращает None.
"""

import cv2
import logging
import numpy as np
from typing import Tuple, List, Optional, Any, Dict

logger = logging.getLogger(__name__)

class Geolocator:
    # Пороги принятия этапа 1.4 (PROJECT_PLAN.md §1.4). Настраиваются через kwargs __init__.
    DEFAULT_MIN_INLIERS = 8
    DEFAULT_MIN_INLIER_RATIO = 0.15
    DEFAULT_MAX_SCALE_CHANGE = 1.5
    DEFAULT_MAX_SPEED_METERS = 70.0

    def __init__(
        self,
        bbox: Tuple[float, float, float, float],
        map_shape: Tuple[int, int],
        min_inliers: int = DEFAULT_MIN_INLIERS,
        min_inlier_ratio: float = DEFAULT_MIN_INLIER_RATIO,
        max_scale_change: float = DEFAULT_MAX_SCALE_CHANGE,
        max_speed_meters: float = DEFAULT_MAX_SPEED_METERS,
    ):
        """
        :param bbox: Границы карты (west, south, east, north)
        :param map_shape: Разрешение карты (height, width)
        :param min_inliers: минимум RANSAC-инлайеров для принятия лока.
        :param min_inlier_ratio: минимальное inliers/raw_matches.
        :param max_scale_change: максимальное относительное изменение масштаба
            гомографии между соседними успешными кадрами; нарушение часто
            означает, что матчер залип на ложный регион.
        :param max_speed_meters: максимальный физически правдоподобный
            прыжок raw_gps между соседними успешными кадрами (м).
        """
        self.west, self.south, self.east, self.north = bbox
        self.map_h, self.map_w = map_shape[:2]

        self.min_inliers = int(min_inliers)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.max_scale_change = float(max_scale_change)
        self.max_speed_meters = float(max_speed_meters)

        # История
        self.last_gps: Optional[Tuple[float, float]] = None
        self.trajectory_gps: List[Tuple[float, float]] = []
        self.last_raw_gps: Optional[Tuple[float, float, float]] = None
        self.last_homography_scale: Optional[float] = None

        # Диагностика этапа 1.1: всегда содержит данные о последнем кадре,
        # даже если estimate_center_gps вернула None.
        self.last_diagnostics: Dict[str, Any] = {}

        # Фильтр Калмана: 4 состояния [lat, lon, v_lat, v_lon], 2 измерения [lat, lon]
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
        """Обновляет географические границы карты при сдвиге окна MapManager."""
        self.west, self.south, self.east, self.north = bbox
        self.map_h, self.map_w = map_shape[:2]
        logger.debug(f"[Geolocator] bbox обновлён: {bbox}")

    def pixel_to_gps(self, x: float, y: float) -> Tuple[float, float]:
        """Переводит пиксели (x, y) на скачанной карте в (Latitude, Longitude)."""
        # Долгота: линейно от запада к востоку
        lon = self.west + (self.east - self.west) * (x / self.map_w) if self.map_w > 0 else self.west
        # Широта: y=0 это север, y=map_h это юг
        lat = self.north - (self.north - self.south) * (y / self.map_h) if self.map_h > 0 else self.north
        return lat, lon

    def _reset_diagnostics(self, num_matches: int) -> None:
        self.last_diagnostics = {
            "num_matches": int(num_matches),
            "num_inliers": 0,
            "inlier_ratio": float("nan"),
            "reproj_error_px": float("nan"),
            "homography_scale": float("nan"),
            "reject_reason": "",
            "raw_lat": float("nan"),
            "raw_lon": float("nan"),
            "predicted_lat": float("nan"),
            "predicted_lon": float("nan"),
            "kalman_lat": float("nan"),
            "kalman_lon": float("nan"),
            "homography": None,
        }

    @staticmethod
    def _homography_scale(H: np.ndarray) -> float:
        a = np.asarray(H, dtype=float)[:2, :2]
        det = float(np.linalg.det(a))
        if det <= 0:
            return float("nan")
        return float(np.sqrt(det))

    @staticmethod
    def _reproj_error(pts0: np.ndarray, pts1: np.ndarray, H: np.ndarray, mask: Optional[np.ndarray]) -> float:
        if pts0 is None or len(pts0) == 0:
            return float("nan")
        if mask is not None:
            m = mask.ravel().astype(bool)
            if m.any():
                pts0 = pts0[m]
                pts1 = pts1[m]
        if len(pts0) == 0:
            return float("nan")
        src = np.asarray(pts0, dtype=np.float64).reshape(-1, 1, 2)
        try:
            warped = cv2.perspectiveTransform(src, H).reshape(-1, 2)
        except Exception:
            return float("nan")
        diffs = warped - np.asarray(pts1, dtype=np.float64).reshape(-1, 2)
        return float(np.sqrt((diffs ** 2).sum(axis=1)).mean())

    def estimate_center_gps(
        self,
        frame_shape: Tuple[int, int],
        mkpts_plane: np.ndarray,
        mkpts_map: np.ndarray,
        is_keyframe: bool = False
    ) -> Optional[Tuple[float, float, float]]:
        """
        Рассчитывает GPS-координаты, в которую смотрит центр камеры самолета.

        :param frame_shape: (height, width) кадра самолёта
        :param mkpts_plane: Пиксели на кадре
        :param mkpts_map: Пиксели на карте
        :param is_keyframe: Флаг надежного кадра от нейросети (сбрасывает Калмана)
        :return: (lat, lon, altitude=0.0) сглаженные фильтром Калмана
        """
        num_matches = int(len(mkpts_plane))
        self._reset_diagnostics(num_matches)

        if num_matches < 4:
            self.last_diagnostics["reject_reason"] = "too_few_matches"
            return None

        # 1. Гомография + RANSAC
        H, mask = cv2.findHomography(mkpts_plane, mkpts_map, cv2.RANSAC, 5.0)
        if H is None:
            self.last_diagnostics["reject_reason"] = "homography_failed"
            return None

        num_inliers = int(mask.sum()) if mask is not None else 0
        inlier_ratio = num_inliers / max(1, num_matches)
        scale = self._homography_scale(H)
        reproj_px = self._reproj_error(mkpts_plane, mkpts_map, H, mask)

        self.last_diagnostics.update({
            "num_inliers": num_inliers,
            "inlier_ratio": float(inlier_ratio),
            "homography_scale": scale,
            "reproj_error_px": reproj_px,
            "homography": H,
        })

        # Этап 1.4: отбрасываем «мусорные» фиксации по трём критериям:
        # (a) мало inliers — низкая статистическая поддержка,
        # (b) низкий inlier_ratio — RANSAC вытащил капли из шума (залипание на ложный регион),
        # (c) скачок масштаба гомографии >1.5× — матчер переключился на другой регион тайла.
        if num_inliers < self.min_inliers:
            self.last_diagnostics["reject_reason"] = "too_few_inliers"
            return None
        if not np.isfinite(inlier_ratio) or inlier_ratio < self.min_inlier_ratio:
            self.last_diagnostics["reject_reason"] = "low_inlier_ratio"
            return None
        if (
            self.last_homography_scale is not None
            and np.isfinite(scale)
            and scale > 0
            and self.last_homography_scale > 0
        ):
            ratio_up = scale / self.last_homography_scale
            ratio_dn = self.last_homography_scale / scale
            if max(ratio_up, ratio_dn) > self.max_scale_change:
                self.last_diagnostics["reject_reason"] = "scale_change"
                self.last_diagnostics["scale_change_ratio"] = float(max(ratio_up, ratio_dn))
                return None

        frame_h, frame_w = frame_shape[:2]
        # 2. Берем центральную точку кадра самолёта
        center_pt = np.array([[[frame_w / 2.0, frame_h / 2.0]]], dtype=np.float32)

        # 3. Проецируем центр кадра на спутниковую карту
        try:
            center_map_pt = cv2.perspectiveTransform(center_pt, H)
        except Exception:
            self.last_diagnostics["reject_reason"] = "perspective_transform_failed"
            return None

        map_x, map_y = center_map_pt[0][0]

        # Проверка, что точка не улетела вообще за пределы карты
        if map_x < -self.map_w or map_x > self.map_w * 2 or map_y < -self.map_h or map_y > self.map_h * 2:
            self.last_diagnostics["reject_reason"] = "out_of_map"
            return None

        # 4. Переводим пиксель карты в GPS
        raw_lat, raw_lon = self.pixel_to_gps(map_x, map_y)
        self.last_diagnostics["raw_lat"] = float(raw_lat)
        self.last_diagnostics["raw_lon"] = float(raw_lon)

        # ФИЗИЧЕСКАЯ ЗАЩИТА (скоростной лимит)
        if self.last_raw_gps is not None and not is_keyframe:
            d_lat_m = (raw_lat - self.last_raw_gps[0]) * 111111.0
            d_lon_m = (raw_lon - self.last_raw_gps[1]) * 111111.0 * np.cos(np.radians(self.last_raw_gps[0]))
            dist_meters = float(np.sqrt(d_lat_m**2 + d_lon_m**2))
            self.last_diagnostics["raw_jump_meters"] = dist_meters

            # Если прыжок > max_speed_meters, а это не Keyframe — отбрасываем как ошибку матчинга
            if dist_meters > self.max_speed_meters:
                self.last_diagnostics["reject_reason"] = "speed_violation"
                return None

        self.last_raw_gps = (raw_lat, raw_lon)
        # Сохраняем масштаб ТОЛЬКО на принятых кадрах, чтобы scale_change
        # check сравнивал с последним «хорошим» масштабом, а не с любым.
        if np.isfinite(scale):
            self.last_homography_scale = float(scale)

        # 5. Фильтр Калмана. Перед correct() запоминаем prediction для CSV.
        if not self.is_kalman_inited or is_keyframe:
            self.kalman.statePre = np.array([[raw_lat], [raw_lon], [0.0], [0.0]], np.float32)
            self.kalman.statePost = np.array([[raw_lat], [raw_lon], [0.0], [0.0]], np.float32)
            self.is_kalman_inited = True
            kalman_lat, kalman_lon = float(raw_lat), float(raw_lon)
            pred_lat, pred_lon = float(raw_lat), float(raw_lon)
        else:
            predicted = self.kalman.predict()
            pred_lat = float(np.asarray(predicted).ravel()[0])
            pred_lon = float(np.asarray(predicted).ravel()[1])

            measurement = np.array([[np.float32(raw_lat)], [np.float32(raw_lon)]])
            estimated = self.kalman.correct(measurement)
            kalman_lat = float(np.asarray(estimated).ravel()[0])
            kalman_lon = float(np.asarray(estimated).ravel()[1])

        self.last_diagnostics["predicted_lat"] = pred_lat
        self.last_diagnostics["predicted_lon"] = pred_lon
        self.last_diagnostics["kalman_lat"] = kalman_lat
        self.last_diagnostics["kalman_lon"] = kalman_lon

        # last_gps — 2-tuple (lat, lon) для совместимости с существующими потребителями.
        self.last_gps = (kalman_lat, kalman_lon)
        self.trajectory_gps.append(self.last_gps)

        return kalman_lat, kalman_lon, 0.0

    def get_trajectory(self):
        """Возвращает историю успешно рассчитанных координат."""
        return self.trajectory_gps

    def predict_only(self) -> Optional[Tuple[float, float]]:
        """Счисление пути (этап 1.3): шаг Калмана без измерения.

        Применяется на кадрах, где ``estimate_center_gps`` отказал,
        чтобы поиск следующего тайла следовал предсказанной траектории.

        Возвращает (lat, lon) или None до первой успешной инициализации.
        """
        if not self.is_kalman_inited:
            return None
        predicted = self.kalman.predict()
        pred_arr = np.asarray(predicted).ravel()
        pred_lat, pred_lon = float(pred_arr[0]), float(pred_arr[1])
        # Синхронизируем statePost с statePre: без correct() повторные predict()
        # бесконечно раздувают ковариацию.
        self.kalman.statePost = self.kalman.statePre.copy()
        self.last_diagnostics["predicted_lat"] = pred_lat
        self.last_diagnostics["predicted_lon"] = pred_lon
        return pred_lat, pred_lon
