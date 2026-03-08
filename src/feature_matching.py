"""
Модуль для сопоставления изображений (кадр с БПЛА и спутниковая карта).
Реализованы базовые методы (SIFT, ORB), которые послужат baseline-моделью.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

class FeatureMatcher:
    """Класс для поиска и сопоставления ключевых точек между двумя изображениями."""
    
    def __init__(self, method: str = 'sift', max_features: int = 2000):
        """
        :param method: 'sift' или 'orb'
        :param max_features: ограничение на количество извлекаемых фичей
        """
        self.method = method.lower()
        self.max_features = max_features
        
        if self.method == 'sift':
            self.detector = cv2.SIFT_create(nfeatures=self.max_features)
            # Для SIFT используем FLANN based matcher 
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
            
        elif self.method == 'orb':
            self.detector = cv2.ORB_create(nfeatures=self.max_features)
            # Для ORB используем Brute-Force matcher с Hamming distance
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
        else:
            raise ValueError(f"Неизвестный метод {method}, доступны: 'sift', 'orb'")

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Переводит изображение в градации серого, если оно цветное."""
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def match(self, img_drone: np.ndarray, img_map: np.ndarray) -> Dict[str, Any]:
        """
        Ищет соответствия между кадром с дрона и спутниковой картой.
        
        :return: Словарь с ключевыми точками, совпадениями (good matches)
                 и готовым изображением отрисованных совпадений (debug_img).
        """
        # Препроцессинг
        gray_drone = self._preprocess_image(img_drone)
        gray_map = self._preprocess_image(img_map)
        
        # Нахождение ключевых точек и дескрипторов
        logger.info(f"Извлечение фичей ({self.method})...")
        kp1, des1 = self.detector.detectAndCompute(gray_drone, None)
        kp2, des2 = self.detector.detectAndCompute(gray_map, None)
        
        if des1 is None or des2 is None:
            logger.warning("Не удалось найти дескрипторы на одном из изображений.")
            return {'matches': [], 'kp1': kp1, 'kp2': kp2, 'debug_img': None}
        
        # Сопоставление (Matching)
        logger.info("Сопоставление фичей...")
        good_matches = []
        
        if self.method == 'sift':
            # k-Nearest Neighbors (k=2) для применения Lowe's ratio test
            matches = self.matcher.knnMatch(des1, des2, k=2)
            for m, n in matches:
                # Тест Лоу (Lowe's ratio test) - фильтрует неоднозначные мэтчи
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        else:
            # Для ORB (без kNN, используется crossCheck=True)
            matches = self.matcher.match(des1, des2)
            good_matches = sorted(matches, key=lambda x: x.distance)
            keep_count = max(10, int(len(good_matches) * 0.15))
            good_matches = good_matches[:keep_count]
            
        # --- ФИЛЬТРАЦИЯ ЧЕРЕЗ RANSAC ---
        inlier_matches = []
        if len(good_matches) > 4:  # Для Гомографии нужно минимум 4 точки
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Находим матрицу гомографии и маску Inliers (правильных точек)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if mask is not None:
                matchesMask = mask.ravel().tolist()
                for i, is_inlier in enumerate(matchesMask):
                    if is_inlier:
                        inlier_matches.append(good_matches[i])
        
        logger.info(f"Найдено совпадений: {len(good_matches)}, после RANSAC: {len(inlier_matches)}")
        
        # Отрисовка результатов (для дебага)
        debug_img = cv2.drawMatches(
            img_drone, kp1, 
            img_map, kp2, 
            inlier_matches, None, # Отрисовываем ТОЛЬКО INLIERS
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchColor=(0, 255, 0) # Зеленые линии
        )
        
        return {
            'kp1': kp1,
            'kp2': kp2,
            'matches': good_matches,
            'debug_img': debug_img
        }
