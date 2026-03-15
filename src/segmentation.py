"""
Модуль для сегментации изображений.
Удаление неба (Sky Masking) для отсечения ложных срабатываний при сопоставлении,
так как небо (и облака) не присутствуют на спутниковых надир-снимках.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SkySegmenter:
    def __init__(self, canny_low: int = 50, canny_high: int = 150):
        self.canny_low = canny_low
        self.canny_high = canny_high

    def remove_sky(self, frame: np.ndarray) -> np.ndarray:
        """
        Определяет линию горизонта и закрашивает все, что выше (небо) черным цветом,
        чтобы алгоритмы (LoFTR/SIFT) не искали там ключевые точки.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Применяем детектор границ (Canny)
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        
        # 2. Ищем длинные прямые линии (отрезки горизонта)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=20)
        
        h, w = frame.shape[:2]
        horizon_y = h // 2  # Дефолтное значение - половина экрана (если горизонт не найден)
        
        if lines is not None:
            # Ищем самую верхнюю горизонтальную линию (угол наклона < 15 градусов)
            highest_y = h
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Считаем угол
                if x2 - x1 == 0:
                    continue
                angle = abs(np.degrees(np.arctan((y2 - y1) / (x2 - x1))))
                
                if angle < 15: # Линия почти горизонтальна
                    avg_y = (y1 + y2) // 2
                    if avg_y < highest_y:
                        highest_y = avg_y
                        
            # Если мы нашли правдоподобный горизонт (не у самого края)
            if highest_y > int(h * 0.1):
                horizon_y = highest_y
                
        # 3. Делаем маску (закрашиваем небо черным)
        masked_frame = frame.copy()
        
        # Для реалистичности мы немного опускаем горизонт (на 5% от высоты), 
        # чтобы точно зацепить дымку/облака
        safe_horizon = min(horizon_y + int(h * 0.05), h)
        cv2.rectangle(masked_frame, (0, 0), (w, safe_horizon), (0, 0, 0), -1)
        
        return masked_frame
