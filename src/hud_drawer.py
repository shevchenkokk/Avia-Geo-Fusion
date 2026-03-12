"""
Модуль для генерации видеоотчета (Картинка-в-картинке / HUD).
Отображает видео с GoPro, соединенное с картой, и накладывает телеметрию поверх.
"""

import cv2
import numpy as np
from typing import Tuple, Optional

class HUDDrawer:
    @staticmethod
    def draw_telemetry(frame: np.ndarray, 
                       gps: Optional[Tuple[float, float]], 
                       inliers_count: int, 
                       is_ai: bool = True) -> np.ndarray:
        """
        Рисует плашку с данными поверх кадра (как в авиасимуляторах).
        :param frame: Изображение
        :param gps: (lat, lon) расчетные
        :param inliers_count: Количество найденных хороших точек связи
        """
        h, w = frame.shape[:2]
        
        # Полупрозрачная черная подложка в левом верхнем углу
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 150), (0, 0, 0), -1)
        
        # Смешиваем подложку с кадром (альфа 0.6)
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Настройки шрифта
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 255, 0) # Зеленый текст (HUD)
        
        # Пишем статус AI
        algo_name = "LoFTR (Transformer)" if is_ai else "Classical (SIFT)"
        cv2.putText(frame, f"ALGO : {algo_name}", (20, 40), font, 0.7, (200, 200, 200), 2)
        
        # Пишем количество точек (качество связи)
        status_color = (0, 255, 0) if inliers_count > 10 else (0, 0, 255)
        cv2.putText(frame, f"INLIER POINTS : {inliers_count}", (20, 75), font, 0.7, status_color, 2)
        
        # Пишем GPS
        if gps:
            lat, lon = gps
            cv2.putText(frame, f"EST. LAT : {lat:.6f}", (20, 110), font, 0.7, color, 2)
            cv2.putText(frame, f"EST. LON : {lon:.6f}", (20, 140), font, 0.7, color, 2)
        else:
            cv2.putText(frame, "EST. LAT : NO LOCK", (20, 110), font, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EST. LON : NO LOCK", (20, 140), font, 0.7, (0, 0, 255), 2)
            
        return frame
