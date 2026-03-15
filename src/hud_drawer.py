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
        border_radius = 5
        rect_color = (25, 25, 25)
        text_color = (0, 255, 120)  # Кибер-зеленый HUD
        warning_color = (0, 50, 255)
        
        # Основной блок HUD
        cv2.rectangle(overlay, (15, 15), (520, 180), rect_color, -1)
        cv2.rectangle(overlay, (15, 15), (520, 180), text_color, 2)
        
        # Смешиваем подложку с кадром (альфа 0.75 для контрастности)
        alpha = 0.75
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Декоративные элементы HUD
        cv2.line(frame, (15, 60), (520, 60), text_color, 1)
        cv2.circle(frame, (35, 38), 5, text_color if inliers_count > 10 else warning_color, -1)
        
        # Настройки шрифта
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Векторный заголовок
        cv2.putText(frame, "AVIA-GEO-FUSION WGS84 TRACKER", (60, 42), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Статус алгоритма
        mode_text = "[AI KEYFRAME + OPTICAL FLOW]" if is_ai else "[CLASSICAL MATCHING]"
        cv2.putText(frame, f"TRACKING MODE : {mode_text}", (25, 90), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Количество точек
        cv2.putText(frame, f"SATELLITE BEACONS (INLIERS) : {inliers_count}", (25, 120), font, 0.6, 
                    text_color if inliers_count > 10 else warning_color, 1, cv2.LINE_AA)
        
        # Пишем GPS с красивым форматированием
        if gps:
            lat, lon = gps
            # Эфмулируем приборную панель
            cv2.putText(frame, f"LAT: {lat:.7f} N", (25, 155), font, 0.8, text_color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"LON: {lon:.7f} E", (280, 155), font, 0.8, text_color, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "LAT: --.------- N", (25, 155), font, 0.8, warning_color, 2, cv2.LINE_AA)
            cv2.putText(frame, "LON: --.------- E", (280, 155), font, 0.8, warning_color, 2, cv2.LINE_AA)
            
            # Мигающая надпись GPS LOST
            cv2.putText(frame, "WARNING: TARGET LOST", (25, 215), font, 0.8, warning_color, 2, cv2.LINE_AA)

            cv2.putText(frame, "EST. LON : NO LOCK", (20, 140), font, 0.7, (0, 0, 255), 2)
            
        return frame
