"""
Модуль для генерации видеоотчета (Картинка-в-картинке / HUD).
Отображает видео с камеры и накладывает телеметрию поверх.
"""

import cv2
import numpy as np
from typing import Tuple, Optional

class HUDDrawer:
    @staticmethod
    def draw_telemetry(
        frame: np.ndarray,
        gps: Optional[Tuple[float, float]],
        inliers_count: int,
        is_ai: bool = True,
        gps_locked: bool = True,
        reject_reason: str = "",
        map_zoom: Optional[int] = None,
        seg_status: Optional[str] = None,
        match_used: Optional[int] = None,
        match_raw: Optional[int] = None,
        class_stats_line: Optional[str] = None,
    ) -> np.ndarray:
        """
        Рисует плашку с данными поверх кадра (как в авиасимуляторах).
        :param frame: Изображение
        :param gps: (lat, lon) расчетные
        :param inliers_count: Количество найденных хороших точек связи
        """
        _h, _w = frame.shape[:2]

        box_width = max(550, min(int(_w * 0.35), 800))
        x0, y0 = 15, 15
        # Если есть строка распределения keypoint'ов по классам — добавляем ей место.
        box_h = 240 if not class_stats_line else 268
        x1, y1 = x0 + box_width, box_h

        # Если кадр совсем крошечный, не даем плашке вылезти за экран
        x1 = min(x1, _w - 15)
        y1 = min(y1, _h - 15)

        rect_color = (25, 25, 25)
        text_color = (0, 255, 120)
        warning_color = (0, 50, 255)
        soft_text = (210, 210, 210)

        roi = frame[y0:y1, x0:x1]
        overlay_roi = roi.copy()
        
        # Рисуем фон только на маленьком кусочке
        cv2.rectangle(overlay_roi, (0, 0), (x1 - x0, y1 - y0), rect_color, -1)
        
        # Смешиваем и вставляем обратно в большой кадр
        alpha = 0.75
        frame[y0:y1, x0:x1] = cv2.addWeighted(overlay_roi, alpha, roi, 1 - alpha, 0)

        # Рисуем рамку вокруг плашки
        cv2.rectangle(frame, (x0, y0), (x1, y1), text_color, 2)

        # Декоративные элементы HUD
        cv2.line(frame, (x0, y0 + 45), (x1, y0 + 45), text_color, 1)
        cv2.circle(frame, (x0 + 20, y0 + 23), 5, text_color if inliers_count > 10 else warning_color, -1)

        # Настройки шрифта
        font = cv2.FONT_HERSHEY_SIMPLEX

        current_y = y0 + 27  # Начальная высота для текста
        line_step = 26       # Шаг между строками

        # Векторный заголовок
        cv2.putText(frame, "AVIA-GEO-FUSION TRACKER", (x0 + 45, current_y), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        current_y += 46 # Большой отступ после заголовка

        # Статус алгоритма
        mode_text = "[AI KEYFRAME + OPTICAL FLOW]" if is_ai else "[CLASSICAL MATCHING]"
        cv2.putText(frame, f"MODE: {mode_text}", (x0 + 10, current_y), font, 0.58, soft_text, 1, cv2.LINE_AA)
        current_y += line_step

        # Количество точек
        cv2.putText(
            frame,
            f"BEACONS (INLIERS): {inliers_count}",
            (x0 + 10, current_y),
            font,
            0.58,
            text_color if inliers_count > 10 else warning_color,
            1,
            cv2.LINE_AA,
        )
        current_y += line_step

        # Статус локализации
        lock_color = text_color if gps_locked else warning_color
        reject_suffix = f" [{reject_reason}]" if (not gps_locked and reject_reason) else ""
        cv2.putText(frame, f"GPS LOCK: {'YES' if gps_locked else 'NO'}{reject_suffix}", (x0 + 10, current_y), font, 0.58, lock_color, 1, cv2.LINE_AA)
        current_y += line_step

        # Зум и Матчинг
        zoom_text = f"ZOOM: {map_zoom}" if map_zoom is not None else "ZOOM: n/a"
        if seg_status is not None and match_used is not None and match_raw is not None:
            zoom_text += f" | SEG: {seg_status}  | MATCH: {match_used}/{match_raw}"
        cv2.putText(frame, zoom_text, (x0 + 10, current_y), font, 0.58, soft_text, 1, cv2.LINE_AA)
        current_y += line_step + 4

        # Координаты: показываем lock-координаты; при no-lock можно показать кандидатные.
        if gps is not None:
            lat, lon = float(gps[0]), float(gps[1])
            coord_color = text_color if gps_locked else (0, 190, 255)
            tag = "" if gps_locked else "*"

            str_lat = f"LAT{tag}: {lat:+010.6f} N"
            str_lon = f"LON{tag}: {lon:+010.6f} E"

            cv2.putText(frame, str_lat, (x0 + 10, current_y), font, 0.7, coord_color, 2, cv2.LINE_AA)
            cv2.putText(frame, str_lon, (x0 + 280, current_y), font, 0.7, coord_color, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "LAT: --.------- N", (x0 + 10, current_y), font, 0.7, warning_color, 2, cv2.LINE_AA)
            cv2.putText(frame, "LON: --.------- E", (x0 + 280, current_y), font, 0.7, warning_color, 2, cv2.LINE_AA)
            current_y += line_step
            cv2.putText(frame, "WARNING: TARGET LOST", (x0 + 10, current_y), font, 0.62, warning_color, 2, cv2.LINE_AA)

        if class_stats_line:
            current_y += line_step
            cv2.putText(
                frame,
                f"KPTS  {class_stats_line}",
                (x0 + 10, current_y),
                font,
                0.48,
                (180, 200, 255),
                1,
                cv2.LINE_AA,
            )

        return frame
