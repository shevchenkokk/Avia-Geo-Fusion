"""
Модуль для генерации видеоотчета (Картинка-в-картинке / HUD).
Отображает видео с камеры и накладывает телеметрию поверх.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class HudState:
    """Срез состояния пайплайна для HUD-панели (run_full_pipeline.py)."""
    # Position / state
    lat: float = float("nan")
    lon: float = float("nan")
    sigma_pos_m: float = float("nan")
    speed_mps: float = 0.0
    heading_deg: float = 0.0
    bank_deg: float = 0.0
    agl_m: float = float("nan")
    scale_bias: float = 1.0
    scale_sigma: float = 0.0
    # Mode / flags
    track_mode: str = "bootstrap"
    obstructed: bool = False
    mask_method: str = ""
    mask_confidence: float = 1.0
    # Channels — running tally
    app_attempts: int = 0
    app_accepts: int = 0
    struct_attempts: int = 0
    struct_accepts: int = 0
    # Per-frame indicators (последний кадр)
    map_accepted_now: bool = False
    structural_accepted_now: bool = False
    consistency_rejected_now: bool = False
    # Timing
    t_frame_ms: float = 0.0
    t_vo_ms: float = 0.0
    t_app_ms: float = 0.0
    t_struct_ms: float = 0.0
    # Trajectory mini-map
    enu_xy_history: list = field(default_factory=list)  # [(x_e_m, y_n_m), ...]


_TRACK_MODE_COLORS = {
    "track":      (90, 220, 90),    # green
    "weak":       (50, 200, 240),   # amber
    "relocalize": (50, 80, 240),    # red
    "bootstrap":  (200, 200, 200),  # grey
}


class HUDDrawer:
    @staticmethod
    def draw_pipeline_state(frame: np.ndarray, state: HudState) -> np.ndarray:
        """Современный HUD для run_full_pipeline.py: режим, позиция, σ,
        кинематика, каналы, тайминги, мини-траектория.
        """
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        soft = (210, 210, 210)
        warn = (40, 60, 240)
        ok = (90, 220, 90)

        # Панель состояния слева сверху.
        panel_w = min(560, int(w * 0.32))
        panel_h = 280
        x0, y0 = 18, 18
        x1, y1 = x0 + panel_w, y0 + panel_h
        roi = frame[y0:y1, x0:x1]
        bg = np.zeros_like(roi)
        cv2.rectangle(bg, (0, 0), (panel_w, panel_h), (28, 28, 28), -1)
        frame[y0:y1, x0:x1] = cv2.addWeighted(bg, 0.72, roi, 0.28, 0)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (90, 90, 90), 1)

        # Заголовок и индикатор режима.
        mode = state.track_mode.lower()
        mode_color = _TRACK_MODE_COLORS.get(mode, soft)
        cv2.putText(frame, "AVIA-GEO-FUSION", (x0 + 12, y0 + 28),
                    font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        # Цветная метка режима.
        bx0 = x0 + panel_w - 130
        by0 = y0 + 8
        cv2.rectangle(frame, (bx0, by0), (bx0 + 118, by0 + 30), mode_color, -1)
        cv2.putText(frame, mode.upper(), (bx0 + 8, by0 + 22),
                    font, 0.55, (10, 10, 10), 2, cv2.LINE_AA)
        cv2.line(frame, (x0 + 8, y0 + 42), (x1 - 8, y0 + 42), (90, 90, 90), 1)

        cy = y0 + 64
        step = 22

        # Текущая географическая позиция.
        if np.isfinite(state.lat) and np.isfinite(state.lon):
            cv2.putText(
                frame,
                f"LAT {state.lat:+.6f}   LON {state.lon:+.6f}",
                (x0 + 12, cy), font, 0.52, soft, 1, cv2.LINE_AA,
            )
        else:
            cv2.putText(frame, "LAT --.------   LON --.------",
                        (x0 + 12, cy), font, 0.52, warn, 1, cv2.LINE_AA)
        cy += step
        sigma_color = ok if state.sigma_pos_m < 60 else (
            (60, 200, 240) if state.sigma_pos_m < 200 else warn
        )
        cv2.putText(
            frame, f"sigma {state.sigma_pos_m:6.1f} m   AGL {state.agl_m:5.0f} m",
            (x0 + 12, cy), font, 0.52, sigma_color, 1, cv2.LINE_AA,
        )
        cy += step

        # Кинематика. Используем deg вместо °, потому что OpenCV рисует только ASCII.
        cv2.putText(
            frame,
            f"V {state.speed_mps:5.1f} m/s   HDG {((state.heading_deg + 360) % 360):5.1f}deg   BANK {state.bank_deg:+5.1f}deg",
            (x0 + 12, cy), font, 0.52, soft, 1, cv2.LINE_AA,
        )
        cy += step
        cv2.putText(
            frame,
            f"scale_bias {state.scale_bias:.3f} +/- {state.scale_sigma:.3f}",
            (x0 + 12, cy), font, 0.48, (180, 200, 255), 1, cv2.LINE_AA,
        )
        cy += step + 4

        # Каналы абсолютных измерений.
        app_pct = (100.0 * state.app_accepts / max(1, state.app_attempts))
        cv2.putText(
            frame,
            f"APP    {state.app_accepts:>3}/{state.app_attempts:<3}  ({app_pct:4.1f}%)",
            (x0 + 12, cy), font, 0.48,
            ok if state.map_accepted_now else soft, 1, cv2.LINE_AA,
        )
        cy += step - 2
        struct_pct = (100.0 * state.struct_accepts / max(1, state.struct_attempts))
        cv2.putText(
            frame,
            f"STRUCT {state.struct_accepts:>3}/{state.struct_attempts:<3}  ({struct_pct:4.1f}%)",
            (x0 + 12, cy), font, 0.48,
            ok if state.structural_accepted_now else soft, 1, cv2.LINE_AA,
        )
        cy += step

        # Диагностические флаги.
        flag_y = cy
        if state.obstructed:
            cv2.putText(frame, "OBSTRUCTION", (x0 + 12, flag_y),
                        font, 0.5, warn, 2, cv2.LINE_AA)
        if state.mask_method == "fallback_anchor":
            cv2.putText(frame, "MASK FALLBACK", (x0 + 160, flag_y),
                        font, 0.5, (50, 200, 240), 1, cv2.LINE_AA)
        if state.consistency_rejected_now:
            cv2.putText(frame, "CONSIST.REJ", (x0 + 320, flag_y),
                        font, 0.5, warn, 1, cv2.LINE_AA)
        cy += step - 2

        # Времена основных этапов.
        cv2.putText(
            frame,
            f"frame {state.t_frame_ms:5.1f}ms  vo {state.t_vo_ms:4.1f}  app {state.t_app_ms:5.1f}  str {state.t_struct_ms:5.1f}",
            (x0 + 12, cy), font, 0.42, (180, 200, 255), 1, cv2.LINE_AA,
        )

        # Мини-карта траектории справа сверху.
        if state.enu_xy_history:
            HUDDrawer._draw_minimap(frame, state.enu_xy_history, w, state.sigma_pos_m)

        return frame

    @staticmethod
    def _draw_minimap(
        frame: np.ndarray, history: list, frame_w: int, sigma_m: float,
    ) -> None:
        """Мини-карта в правом верхнем углу: последние N сек траектории + σ-эллипс."""
        if len(history) < 2:
            return
        size = 220
        margin = 18
        x0 = frame_w - size - margin
        y0 = margin
        # фон
        roi = frame[y0:y0 + size, x0:x0 + size]
        bg = np.zeros_like(roi)
        cv2.rectangle(bg, (0, 0), (size, size), (28, 28, 28), -1)
        frame[y0:y0 + size, x0:x0 + size] = cv2.addWeighted(bg, 0.7, roi, 0.3, 0)
        cv2.rectangle(frame, (x0, y0), (x0 + size, y0 + size), (90, 90, 90), 1)

        pts = np.array(history, dtype=np.float64)
        cur_x, cur_y = pts[-1]
        # Центрируем на текущей позиции
        rel = pts - np.array([cur_x, cur_y])
        # Находим расстояние максимально удалённой точки
        max_r = float(np.max(np.linalg.norm(rel, axis=1)))
        scale = (0.45 * size) / max(max_r, 50.0)  # минимум 50м показываем
        # ENU x = east → пиксельный x; ENU y = north → -пиксельный y
        cx = x0 + size // 2
        cy = y0 + size // 2
        proj = rel.copy()
        proj[:, 0] = cx + proj[:, 0] * scale
        proj[:, 1] = cy - proj[:, 1] * scale
        # рисуем линию траектории
        for i in range(1, len(proj)):
            p0 = (int(proj[i - 1, 0]), int(proj[i - 1, 1]))
            p1 = (int(proj[i, 0]), int(proj[i, 1]))
            cv2.line(frame, p0, p1, (90, 220, 90), 1, cv2.LINE_AA)
        # текущая точка
        cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
        # σ-эллипс (круг при 1D σ)
        if np.isfinite(sigma_m) and sigma_m > 0:
            r = max(2, int(sigma_m * scale))
            r = min(r, size // 2 - 4)
            cv2.circle(frame, (cx, cy), r, (60, 200, 240), 1, cv2.LINE_AA)
        # масштабная линейка
        bar_m = 100.0 if max_r < 500 else 500.0
        bar_px = int(bar_m * scale)
        cv2.line(frame, (x0 + 10, y0 + size - 12),
                 (x0 + 10 + bar_px, y0 + size - 12), (200, 200, 200), 2)
        cv2.putText(frame, f"{int(bar_m)}m", (x0 + 10, y0 + size - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

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
        mode_text = "[КЛЮЧЕВОЙ КАДР + ОПТИЧЕСКИЙ ПОТОК]" if is_ai else "[КЛАССИЧЕСКОЕ СОПОСТАВЛЕНИЕ]"
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
