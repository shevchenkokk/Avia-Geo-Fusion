"""
Модуль для работы с геолокацией и метаданными видеофайлов.
Включает парсинг телеметрии (если она есть) и создание опорных данных (если ее нет).
"""

import subprocess
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GeoPosition:
    latitude: float
    longitude: float
    altitude: Optional[float] = None
    heading: Optional[float] = None  # yaw
    pitch: Optional[float] = None
    roll: Optional[float] = None


class TelemetryExtractor:
    """Извлекает GPS и IMU данные из видеофайлов, обеспечивает фоллбэки."""

    def __init__(self, video_path: str, default_pos: Optional[Tuple[float, float, float]] = None):
        """
        :param video_path: Путь к видео (GoPro MP4).
        :param default_pos: Кортеж (lat, lon, alt) по умолчанию, если телеметрии нет.
        """
        self.video_path = Path(video_path)
        self.default_pos = default_pos
        self.has_telemetry = False

    def check_telemetry_stream(self) -> bool:
        """Проверяет наличие потока данных GPMF/Metadata через ffprobe."""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_streams', str(self.video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            
            for stream in info.get('streams', []):
                # Ищем потоки данных (обычно type: data, codec_tag_string: gpmd)
                if stream.get('codec_type') == 'data' and stream.get('codec_tag_string') == 'gpmd':
                    self.has_telemetry = True
                    return True
            return False
        except Exception as e:
            logger.warning(f"Ошибка проверки потоков ffprobe: {e}")
            return False

    def get_position_at_frame(self, frame_time_sec: float) -> GeoPosition:
        """
        Возвращает гео-позицию для конкретного момента времени.
        Если телеметрии нет, использует интерполяцию или дефолтные значения.
        """
        # Если телеметрии нет или мы не смогли ее прочитать:
        if not self.has_telemetry and self.default_pos:
            logger.info("Используются стартовые (дефолтные) координаты: телеметрия недоступна.")
            return GeoPosition(
                latitude=self.default_pos[0],
                longitude=self.default_pos[1],
                altitude=self.default_pos[2],
                heading=0.0,
                pitch=-30.0,  # Предполагаем, что камера смотрит немного вниз
                roll=0.0
            )
            
        elif not self.has_telemetry:
            # Если вообще ничего нет
            raise ValueError("Нет телеметрии в видеофайле и не заданы стартовые координаты (default_pos).")
            
        # TODO: Реализовать полноценный парсинг GPMF потока (через gopro-telemetry или ffmpeg).
        # Для диплома пока возвращаем заглушку, которая будет заменена на реальный парсер.
        logger.debug(f"Чтение GPMF на {frame_time_sec} сек... (Заглушка)")
        return GeoPosition(
            latitude=self.default_pos[0] if self.default_pos else 0.0,
            longitude=self.default_pos[1] if self.default_pos else 0.0,
            altitude=self.default_pos[2] if self.default_pos else 0.0,
        )
