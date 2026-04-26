"""
Модуль для обработки видео и извлечения метаданных из GoPro видеофайлов.

Функциональность:
- Анализ видеофайлов (разрешение, FPS, продолжительность)
- Извлечение метаданных (GPS, высота, датчики)
- Геореференцирование кадров
- Сегментация изображений
"""

import cv2
import numpy as np
import pandas as pd
import subprocess
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

try:
    from .telemetry import TelemetryExtractor, GeoPosition
except ImportError:
    # Фоллбэк, если запуск идет не как пакет
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from telemetry import TelemetryExtractor, GeoPosition


@dataclass
class VideoInfo:
    """Информация о видеофайле"""
    filename: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float
    
    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"
    
    @property
    def duration_min(self) -> float:
        return self.duration_sec / 60


class VideoProcessor:
    """Класс для обработки видеофайлов и анализа данных"""
    
    def __init__(self, video_path: str, default_geo: Optional[Tuple[float, float, float]] = None):
        """
        Инициализирует процессор видео
        
        Args:
            video_path: путь к видеофайлу
            default_geo: (lat, lon, alt) - стартовые координаты самолёта для фоллбэка, если телеметрии в кадрах нет.
        """
        self.video_path = Path(video_path)
        self.info = self._analyze_video()
        
        # Интеграция телеметрии:
        self.telemetry = TelemetryExtractor(str(self.video_path), default_pos=default_geo)
        self.has_telemetry = self.telemetry.check_telemetry_stream()
        print(f"[{self.video_path.name}] Телеметрия GPMF найдена: {self.has_telemetry}")
    
    def _analyze_video(self) -> VideoInfo:
        """Анализирует видеофайл и извлекает основную информацию"""
        cap = cv2.VideoCapture(str(self.video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Не удается открыть видеофайл: {self.video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return VideoInfo(
            filename=self.video_path.name,
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            duration_sec=duration
        )
    
    def extract_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Извлекает кадр из видео по номеру
        
        Args:
            frame_number: номер кадра (0-индексирован)
            
        Returns:
            Кадр в формате BGR или None если ошибка
        """
        cap = cv2.VideoCapture(str(self.video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        return frame if ret else None
    
    def extract_frames_sample(self, sample_rate: int = 30) -> List[np.ndarray]:
        """
        Извлекает каждый N-ый кадр из видео
        
        Args:
            sample_rate: каждый N-ый кадр
            
        Returns:
            Список кадров
        """
        cap = cv2.VideoCapture(str(self.video_path))
        frames = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                frames.append(frame)
            
            frame_idx += 1
        
        cap.release()
        return frames
    
    def extract_metadata(self) -> Optional[Dict]:
        """
        Извлекает метаданные из видеофайла с помощью ffprobe
        
        Returns:
            Словарь с метаданными или None если ошибка
        """
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(self.video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                return json.loads(result.stdout)
        except Exception as e:
            print(f"Ошибка при извлечении метаданных: {e}")
        
        return None
    
    def georeference_frames(self, start_lat: float, start_lon: float, 
                           start_height: float, sample_rate: int = 30) -> pd.DataFrame:
        """
        Геореференцирует кадры видео
        
        Args:
            start_lat: начальная широта
            start_lon: начальная долгота
            start_height: начальная высота (метры)
            sample_rate: каждый N-ый кадр
            
        Returns:
            DataFrame с геореференцированными кадрами
        """
        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames_data = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                time_sec = frame_idx / fps
                
                frames_data.append({
                    'frame_id': frame_idx,
                    'time_sec': time_sec,
                    'latitude': start_lat,
                    'longitude': start_lon,
                    'height_m': start_height,
                    'frame_shape': frame.shape
                })
            
            frame_idx += 1
        
        cap.release()
        return pd.DataFrame(frames_data)


class ImageSegmenter:
    """Класс для сегментации изображений"""
    
    @staticmethod
    def segment_by_color_range(frame: np.ndarray, lower_hsv: np.ndarray, 
                               upper_hsv: np.ndarray) -> np.ndarray:
        """Сегментирует изображение по диапазону цветов в HSV"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        return mask
    
    @staticmethod
    def segment_by_threshold(frame: np.ndarray, threshold: int = 127) -> np.ndarray:
        """Сегментирует изображение по порогу яркости"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return binary
    
    @staticmethod
    def segment_by_edges(frame: np.ndarray, canny_low: int = 50, 
                        canny_high: int = 150) -> np.ndarray:
        """Сегментирует изображение по контурам (Canny)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, canny_low, canny_high)
        return edges
    
    @staticmethod
    def segment_sky_ground(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Сегментирует небо и землю
        
        Returns:
            Кортеж (маска_неба, маска_земли)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Диапазон для неба (синие и голубые цвета)
        lower_sky = np.array([90, 50, 50])
        upper_sky = np.array([130, 255, 255])
        
        sky_mask = cv2.inRange(hsv, lower_sky, upper_sky)
        ground_mask = cv2.bitwise_not(sky_mask)
        
        return sky_mask, ground_mask
    
    @staticmethod
    def analyze_objects(frame: np.ndarray, mask: np.ndarray, 
                       min_area: int = 100) -> Tuple[List[Dict], np.ndarray, List]:
        """
        Анализирует объекты на основе маски
        
        Returns:
            Кортеж (список_объектов, очищенная_маска, контуры)
        """
        # Морфологические операции
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
        
        # Поиск контуров
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                objects.append({
                    'x': x, 'y': y,
                    'width': w, 'height': h,
                    'area': area,
                    'perimeter': cv2.arcLength(cnt, True)
                })
        
        return objects, mask_cleaned, contours
