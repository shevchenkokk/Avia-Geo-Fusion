"""Маска снега для более устойчивого сезонного матчинга.

Разрыв между сезонами (зимнее видео с самолёта против летних/осенних
спутниковых тайлов) делает сопоставление по самому снегу ненадёжным:
это большие области почти без текстуры, где матчер либо не находит признаки,
либо собирает случайные согласованные совпадения. Между сезонами стабильнее:

  * дороги (асфальт или грунт, часто заметны через снег)
  * группы деревьев и границы леса
  * застройка
  * границы полей: заборы, посадки, канавы и ручьи

Сама снежная поверхность нестабильна: зимой она почти без текстуры, а на
летнем спутниковом снимке становится зелёной, коричневой или жёлтой.
Удаление снежных пикселей заставляет матчер работать по более стабильным
объектам.

Детектор специально сделан простым и консервативным: один порог в HSV вместо
обученной сегментации:

    snow = (V > v_threshold) AND (S < s_threshold)

Значения по умолчанию подобраны под дневную GoPro-съёмку над снежными полями
в центральной России. Яркий снег на солнце даёт V > 0.9 при S < 0.10;
снег в тени падает до V ~ 0.7, но S обычно остаётся < 0.15. Порог балансирует
между ложными срабатываниями и пропущенным снегом. Смещение сделано в сторону
ложных срабатываний: потерять несколько совпадений дешевле, чем пустить шум
от снега в матчер.
"""

from __future__ import annotations

import cv2
import numpy as np


def detect_snow_mask(
    img_bgr: np.ndarray,
    v_threshold: float = 0.80,
    s_threshold: float = 0.15,
    morph_kernel: int = 5,
) -> np.ndarray:
    """Вернуть uint8-маску 0/255, где 255 означает снег.

    Оба порога нормированы в [0, 1]. ``morph_kernel`` больше нуля закрывает
    мелкие дырки внутри снежных областей и убирает одиночный шум, делая
    границы маски чище. Значение 0 отключает морфологию.
    """
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[:, :, 1] / 255.0
    v = hsv[:, :, 2] / 255.0
    snow = ((v > v_threshold) & (s < s_threshold)).astype(np.uint8) * 255
    if morph_kernel > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        snow = cv2.morphologyEx(snow, cv2.MORPH_CLOSE, kernel)
    return snow


def combine_with_aircraft_mask(
    snow_mask: np.ndarray, aircraft_mask: np.ndarray | None
) -> np.ndarray:
    """Объединить снег и запретную область самолёта в одну маску через OR.

    Возвращает uint8-маску 0/255, где 255 отмечает пиксели, которые матчер
    не должен использовать. Параметр ``aircraft_mask`` у матчера принимает
    ровно такой формат.
    """
    if aircraft_mask is None:
        return snow_mask
    if snow_mask.shape[:2] != aircraft_mask.shape[:2]:
        aircraft_mask = cv2.resize(
            aircraft_mask, (snow_mask.shape[1], snow_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return cv2.bitwise_or(snow_mask, aircraft_mask)
