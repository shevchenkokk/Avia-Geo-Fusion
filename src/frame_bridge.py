"""Переход ENU <-> WGS84 в приближении плоской Земли.

Фильтр состояния (src/ekf.py) работает в локальной декартовой системе
East-North-Up (ENU), привязанной к одной точке старта миссии. Все map-fix
приходят в WGS84 (lat, lon), поэтому перед подачей в фильтр их нужно поднять
в ENU. Обратно, позиция фильтра переводится из ENU в WGS84 при запросе тайлов
или DEM.

Для области полёта порядка 50 км плоское приближение даёт ошибку меньше
метра — заметно ниже ожидаемой точности матчера и вертикальной точности DEM.
Если будущая миссия вырастет за ~200 км, правильнее будет перейти на мост ENU
через ECEF с эллипсоидальными lat/lon. Пока оставляем простой вариант.

Единицы: метры для ENU, десятичные градусы для lat/lon, метры MSL для высоты.
"""

from __future__ import annotations

from dataclasses import dataclass

import math
import numpy as np


# Сфера вместо эллипсоида WGS84: погрешность <0.1% на 55°N — несущественна для наших масштабов.
# Константа 111 320 м/° для широты; масштаб долготы умножается на cos(lat) в вызывающем коде.
_METERS_PER_DEG_LAT = 111320.0


@dataclass(frozen=True)
class FrameBridge:
    """Привязка к точке (lat0, lon0, alt0_msl).

    Система ENU: x_e = восток, y_n = север, z_u = вверх. Начало координат
    лежит на земле в точке привязки, то есть ``z_u = 0`` соответствует высоте
    MSL этой точки. Высота самолёта в ENU поэтому равна
    ``alt_msl - alt0_msl``.
    """

    lat0: float
    lon0: float
    alt0_msl: float

    @property
    def m_per_deg_lat(self) -> float:
        return _METERS_PER_DEG_LAT

    @property
    def m_per_deg_lon(self) -> float:
        return _METERS_PER_DEG_LAT * math.cos(math.radians(self.lat0))

    def wgs84_to_enu(
        self, lat: float, lon: float, alt_msl: float
    ) -> tuple[float, float, float]:
        x_e = (lon - self.lon0) * self.m_per_deg_lon
        y_n = (lat - self.lat0) * self.m_per_deg_lat
        z_u = alt_msl - self.alt0_msl
        return float(x_e), float(y_n), float(z_u)

    def enu_to_wgs84(
        self, x_e: float, y_n: float, z_u: float = 0.0
    ) -> tuple[float, float, float]:
        lat = self.lat0 + y_n / self.m_per_deg_lat
        lon = self.lon0 + x_e / self.m_per_deg_lon
        alt_msl = self.alt0_msl + z_u
        return float(lat), float(lon), float(alt_msl)

    def cov_enu_to_wgs84(self, cov_enu: np.ndarray) -> np.ndarray:
        """Approximate position covariance lifted into (lat, lon) units (deg).

        Diagonal scaling only — the 2x2 (E, N) block is rescaled by
        (1/m_per_deg_lon, 1/m_per_deg_lat). Cross terms preserved.
        """
        c = np.asarray(cov_enu, dtype=np.float64).copy()
        s = np.array([1.0 / self.m_per_deg_lon, 1.0 / self.m_per_deg_lat])
        c[:2, :2] = (c[:2, :2] * s[:, None]) * s[None, :]
        return c
