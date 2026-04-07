"""ENU <-> WGS84 frame bridge (flat-Earth approximation).

The state filter (src/ekf.py) operates in a *local* East-North-Up
(ENU) Cartesian frame anchored at a single mission origin. All map
fixes are reported in WGS84 (lat, lon) and need to be lifted into ENU
before being fed to the filter; conversely, the filter's position
state is lowered back to WGS84 when querying tiles or DEM.

For a flight bbox of ~50 km the flat-Earth approximation is good to
better than 1 m — well below the matcher's expected position
accuracy and the DEM's vertical accuracy. If a future mission grows
beyond ~200 km the right answer is to switch to a proper ECEF-based
ENU bridge with ellipsoidal lat/lon; for now we keep it simple.

All units: metres for ENU, decimal degrees for lat/lon, metres MSL
for altitude.
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
    """Anchored at (lat0, lon0, alt0_msl).

    ENU frame: x_e = east, y_n = north, z_u = up. The origin is on
    the ground at the anchor point — i.e. ``z_u = 0`` is at the
    anchor's MSL elevation. Aircraft altitude in ENU is therefore
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
