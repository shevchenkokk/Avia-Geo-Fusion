"""Этап 2.1: запрос высоты рельефа из DEM с билинейной интерполяцией.

DEM (Digital Elevation Model) выдаёт высоту рельефа над уровнем моря (MSL)
по (lat, lon). Разность altitude_MSL − terrain_MSL = AGL — величина, нужная
для VO (этап 2.2), чтобы перевести пиксельные скорости в метрические.

Требования к файлу DEM: GeoTIFF в EPSG:4326 (SRTM 1-arc-sec или ALOS AW3D30).
При запросе вне растра возвращается None, а не дефолтная высота, потому что
неверный AGL молча портит всё последующее.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio


class DemLookup:
    """Читает один GeoTIFF-DEM и отвечает на запросы (lat, lon) → высота.

    Растр загружается в память один раз при создании объекта.
    """

    def __init__(self, dem_path: Path | str):
        self.dem_path = Path(dem_path)
        if not self.dem_path.exists():
            raise FileNotFoundError(f"DEM not found: {self.dem_path}")

        with rasterio.open(self.dem_path) as ds:
            if ds.count < 1:
                raise ValueError(f"DEM has no bands: {self.dem_path}")
            # float32 — чтобы nodata-сентинел int16 (-32768) не вызвал переполнения при интерполяции.
            self._band = ds.read(1).astype(np.float32)
            self._transform = ds.transform
            self._inv_transform = ~ds.transform
            self._crs = ds.crs
            self._nodata = ds.nodata
            self._bounds = ds.bounds
            self._height, self._width = self._band.shape

        if self._crs is None:
            raise ValueError(f"DEM has no CRS: {self.dem_path}")
        # Поддерживаем только географические DEM (EPSG:4326).
        # Перепроецирование на лету замедлит покадровые запросы; при необходимости
        # UTM-DEM добавить однократный warp-шаг.
        if self._crs.to_epsg() != 4326:
            raise ValueError(
                f"DEM CRS must be EPSG:4326 (geographic); got {self._crs}. "
                f"Reproject the source raster with gdalwarp -t_srs EPSG:4326."
            )

        if self._nodata is not None:
            self._nodata_mask = self._band == self._nodata
        else:
            self._nodata_mask = np.zeros_like(self._band, dtype=bool)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(west_lon, south_lat, east_lon, north_lat) растра."""
        return (
            float(self._bounds.left),
            float(self._bounds.bottom),
            float(self._bounds.right),
            float(self._bounds.top),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self._band.shape

    def contains(self, lat: float, lon: float) -> bool:
        west, south, east, north = self.bounds
        return west <= lon <= east and south <= lat <= north

    def elevation(self, lat: float, lon: float) -> Optional[float]:
        """Высота рельефа (м MSL) с билинейной интерполяцией.

        Возвращает None если точка за пределами растра или попадает в nodata.
        """
        if not self.contains(lat, lon):
            return None

        # rasterio: обратное affine-преобразование даёт дробные (col, row).
        col_f, row_f = self._inv_transform * (lon, lat)
        # Зажимаем, чтобы индекс +1 не вышел за границу массива на крайних точках.
        col_f = float(np.clip(col_f, 0.0, self._width - 1.000001))
        row_f = float(np.clip(row_f, 0.0, self._height - 1.000001))

        c0 = int(np.floor(col_f))
        r0 = int(np.floor(row_f))
        c1 = c0 + 1
        r1 = r0 + 1
        dc = col_f - c0
        dr = row_f - r0

        v00 = self._band[r0, c0]; v01 = self._band[r0, c1]
        v10 = self._band[r1, c0]; v11 = self._band[r1, c1]
        m00 = self._nodata_mask[r0, c0]; m01 = self._nodata_mask[r0, c1]
        m10 = self._nodata_mask[r1, c0]; m11 = self._nodata_mask[r1, c1]

        if m00 and m01 and m10 and m11:
            return None

        if not (m00 or m01 or m10 or m11):
            top = v00 * (1.0 - dc) + v01 * dc
            bot = v10 * (1.0 - dc) + v11 * dc
            return float(top * (1.0 - dr) + bot * dr)

        # Часть углов — nodata: усредняем по валидным.
        vals = [v for v, m in [(v00, m00), (v01, m01), (v10, m10), (v11, m11)] if not m]
        return float(np.mean(vals))

    def height_agl(
        self, lat: float, lon: float, altitude_msl: float
    ) -> Optional[float]:
        """AGL (м) = altitude_MSL − terrain_MSL.

        Отрицательный AGL допустим (посадка ниже разрешения DEM); решает вызывающий.
        """
        ground = self.elevation(lat, lon)
        if ground is None:
            return None
        return float(altitude_msl - ground)
