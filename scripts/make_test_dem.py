"""Сгенерировать небольшой синтетический GeoTIFF DEM для проверки этапа 2.1.

Код lookup в src/dem_lookup.py не зависит от источника: подойдёт любой
EPSG:4326 GeoTIFF. Для проверки критерия нужен DEM, где каждая высота
известна *аналитически*, чтобы можно было сравнить
``DemLookup.elevation(lat, lon)`` с ground truth и доказать, что билинейная
интерполяция работает как ожидается.

Форма рельефа (единицы: метры MSL):

    z(lat, lon) = base + slope_lat * (lat - lat0) * 111111
                       + slope_lon * (lon - lon0) * 111111 * cos(lat0)
                       + hill * exp(-((lat - lat_h)^2 + (lon - lon_h)^2) / sigma^2)

Имитирует рельеф центральной России (Коломна / долина Оки): базовая высота
около 130 м, мягкий подъём к северу и один гауссов холм в середине bbox
миссии. Синтетический DEM достигает ~210 м на вершине холма, поэтому
``height_AGL`` при крейсерской высоте 750 м меняется от ~540 м над холмом
до ~620 м в низинах. Именно такая вариация нужна VO на optical flow из
этапа 2.2, чтобы переводить пиксельные скорости в метрические.

Разрешение: 3 угловые секунды (~90 м). Это сделано намеренно: совпадает с
шагом сетки SRTM v4.1, файл достаточно лёгкий для репозитория (~1 МБ), а сетка
достаточно грубая, чтобы билинейная интерполяция давала заметно *непрерывную*
кривую AGL вдоль траектории. При nearest-neighbour был бы виден ступенчатый
артефакт. Реальный SRTM 1-arc-sec ведёт себя так же.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


# Описание рельефа вынесено на уровень модуля, чтобы проверочный скрипт мог
# импортировать его и сверять lookup с тем же аналитическим полем.
BASE_MSL = 130.0
SLOPE_NORTH_PER_M = 0.0008      # +0.08 м высоты на метр движения к северу
SLOPE_EAST_PER_M = -0.0003      # лёгкий уклон вниз к востоку, в сторону Оки
HILL_LAT = 55.25
HILL_LON = 38.35
HILL_HEIGHT = 80.0
HILL_SIGMA_DEG = 0.06           # гауссиан шириной ~6.6 км


def expected_elevation(lat: float, lon: float, lat0: float, lon0: float) -> float:
    """Аналитический ground truth: должен точно совпадать с тем, что растеризуем."""
    cos_lat0 = np.cos(np.deg2rad(lat0))
    d_north_m = (lat - lat0) * 111111.0
    d_east_m = (lon - lon0) * 111111.0 * cos_lat0
    base = BASE_MSL + SLOPE_NORTH_PER_M * d_north_m + SLOPE_EAST_PER_M * d_east_m
    dlat = lat - HILL_LAT
    dlon = lon - HILL_LON
    hill = HILL_HEIGHT * np.exp(-(dlat * dlat + dlon * dlon) / (HILL_SIGMA_DEG ** 2))
    return float(base + hill)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--west", type=float, default=38.0)
    p.add_argument("--south", type=float, default=55.0)
    p.add_argument("--east", type=float, default=38.7)
    p.add_argument("--north", type=float, default=55.5)
    p.add_argument("--arcsec", type=float, default=3.0,
                   help="cell size in arc-seconds (3 ~ 90 m, 1 ~ 30 m)")
    p.add_argument("--output", type=Path, default=Path("data/dem/test_synthetic.tif"))
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    cell_deg = args.arcsec / 3600.0
    width = int(round((args.east - args.west) / cell_deg))
    height = int(round((args.north - args.south) / cell_deg))

    # Соглашение по центрам пикселей: пиксель (0,0) покрывает северо-западный
    # угол, а его центр лежит на пол-ячейки юго-восточнее (west, north).
    # affine в rasterio использует координаты углов, поэтому задаём верхний
    # левый край как (west, north), а остальное оставляем from_origin.
    transform = from_origin(args.west, args.north, cell_deg, cell_deg)

    # Заранее считаем сетки lat/lon центров пикселей, чтобы аналитическое поле
    # и растеризованный канал совпадали до точности float — на это опирается
    # проверочный скрипт.
    lats = args.north - (np.arange(height) + 0.5) * cell_deg
    lons = args.west + (np.arange(width) + 0.5) * cell_deg
    lat0, lon0 = args.south, args.west

    cos_lat0 = np.cos(np.deg2rad(lat0))
    d_north_m = (lats - lat0) * 111111.0
    d_east_m = (lons - lon0) * 111111.0 * cos_lat0
    z_base = BASE_MSL + SLOPE_NORTH_PER_M * d_north_m[:, None] + SLOPE_EAST_PER_M * d_east_m[None, :]
    dlat = lats[:, None] - HILL_LAT
    dlon = lons[None, :] - HILL_LON
    hill = HILL_HEIGHT * np.exp(-(dlat * dlat + dlon * dlon) / (HILL_SIGMA_DEG ** 2))
    z = (z_base + hill).astype(np.float32)

    print(f"[dem] bbox=[{args.west}, {args.south}, {args.east}, {args.north}]")
    print(f"[dem] cell={args.arcsec}\" ({cell_deg:.6f} deg)  shape={(height, width)}")
    print(f"[dem] elevation range: [{z.min():.1f}, {z.max():.1f}] m MSL")

    with rasterio.open(
        args.output, "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        compress="deflate",
        predictor=3,
    ) as dst:
        dst.write(z, 1)
    print(f"[dem] wrote {args.output}  ({args.output.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
