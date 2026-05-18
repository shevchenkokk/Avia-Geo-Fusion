"""Конвертировать SRTM HGT-тайл 1 arc-second в GeoTIFF EPSG:4326.

Формат HGT — это сырая big-endian int16 сетка 3601×3601 (1-arc-sec) или
1201×1201 (3-arc-sec), покрывающая одну ячейку 1°×1°. Имя файла кодирует
юго-западный угол: ``N55E038.hgt`` означает lat ∈ [55°, 56°],
lon ∈ [38°, 39°]. Строка 0 — северная граница, последняя строка — южная.

Это вынесено в отдельный одноразовый скрипт, а не спрятано внутрь
``DemLookup``: lookup должен оставаться независимым от источника (любой
EPSG:4326 GeoTIFF), а предварительная конвертация HGT в GeoTIFF сохраняет этот
контракт чистым.

Пример::

    python scripts/srtm_hgt_to_geotiff.py data/dem/N55E038.hgt \\
        --out data/dem/srtm_n55_e038.tif
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_hgt_corner(name: str) -> tuple[int, int]:
    """Разобрать ``N55E038`` -> (lat=55, lon=38). Возвращает юго-западный угол."""
    m = re.match(r"([NS])(\d{2})([EW])(\d{3})", name.upper())
    if m is None:
        raise ValueError(f"unrecognised HGT name: {name}")
    ns, lat, ew, lon = m.groups()
    lat = int(lat); lon = int(lon)
    if ns == "S":
        lat = -lat
    if ew == "W":
        lon = -lon
    return lat, lon


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("hgt", type=Path, help="path to the .hgt file")
    p.add_argument("--out", type=Path, required=True, help="output GeoTIFF path")
    p.add_argument("--nodata", type=int, default=-32768,
                   help="SRTM void value (default -32768, the spec sentinel)")
    args = p.parse_args()

    if not args.hgt.exists():
        raise SystemExit(f"hgt not found: {args.hgt}")

    sw_lat, sw_lon = parse_hgt_corner(args.hgt.stem)

    raw = np.fromfile(args.hgt, dtype=">i2")  # big-endian int16
    n_pixels = raw.size
    side = int(round(np.sqrt(n_pixels)))
    if side * side != n_pixels or side not in (1201, 3601):
        raise SystemExit(
            f"unexpected HGT size {n_pixels} -> {side}x{side}; "
            f"expected 1201x1201 (3-arc-sec) or 3601x3601 (1-arc-sec)"
        )
    band = raw.reshape(side, side).astype(np.int16)

    # У SRTM углы — это ЦЕНТРЫ ПИКСЕЛЕЙ, поэтому тайл 3601×3601 покрывает
    # ровно 1° по каждой оси с размером пикселя 1/3600. Начало transform —
    # центр верхнего левого пикселя, а rasterio.from_origin ждёт верхний левый
    # УГОЛ, поэтому сдвигаем на полпикселя.
    pixel_deg = 1.0 / (side - 1)
    upper_left_lon = sw_lon - 0.5 * pixel_deg
    upper_left_lat = (sw_lat + 1.0) + 0.5 * pixel_deg
    transform = from_origin(upper_left_lon, upper_left_lat, pixel_deg, pixel_deg)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        args.out, "w",
        driver="GTiff", height=side, width=side, count=1,
        dtype=band.dtype, crs="EPSG:4326",
        transform=transform, nodata=args.nodata,
        compress="DEFLATE",
    ) as dst:
        dst.write(band, 1)

    valid = band[band != args.nodata]
    print(f"[hgt] in       : {args.hgt}  ({side}×{side})")
    print(f"[hgt] sw corner: ({sw_lat}, {sw_lon}) -- ({sw_lat+1}, {sw_lon+1})")
    print(f"[hgt] elev     : min={valid.min()}m  max={valid.max()}m  "
          f"mean={float(valid.mean()):.1f}m  voids={int((band == args.nodata).sum())}")
    print(f"[hgt] out      : {args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
