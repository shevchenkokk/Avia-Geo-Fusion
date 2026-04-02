"""Generate a small synthetic GeoTIFF DEM for Stage 2.1 verification.

The lookup code in src/dem_lookup.py is source-agnostic — any
EPSG:4326 GeoTIFF works. For criterion verification we want a DEM
where every elevation is *analytically* known, so we can compare
``DemLookup.elevation(lat, lon)`` to ground truth and prove the
bilinear interpolation does what it claims.

Terrain shape (units: metres, MSL):

    z(lat, lon) = base + slope_lat * (lat - lat0) * 111111
                       + slope_lon * (lon - lon0) * 111111 * cos(lat0)
                       + hill * exp(-((lat - lat_h)^2 + (lon - lon_h)^2) / sigma^2)

Mimics the kind of terrain we'd see over central Russia (Kolomna /
Oka valley): ~130 m base, gentle north-rising slope, a single Gaussian
hill in the middle of the mission bbox. The synthetic DEM reaches
~210 m at the hill peak, so ``height_AGL`` at cruise altitude 750 m
varies between ~540 m (over the hill) and ~620 m (in the valleys) —
the kind of variation Stage 2.2's optical-flow VO needs to translate
pixel velocities into metric speeds.

Resolution: 3 arc-seconds (~90 m). That's deliberate — it matches
SRTM v4.1 grid spacing, is light enough to commit (~1 MB), and is
coarse enough that bilinear interpolation produces a visibly
*continuous* AGL curve along a flight path (cf. the step-edge
artifact you'd see with nearest-neighbour). Real SRTM 1-arc-sec
behaves identically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


# Terrain definition. Kept module-level so the verification script
# can import it and check the lookup against the same analytic field.
BASE_MSL = 130.0
SLOPE_NORTH_PER_M = 0.0008      # +0.08 m elevation per metre of north travel
SLOPE_EAST_PER_M = -0.0003      # slight downhill to the east (toward Oka)
HILL_LAT = 55.25
HILL_LON = 38.35
HILL_HEIGHT = 80.0
HILL_SIGMA_DEG = 0.06           # ~6.6 km wide gaussian


def expected_elevation(lat: float, lon: float, lat0: float, lon0: float) -> float:
    """Analytic ground truth — must match exactly what we rasterise."""
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

    # Pixel-CENTRE convention: pixel (0,0) covers the NW corner; its
    # centre is half a cell south-east of (west, north). rasterio's
    # affine uses corner coordinates, so we set the top-left edge to
    # (west, north) and let from_origin handle the rest.
    transform = from_origin(args.west, args.north, cell_deg, cell_deg)

    # Pre-compute pixel-centre lat/lon grids; that way the analytic
    # field and the rasterised band agree to floating-point precision,
    # which is what the verification script relies on.
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
