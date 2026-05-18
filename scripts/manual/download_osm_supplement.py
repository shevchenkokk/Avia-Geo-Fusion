"""Скачивает OSM water + buildings через Overpass API per-region и
сохраняет в `vectors/<region>/osm_water.parquet` и `osm_buildings.parquet`.
Нужно для Phase C step 2: дополнить Overture-маски классами, где Overture
неполон (мелкие водоёмы, индустриальные здания).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests
import geopandas as gpd
from shapely.geometry import Polygon, LineString, MultiPolygon


OVERPASS_URL = "https://overpass-api.de/api/interpreter"

WATER_QUERY = """
[out:json][timeout:120];
(
  way["natural"="water"]({s},{w},{n},{e});
  relation["natural"="water"]({s},{w},{n},{e});
  way["waterway"~"river|stream|canal|drain"]({s},{w},{n},{e});
  way["landuse"="reservoir"]({s},{w},{n},{e});
  way["landuse"="basin"]({s},{w},{n},{e});
);
out body geom;
"""

BUILDING_QUERY = """
[out:json][timeout:120];
(
  way["building"]({s},{w},{n},{e});
  relation["building"]({s},{w},{n},{e});
);
out body geom;
"""


HEADERS = {"User-Agent": "Avia-Geo-Fusion-OSM/1.0 (research)"}


def overpass(query: str, bbox: tuple[float, float, float, float], retries: int = 3):
    """bbox = (lon_min, lat_min, lon_max, lat_max) → (s,w,n,e) для Overpass."""
    w, s, e, n = bbox
    body = query.format(s=s, w=w, n=n, e=e)
    for attempt in range(retries):
        try:
            r = requests.post(OVERPASS_URL, data={"data": body},
                              headers=HEADERS, timeout=180)
            r.raise_for_status()
            return r.json()
        except Exception as ex:
            print(f"  attempt {attempt+1} failed: {ex}")
            if attempt < retries - 1:
                time.sleep(20)
    raise RuntimeError(f"Overpass failed after {retries} retries")


def parse_water(data) -> gpd.GeoDataFrame:
    """OSM water elements → GeoDataFrame с columns subtype, class, geometry."""
    rows = []
    for el in data.get("elements", []):
        if el.get("type") not in {"way", "relation"}:
            continue
        tags = el.get("tags", {}) or {}
        # Determine semantic class
        if tags.get("natural") == "water":
            cls = (tags.get("water") or "lake").lower()
        elif tags.get("waterway"):
            cls = "river"
        elif tags.get("landuse") in {"reservoir", "basin"}:
            cls = "reservoir"
        else:
            cls = "lake"

        geom = _element_to_geometry(el)
        if geom is None:
            continue
        rows.append({"subtype": "water", "class": cls, "geometry": geom})

    if not rows:
        return gpd.GeoDataFrame(columns=["subtype", "class", "geometry"], crs="EPSG:4326")
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def parse_buildings(data) -> gpd.GeoDataFrame:
    rows = []
    for el in data.get("elements", []):
        if el.get("type") not in {"way", "relation"}:
            continue
        tags = el.get("tags", {}) or {}
        if "building" not in tags:
            continue
        geom = _element_to_geometry(el)
        if geom is None or geom.is_empty:
            continue
        # Force polygon (skip non-closed ways)
        if geom.geom_type not in {"Polygon", "MultiPolygon"}:
            continue
        rows.append({"subtype": "building", "class": "building", "geometry": geom})

    if not rows:
        return gpd.GeoDataFrame(columns=["subtype", "class", "geometry"], crs="EPSG:4326")
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def _element_to_geometry(el):
    if el["type"] == "way":
        coords = [(node["lon"], node["lat"]) for node in el.get("geometry", [])]
        if len(coords) < 2:
            return None
        if len(coords) >= 4 and coords[0] == coords[-1]:
            try:
                return Polygon(coords)
            except Exception:
                return None
        return LineString(coords)
    if el["type"] == "relation":
        outer_coords = []
        for member in el.get("members", []):
            if member.get("role") not in {"outer", ""}:
                continue
            geom = member.get("geometry") or []
            ring = [(p["lon"], p["lat"]) for p in geom]
            if len(ring) >= 3:
                outer_coords.append(ring)
        if not outer_coords:
            return None
        try:
            polys = [Polygon(c) for c in outer_coords if len(c) >= 3]
            polys = [p for p in polys if p.is_valid]
            if not polys:
                return None
            if len(polys) == 1:
                return polys[0]
            return MultiPolygon(polys)
        except Exception:
            return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", default="data/overture_ru_dataset_starter/vectors")
    ap.add_argument("--sleep", type=float, default=3.0,
                    help="Pause between region requests (rate-limit safety)")
    ap.add_argument("--regions", nargs="*", default=None,
                    help="Subset of regions to fetch")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    regions = cfg["regions"]
    if args.regions:
        regions = [r for r in regions if r["id"] in set(args.regions)]

    for region in regions:
        rid = region["id"]
        bbox = tuple(region["bbox"])  # (lon_min, lat_min, lon_max, lat_max)
        out_dir = out_root / rid
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{rid}] bbox={bbox}")

        water_path = out_dir / "osm_water.parquet"
        if water_path.exists():
            print(f"  water: SKIP (exists)")
        else:
            t0 = time.time()
            data = overpass(WATER_QUERY, bbox)
            gdf = parse_water(data)
            gdf.to_parquet(water_path, index=False)
            print(f"  water: {len(gdf)} features in {time.time()-t0:.1f}s → {water_path.name}")
            time.sleep(args.sleep)

        bld_path = out_dir / "osm_buildings.parquet"
        if bld_path.exists():
            print(f"  buildings: SKIP (exists)")
        else:
            t0 = time.time()
            data = overpass(BUILDING_QUERY, bbox)
            gdf = parse_buildings(data)
            gdf.to_parquet(bld_path, index=False)
            print(f"  buildings: {len(gdf)} features in {time.time()-t0:.1f}s → {bld_path.name}")
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
