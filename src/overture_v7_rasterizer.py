"""Онлайн-растеризация векторных данных Overture в 7-классную маску.

Целевые классы: water | forest_edge | roads | field_boundary | built_up | snow | cloud.
``snow`` и ``cloud`` не генерируются здесь — они определяются per-frame
детекторами (snow_mask.py и obstruction_detector.py) в рантайме.

forest_edge и field_boundary выводятся морфологически из полигонов лесных/сельхоз
покрытий: (dilate − erode), чтобы получить граничную полосу. Это ключевой класс
для тайги: граница лес↔поле — основной структурный признак на 50%+ территории РФ.

Функции:
  load_region_layers(vectors_root, region_id) -> dict GeoDataFrame по слоям
  rasterize_bbox(bbox, layer_gdfs, out_size)  -> (H, W) uint8, значения ∈ {0..5}
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ID классов из src/class_schemas.py::OVERTURE_RU_7.
CID_BG = 0
CID_WATER = 1
CID_FOREST_EDGE = 2
CID_ROADS = 3
CID_FIELD_BOUNDARY = 4
CID_BUILT_UP = 5
CID_SNOW = 6  # только рантайм

# Порядок заливки: старший приоритет заполняется первым; низший — только пустые пиксели.
RASTER_PRIORITY = (CID_WATER, CID_FOREST_EDGE, CID_ROADS, CID_FIELD_BOUNDARY, CID_BUILT_UP)


# Маппинг vector feature → промежуточный ID класса.
# Смещение +100 отличает «прямые» классы от промежуточных (лес/поле),
# которые преобразуются в *_edge/*_boundary через морфологию после растеризации.
_FOREST_RAW_TAGS = {
    "land_cover:forest",
    "land_cover:wood",
}
_FARMLAND_RAW_TAGS = {
    "land_use:farmland",
    "land_use:agriculture",
    "land_use:crop",
    "land_use:cropland",
    "land_use:horticulture",
    "land_use:meadow",
}
_BUILT_UP_TAGS = {
    "land_use:residential",
    "land_use:commercial",
    "land_use:industrial",
    "land_use:retail",
    "land_use:education",
    "land_use:institutional",
}
_WATER_TAGS_PREFIX = "water:"
_TRANSPORTATION_PREFIX = "transportation:"


# Дороги — осевые линии; буферизуются до ширины проезжей части.
_ROAD_BUFFER_M = 4.0


def _classify_feature(layer_name: str, props: dict) -> int | None:
    """Маппинг векторного объекта на промежуточный ID класса.

    Возвращает:
        100 + CID_WATER     — водные объекты
        200                 — лесные полигоны (промежуточный)
        100 + CID_ROADS     — дороги
        300                 — сельхоз-полигоны (промежуточный)
        100 + CID_BUILT_UP  — застройка
        None                — объект не относится ни к одному классу.
    """
    if layer_name == "water":
        return 100 + CID_WATER

    if layer_name == "transportation":
        return 100 + CID_ROADS

    if layer_name in ("land_cover", "land_use"):
        for key in ("class", "subtype", "type", "kind", "landcover", "landuse"):
            v = props.get(key)
            if v is None:
                continue
            v_norm = str(v).lower()
            tag_full = f"{layer_name}:{v_norm}"
            if tag_full in _FOREST_RAW_TAGS:
                return 200
            if tag_full in _FARMLAND_RAW_TAGS:
                return 300
            if tag_full in _BUILT_UP_TAGS:
                return 100 + CID_BUILT_UP

    return None


def _meters_per_pixel(bbox: tuple[float, float, float, float],
                      out_size: tuple[int, int]) -> float:
    west, south, east, north = bbox
    out_w, out_h = out_size
    mid_lat = 0.5 * (south + north)
    mppx_lon = (east - west) * 111320.0 * math.cos(math.radians(mid_lat)) / max(out_w, 1)
    mppx_lat = (north - south) * 111320.0 / max(out_h, 1)
    return float(max(abs(mppx_lon), abs(mppx_lat)))


def _morph_edge(filled: np.ndarray, width_px: int) -> np.ndarray:
    """Граница бинарной маски (dilate − erode), толщиной ``width_px`` пикселей.

    Используется для получения forest_edge и field_boundary из полигонов.
    """
    if width_px < 1:
        width_px = 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width_px + 1, 2 * width_px + 1))
    dil = cv2.dilate(filled, k)
    ero = cv2.erode(filled, k)
    edge = cv2.subtract(dil, ero)
    return edge


def _read_layer(parquet_path: Path):
    import geopandas as gpd
    if not parquet_path.exists():
        return None
    try:
        gdf = gpd.read_parquet(parquet_path)
    except Exception as e:
        logger.warning("rasterize_v7: failed to read %s: %s", parquet_path, e)
        return None
    if gdf.empty:
        return None
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    else:
        gdf = gdf.to_crs(4326)
    return gdf


def load_region_layers(vectors_root: Path, region_id: str) -> dict:
    """Загружает все слои Overture для региона в словарь ``{layer_name: GeoDataFrame}``.

    Пропускает отсутствующие или пустые слои. Дороги буферизуются до ширины
    проезжей части в EPSG:3857, затем конвертируются обратно в EPSG:4326.
    """
    region_dir = Path(vectors_root) / region_id
    out: dict = {}
    for layer in ("water", "transportation", "land_cover", "land_use"):
        gdf = _read_layer(region_dir / f"{layer}.parquet")
        if gdf is None:
            continue
        if layer == "transportation":
            try:
                proj = gdf.to_crs(epsg=3857)
                proj["geometry"] = proj.geometry.buffer(_ROAD_BUFFER_M)
                gdf = proj.to_crs(epsg=4326)
            except Exception as e:
                logger.warning("rasterize_v7: road buffer failed: %s", e)
        out[layer] = gdf
    return out


def rasterize_bbox(
    bbox: tuple[float, float, float, float],
    layer_gdfs: dict,
    out_size: tuple[int, int] = (1024, 1024),
    forest_edge_width_m: float = 12.0,
    field_boundary_width_m: float = 8.0,
) -> np.ndarray:
    """Растеризует векторные данные Overture в 7-классную семантическую маску.

    Args:
        bbox: (west, south, east, north) в градусах.
        layer_gdfs: результат ``load_region_layers``.
        out_size: (W, H) выходной маски в пикселях.
        forest_edge_width_m: толщина полосы forest_edge (морфология, полная ширина).
        field_boundary_width_m: то же для границ сельхозполей.

    Returns:
        ``(H, W) uint8`` со значениями ∈ {0, 1, 2, 3, 4, 5}.
    """
    from rasterio.features import rasterize as rio_rasterize
    from rasterio.transform import from_bounds
    from shapely.geometry import box as shapely_box

    out_w, out_h = out_size
    west, south, east, north = bbox
    transform = from_bounds(west, south, east, north, out_w, out_h)
    tile_poly = shapely_box(west, south, east, north)

    mppx = _meters_per_pixel(bbox, out_size)
    forest_edge_px = max(1, int(round(forest_edge_width_m / mppx / 2)))
    field_edge_px = max(1, int(round(field_boundary_width_m / mppx / 2)))

    shapes_water: list = []
    shapes_forest_raw: list = []
    shapes_roads: list = []
    shapes_farm_raw: list = []
    shapes_builtup: list = []

    for layer_name, gdf in layer_gdfs.items():
        if gdf is None or len(gdf) == 0:
            continue
        clipped = gdf[gdf.geometry.intersects(tile_poly)]
        if clipped.empty:
            continue
        for _, row in clipped.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            try:
                geom = geom.intersection(tile_poly)
            except Exception:
                continue
            if geom is None or geom.is_empty:
                continue

            cls = _classify_feature(layer_name, row.to_dict())
            if cls is None:
                continue
            target_list = {
                100 + CID_WATER: shapes_water,
                200: shapes_forest_raw,
                100 + CID_ROADS: shapes_roads,
                300: shapes_farm_raw,
                100 + CID_BUILT_UP: shapes_builtup,
            }.get(cls)
            if target_list is None:
                continue
            target_list.append((geom, 1))

    def _rasterize_to_binary(shapes: list) -> np.ndarray:
        if not shapes:
            return np.zeros((out_h, out_w), dtype=np.uint8)
        return (rio_rasterize(
            shapes=shapes, out_shape=(out_h, out_w),
            fill=0, transform=transform, dtype="uint8",
            all_touched=False,
        ) * 255).astype(np.uint8)

    bin_water = _rasterize_to_binary(shapes_water)
    bin_forest_raw = _rasterize_to_binary(shapes_forest_raw)
    bin_roads = _rasterize_to_binary(shapes_roads)
    bin_farm_raw = _rasterize_to_binary(shapes_farm_raw)
    bin_builtup = _rasterize_to_binary(shapes_builtup)

    bin_forest_edge = _morph_edge(bin_forest_raw, forest_edge_px) if bin_forest_raw.any() else bin_forest_raw
    bin_field_boundary = _morph_edge(bin_farm_raw, field_edge_px) if bin_farm_raw.any() else bin_farm_raw

    final = np.zeros((out_h, out_w), dtype=np.uint8)
    for cls_id, layer in (
        (CID_WATER, bin_water),
        (CID_FOREST_EDGE, bin_forest_edge),
        (CID_ROADS, bin_roads),
        (CID_FIELD_BOUNDARY, bin_field_boundary),
        (CID_BUILT_UP, bin_builtup),
    ):
        mask = (layer > 0) & (final == 0)
        final[mask] = cls_id

    return final


def colorize_v7(mask: np.ndarray) -> np.ndarray:
    from src.class_schemas import OVERTURE_RU_7
    colors = OVERTURE_RU_7.colors
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cid, color in colors.items():
        out[mask == cid] = color
    return out
