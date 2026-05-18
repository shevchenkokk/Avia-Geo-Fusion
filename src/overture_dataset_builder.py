import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np
import requests
from tqdm import tqdm


@dataclass(frozen=True)
class TileId:
    z: int
    x: int
    y: int

    @property
    def key(self) -> str:
        return f"{self.z}_{self.x}_{self.y}"


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def tile_bbox_wgs84(tile: TileId) -> Tuple[float, float, float, float]:
    n = 2.0 ** tile.z
    lon_min = tile.x / n * 360.0 - 180.0
    lon_max = (tile.x + 1) / n * 360.0 - 180.0

    lat_max_rad = math.atan(math.sinh(math.pi * (1 - 2 * tile.y / n)))
    lat_min_rad = math.atan(math.sinh(math.pi * (1 - 2 * (tile.y + 1) / n)))
    lat_max = math.degrees(lat_max_rad)
    lat_min = math.degrees(lat_min_rad)

    return lon_min, lat_min, lon_max, lat_max


def enumerate_tiles(bbox: Iterable[float], zoom: int) -> List[TileId]:
    min_lon, min_lat, max_lon, max_lat = list(bbox)
    x0, y1 = deg2num(min_lat, min_lon, zoom)
    x1, y0 = deg2num(max_lat, max_lon, zoom)

    xs = range(min(x0, x1), max(x0, x1) + 1)
    ys = range(min(y0, y1), max(y0, y1) + 1)
    return [TileId(zoom, x, y) for y in ys for x in xs]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def image_hash(path: Path) -> str:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return ""
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    bits = img[:, 1:] > img[:, :-1]
    packed = np.packbits(bits.reshape(-1).astype(np.uint8))
    return packed.tobytes().hex()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def resolve_layer_path(region_dir: Path, layer_name: str) -> Path | None:
    for ext in (".parquet", ".geojson", ".gpkg", ".json"):
        candidate = region_dir / f"{layer_name}{ext}"
        if candidate.exists():
            return candidate
    return None


def maybe_run_overture_download(config: Dict[str, Any], region: Dict[str, Any], layer_name: str, output_path: Path) -> None:
    cmd_tmpls = config.get("overture_download_commands", {})
    template = cmd_tmpls.get(layer_name)
    if not template:
        return

    cli_name = config.get("overture_cli", "overturemaps")
    cli_path = shutil.which(cli_name)
    if cli_path is None:
        candidates = []
        # 1) bin рядом с текущим интерпретатором
        candidates.append(Path(sys.executable).resolve().parent / cli_name)
        # 2) активированный virtualenv
        venv_env = os.environ.get("VIRTUAL_ENV")
        if venv_env:
            candidates.append(Path(venv_env) / "bin" / cli_name)
        # 3) локальный .venv в корне репозитория
        repo_root = Path(__file__).resolve().parents[1]
        candidates.append(repo_root / ".venv" / "bin" / cli_name)

        for candidate in candidates:
            if candidate.exists() and os.access(candidate, os.X_OK):
                cli_path = str(candidate)
                break
    if cli_path is None:
        raise RuntimeError(
            "Не найден CLI overturemaps. Установите пакет в активное окружение: pip install overturemaps"
        )

    min_lon, min_lat, max_lon, max_lat = region["bbox"]
    cmd = template.format(
        overture_cli=cli_path,
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
        output=str(output_path),
        region_id=region["id"],
    )
    subprocess.run(cmd, shell=True, check=True)


def download_satellite_tiles(config: Dict[str, Any], root: Path) -> Path:
    zoom = int(config["zoom"])
    tile_size = int(config["tile_size"])
    timeout_sec = int(config.get("http_timeout_sec", 20))

    images_dir = root / "images" / "tiles"
    ensure_dir(images_dir)

    tile_url = config["tile_source"]["url_template"]
    headers = {"User-Agent": config["tile_source"].get("user_agent", "Avia-Geo-Fusion/1.0")}

    rows: List[Dict[str, Any]] = []
    session = requests.Session()

    for region in config["regions"]:
        tiles = enumerate_tiles(region["bbox"], zoom)
        region_name = region["id"]

        for tile in tqdm(tiles, desc=f"tiles:{region_name}"):
            out_name = f"{region_name}_{tile.key}.png"
            out_path = images_dir / out_name
            bbox = tile_bbox_wgs84(tile)

            if not out_path.exists():
                url = tile_url.format(z=tile.z, x=tile.x, y=tile.y)
                resp = session.get(url, timeout=timeout_sec, headers=headers)
                if resp.status_code != 200:
                    continue

                data = resp.content
                arr = np.frombuffer(data, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                if img.shape[0] != tile_size or img.shape[1] != tile_size:
                    img = cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(str(out_path), img)

            rows.append(
                {
                    "tile_id": f"{region_name}_{tile.key}",
                    "region_id": region_name,
                    "z": tile.z,
                    "x": tile.x,
                    "y": tile.y,
                    "bbox_wgs84": f"{bbox[0]:.8f},{bbox[1]:.8f},{bbox[2]:.8f},{bbox[3]:.8f}",
                    "image_path": str(out_path.relative_to(root)),
                    "image_source": config["tile_source"]["name"],
                }
            )

    meta_path = root / "meta_tiles.csv"
    ensure_dir(meta_path.parent)
    with meta_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    return meta_path


def _require_geo_stack() -> None:
    try:
        import geopandas  # noqa: F401
        import rasterio  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Нужны зависимости geopandas и rasterio. Установите: pip install geopandas rasterio"
        ) from exc


def _load_layer_as_gdf(path: Path):
    import geopandas as gpd

    suffix = path.suffix.lower()
    if suffix in {".geojson", ".json", ".gpkg"}:
        return gpd.read_file(path)
    if suffix == ".parquet":
        return gpd.read_parquet(path)
    raise ValueError(f"Неподдерживаемый формат слоя: {path}")


ROAD_BUFFER_M = {
    "motorway": 7.0,
    "trunk": 6.0,
    "primary": 5.0,
    "secondary": 4.0,
    "tertiary": 3.5,
    "residential": 3.0,
    "unclassified": 2.5,
    "living_street": 2.0,
    "service": 2.0,
    "track": 1.5,
    "path": 1.0,
    "driveway": 1.5,
    "parking_aisle": 2.0,
    "pedestrian": 1.5,
    "runway": 25.0,
    "taxiway": 12.0,
}
DEFAULT_ROAD_BUFFER_M = 2.5

# Из аэроперспективы это не «дороги»: тротуары, лестницы, велодорожки.
# Если их рисовать как class=4, в моздовских тайлах они забивают 25% пикселей
# и мешают модели учиться на реальных проездах.
DROP_ROAD_CLASSES = {"footway", "steps", "cycleway", "sidewalk"}

BUILDING_BUFFER_M = 1.5


def _utm_epsg_for_bbox(minx: float, miny: float, maxx: float, maxy: float) -> int:
    """Вернуть EPSG-код подходящей UTM зоны по центроиду bbox в WGS84."""
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    zone = int((cx + 180) // 6) + 1
    return (32600 if cy >= 0 else 32700) + zone


def _buffer_transportation_in_utm(gdf):
    """Class-aware буфер дорог в UTM. Дропает тротуары/лестницы.
    Вход и выход в EPSG:4326."""
    if gdf is None or gdf.empty:
        return gdf

    if "class" in gdf.columns:
        gdf = gdf[~gdf["class"].fillna("").str.lower().isin(DROP_ROAD_CLASSES)].copy()

    if gdf.empty:
        return gdf

    minx, miny, maxx, maxy = gdf.total_bounds
    utm_epsg = _utm_epsg_for_bbox(minx, miny, maxx, maxy)

    g_utm = gdf.to_crs(epsg=utm_epsg)
    if "class" in g_utm.columns:
        widths = g_utm["class"].fillna("").str.lower().map(ROAD_BUFFER_M).fillna(DEFAULT_ROAD_BUFFER_M)
    else:
        widths = [DEFAULT_ROAD_BUFFER_M] * len(g_utm)
    g_utm["geometry"] = [
        geom.buffer(float(w)) if geom is not None and not geom.is_empty else geom
        for geom, w in zip(g_utm.geometry, widths)
    ]
    return g_utm.to_crs(epsg=4326)


def _buffer_buildings_in_utm(gdf, buffer_m: float = BUILDING_BUFFER_M):
    """Раздуваем здания в реальных метрах через UTM, чтобы накрыть съехавшие крыши."""
    if gdf is None or gdf.empty:
        return gdf
    minx, miny, maxx, maxy = gdf.total_bounds
    utm_epsg = _utm_epsg_for_bbox(minx, miny, maxx, maxy)
    g_utm = gdf.to_crs(epsg=utm_epsg)
    g_utm["geometry"] = g_utm.geometry.buffer(float(buffer_m))
    return g_utm.to_crs(epsg=4326)


def _class_from_feature(layer_name: str, props: Dict[str, Any], mapping: Dict[str, int]) -> int:
    default_cls = int(mapping.get(f"{layer_name}:*", 0))

    for key in ("class", "subtype", "type", "kind", "landuse", "landcover"):
        value = props.get(key)
        if value is None:
            continue
        value_norm = str(value).lower()
        key_full = f"{layer_name}:{value_norm}"
        if key_full in mapping:
            return int(mapping[key_full])

        # Более точное правило: можно маппить конкретный атрибут слоя,
        # например transportation.class:residential.
        key_with_attr = f"{layer_name}.{key}:{value_norm}"
        if key_with_attr in mapping:
            return int(mapping[key_with_attr])

    return default_cls


def rasterize_masks(config: Dict[str, Any], root: Path) -> Path:
    _require_geo_stack()
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    from shapely.geometry import box

    tile_size = int(config["tile_size"])
    zoom = int(config["zoom"])

    mask_dir = root / "masks" / "tiles"
    vectors_root = root / "vectors"
    images_dir = root / "images" / "tiles"
    ensure_dir(mask_dir)

    class_mapping = {str(k): int(v) for k, v in config["class_mapping"].items()}
    layers = [layer["name"] for layer in config["overture_layers"]]

    rows: List[Dict[str, Any]] = []

    for region in config["regions"]:
        region_id = region["id"]
        region_vector_dir = vectors_root / region_id
        ensure_dir(region_vector_dir)

        layer_gdfs: Dict[str, Any] = {}
        for layer_name in layers:
            out_path = region_vector_dir / f"{layer_name}.parquet"
            if not resolve_layer_path(region_vector_dir, layer_name):
                maybe_run_overture_download(config, region, layer_name, out_path)

            layer_path = resolve_layer_path(region_vector_dir, layer_name)
            if layer_path is None:
                continue

            try:
                gdf = _load_layer_as_gdf(layer_path)
            except Exception:
                # Файл может быть поврежден после прерванной загрузки; удаляем и скачиваем повторно.
                try:
                    layer_path.unlink(missing_ok=True)
                except Exception:
                    pass

                maybe_run_overture_download(config, region, layer_name, out_path)
                layer_path = resolve_layer_path(region_vector_dir, layer_name)
                if layer_path is None:
                    continue

                gdf = _load_layer_as_gdf(layer_path)

            if gdf.empty:
                continue
            if gdf.crs is None:
                gdf = gdf.set_crs(4326)
            else:
                gdf = gdf.to_crs(4326)

            # --- Модификация геометрии: буферы в реальных метрах через UTM ---
            # EPSG:3857 искажает расстояния по широте (на Москве 1u ≈ 0.57m), что
            # давало одинаковые жирные дороги вместо class-aware ширины. Плюс
            # тротуары/лестницы (transportation:* fallback → class 4) занимали
            # ~10k features в moscow_city и забивали 25% пикселей дорогами.
            if layer_name == "transportation":
                gdf = _buffer_transportation_in_utm(gdf)
            elif layer_name in {"buildings", "osm_buildings"}:
                gdf = _buffer_buildings_in_utm(gdf, buffer_m=BUILDING_BUFFER_M)

            if gdf is None or gdf.empty:
                continue
            layer_gdfs[layer_name] = gdf

        tiles = enumerate_tiles(region["bbox"], zoom)
        for tile in tqdm(tiles, desc=f"masks:{region_id}"):
            image_name = f"{region_id}_{tile.key}.png"
            image_path = images_dir / image_name
            if not image_path.exists():
                continue

            lon_min, lat_min, lon_max, lat_max = tile_bbox_wgs84(tile)
            tile_poly = box(lon_min, lat_min, lon_max, lat_max)
            transform = from_bounds(lon_min, lat_min, lon_max, lat_max, tile_size, tile_size)

            class_mask = np.zeros((tile_size, tile_size), dtype=np.uint8)

            for class_id in config["raster_priority"]:
                if class_id == 0:
                    continue

                shapes = []
                for layer_name, gdf in layer_gdfs.items():
                    clipped = gdf[gdf.geometry.intersects(tile_poly)]
                    if clipped.empty:
                        continue

                    for _, row in clipped.iterrows():
                        props = row.to_dict()
                        cls = _class_from_feature(layer_name, props, class_mapping)
                        if cls != class_id:
                            continue
                        geom = row.geometry
                        if geom is None or geom.is_empty:
                            continue

                        # Гарантируем геометрию только внутри тайла, чтобы убрать размазывание
                        # по границам и артефакты от больших объектов.
                        try:
                            geom = geom.intersection(tile_poly)
                        except Exception:
                            continue

                        if geom is None or geom.is_empty:
                            continue

                        geom_type = geom.geom_type
                        if layer_name in {"water", "land_cover", "land_use", "buildings"}:
                            if geom_type not in {"Polygon", "MultiPolygon"}:
                                continue
                        elif layer_name == "transportation":
                            if geom_type not in {"LineString", "MultiLineString", "Polygon", "MultiPolygon"}:
                                continue

                        shapes.append((geom, class_id))

                if not shapes:
                    continue

                all_touched = False
                burned = rasterize(
                    shapes=shapes,
                    out_shape=(tile_size, tile_size),
                    fill=0,
                    transform=transform,
                    dtype="uint8",
                    all_touched=all_touched,
                )
                # raster_priority задан от более важного к менее важному,
                # поэтому не даем нижним классам перезаписывать уже назначенные пиксели.
                class_mask[(burned == class_id) & (class_mask == 0)] = class_id

            mask_name = f"{region_id}_{tile.key}.png"
            mask_path = mask_dir / mask_name
            cv2.imwrite(str(mask_path), class_mask)

            values, counts = np.unique(class_mask, return_counts=True)
            pix_stats = {int(v): int(c) for v, c in zip(values, counts)}
            total = int(class_mask.size)
            fg_pixels = total - pix_stats.get(0, 0)

            rows.append(
                {
                    "tile_id": f"{region_id}_{tile.key}",
                    "region_id": region_id,
                    "z": tile.z,
                    "x": tile.x,
                    "y": tile.y,
                    "image_path": str(image_path.relative_to(root)),
                    "mask_path": str(mask_path.relative_to(root)),
                    "non_background_ratio": round(fg_pixels / max(1, total), 6),
                    "pixel_hist_json": json.dumps(pix_stats, ensure_ascii=False),
                    "vector_source": "Overture Maps",
                    "zoom": tile.z,
                    "crs": "EPSG:4326",
                }
            )

    meta_path = root / "meta.csv"
    with meta_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    return meta_path


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    dil = cv2.dilate(mask, kernel, iterations=1)
    ero = cv2.erode(mask, kernel, iterations=1)
    return cv2.absdiff(dil, ero)


def run_quality_checks(root: Path, min_non_bg_ratio: float) -> Dict[str, Any]:
    meta_path = root / "meta.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Не найден {meta_path}")

    rows: List[Dict[str, str]] = []
    with meta_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    findings: Dict[str, Any] = {
        "total_pairs": len(rows),
        "missing_files": [],
        "size_mismatch": [],
        "likely_shifted": [],
        "empty_tiles": [],
        "smeared_tiles": [],
        "low_structural": [],
        "duplicate_images": [],
        "duplicate_masks": [],
        "rejected_tiles": [],
    }

    edge_overlap_min = 0.035
    max_dominant_ratio = 0.97
    min_structural_ratio = 0.003

    image_hash_map: Dict[str, List[str]] = defaultdict(list)
    mask_hash_map: Dict[str, List[str]] = defaultdict(list)

    for row in tqdm(rows, desc="qc"):
        tile_id = row["tile_id"]
        image_path = root / row["image_path"]
        mask_path = root / row["mask_path"]

        if not image_path.exists() or not mask_path.exists():
            findings["missing_files"].append(tile_id)
            continue

        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            findings["missing_files"].append(tile_id)
            continue

        if img.shape[:2] != mask.shape[:2]:
            findings["size_mismatch"].append(tile_id)
            continue

        non_bg_ratio = float(row.get("non_background_ratio", 0.0))
        if non_bg_ratio < min_non_bg_ratio:
            findings["empty_tiles"].append(tile_id)

        values, counts = np.unique(mask, return_counts=True)
        hist = {int(v): int(c) for v, c in zip(values, counts)}
        total = int(mask.size)
        dominant_ratio = max(hist.values()) / max(1, total)
        classes_non_bg = sum(1 for k in hist.keys() if k > 0 and hist[k] > 0)
        if dominant_ratio > max_dominant_ratio and classes_non_bg <= 1:
            findings["smeared_tiles"].append({"tile_id": tile_id, "dominant_ratio": round(dominant_ratio, 4)})

        structural_ratio = float(np.isin(mask, [3, 4]).sum() / max(1, total))
        if structural_ratio < min_structural_ratio:
            findings["low_structural"].append({"tile_id": tile_id, "structural_ratio": round(structural_ratio, 4)})

        edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 60, 120)
        boundary = _mask_boundary((mask > 0).astype(np.uint8) * 255)
        overlap = np.logical_and(edges > 0, boundary > 0).sum()
        boundary_count = max(1, int((boundary > 0).sum()))
        score = overlap / boundary_count
        if score < edge_overlap_min:
            findings["likely_shifted"].append({"tile_id": tile_id, "score": round(score, 4)})

        image_hash_map[image_hash(image_path)].append(tile_id)
        raw = mask.tobytes()
        mask_hash_map[sha1_bytes(raw)].append(tile_id)

    findings["duplicate_images"] = [v for v in image_hash_map.values() if len(v) > 1]
    findings["duplicate_masks"] = [v for v in mask_hash_map.values() if len(v) > 1]

    rejected = set(findings["missing_files"]) | set(findings["size_mismatch"]) | set(findings["empty_tiles"])
    rejected |= {item["tile_id"] for item in findings["likely_shifted"]}
    rejected |= {item["tile_id"] for item in findings["smeared_tiles"]}
    rejected |= {item["tile_id"] for item in findings["low_structural"]}
    findings["rejected_tiles"] = sorted(rejected)

    clean_meta_path = root / "meta_clean.csv"
    if rows:
        clean_rows = [row for row in rows if row["tile_id"] not in rejected]
        with clean_meta_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(clean_rows)

    qc_path = root / "qc_report.json"
    save_json(qc_path, findings)
    return findings


def generate_patches_and_split(
    root: Path,
    patch_size: int = 512,
    stride: int = 256,
    min_non_bg_ratio: float = 0.01,
) -> Path:
    del stride  # stride оставлен в интерфейсе, если позже добавим окно > 2x2

    images_tiles_dir = root / "images" / "tiles"
    masks_tiles_dir = root / "masks" / "tiles"
    patches_img_dir = root / "images" / "patches_512"
    patches_mask_dir = root / "masks" / "patches_512"
    ensure_dir(patches_img_dir)
    ensure_dir(patches_mask_dir)

    clean_meta_path = root / "meta_clean.csv"
    allowed_tile_ids: set[str] | None = None
    if clean_meta_path.exists():
        with clean_meta_path.open("r", encoding="utf-8") as f:
            allowed_tile_ids = {row["tile_id"] for row in csv.DictReader(f)}

    tile_index: Dict[Tuple[str, int, int], Dict[str, Path]] = {}
    for img_path in images_tiles_dir.glob("*.png"):
        parts = img_path.stem.split("_")
        if len(parts) < 4:
            continue
        region_id = "_".join(parts[:-3])
        z, x, y = map(int, parts[-3:])
        tile_id = f"{region_id}_{z}_{x}_{y}"
        if allowed_tile_ids is not None and tile_id not in allowed_tile_ids:
            continue
        mask_path = masks_tiles_dir / f"{img_path.stem}.png"
        if mask_path.exists():
            tile_index[(region_id, x, y)] = {"img": img_path, "mask": mask_path, "z": z}

    rows: List[Dict[str, Any]] = []
    for (region_id, x, y), item in tqdm(tile_index.items(), desc="patches"):
        neighbors = [
            tile_index.get((region_id, x, y)),
            tile_index.get((region_id, x + 1, y)),
            tile_index.get((region_id, x, y + 1)),
            tile_index.get((region_id, x + 1, y + 1)),
        ]
        if any(n is None for n in neighbors):
            continue

        img_tl = cv2.imread(str(neighbors[0]["img"]), cv2.IMREAD_COLOR)
        img_tr = cv2.imread(str(neighbors[1]["img"]), cv2.IMREAD_COLOR)
        img_bl = cv2.imread(str(neighbors[2]["img"]), cv2.IMREAD_COLOR)
        img_br = cv2.imread(str(neighbors[3]["img"]), cv2.IMREAD_COLOR)

        mask_tl = cv2.imread(str(neighbors[0]["mask"]), cv2.IMREAD_GRAYSCALE)
        mask_tr = cv2.imread(str(neighbors[1]["mask"]), cv2.IMREAD_GRAYSCALE)
        mask_bl = cv2.imread(str(neighbors[2]["mask"]), cv2.IMREAD_GRAYSCALE)
        mask_br = cv2.imread(str(neighbors[3]["mask"]), cv2.IMREAD_GRAYSCALE)

        if any(v is None for v in [img_tl, img_tr, img_bl, img_br, mask_tl, mask_tr, mask_bl, mask_br]):
            continue

        image_patch = np.vstack([np.hstack([img_tl, img_tr]), np.hstack([img_bl, img_br])])
        mask_patch = np.vstack([np.hstack([mask_tl, mask_tr]), np.hstack([mask_bl, mask_br])])

        if image_patch.shape[0] != patch_size or image_patch.shape[1] != patch_size:
            image_patch = cv2.resize(image_patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
            mask_patch = cv2.resize(mask_patch, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)

        non_bg_ratio = float((mask_patch > 0).sum() / mask_patch.size)
        if non_bg_ratio < min_non_bg_ratio:
            continue

        patch_id = f"{region_id}_{item['z']}_{x}_{y}"
        img_out = patches_img_dir / f"{patch_id}.png"
        mask_out = patches_mask_dir / f"{patch_id}.png"

        cv2.imwrite(str(img_out), image_patch)
        cv2.imwrite(str(mask_out), mask_patch)

        present_classes = sorted([int(c) for c in np.unique(mask_patch).tolist() if int(c) != 0])
        rows.append(
            {
                "patch_id": patch_id,
                "region_id": region_id,
                "image_path": str(img_out.relative_to(root)),
                "mask_path": str(mask_out.relative_to(root)),
                "non_background_ratio": round(non_bg_ratio, 6),
                "classes_present": ",".join(map(str, present_classes)),
            }
        )

    rare_target = {1, 3, 4}
    boosted = [r for r in rows if any(int(c) in rare_target for c in r["classes_present"].split(",") if c)]
    common = [r for r in rows if r not in boosted]

    # Оставляем больше патчей с водой/зданиями/дорогами, чтобы уменьшить дисбаланс.
    cap_common = max(len(boosted) * 2, 1)
    common = sorted(common, key=lambda r: r["non_background_ratio"], reverse=True)[:cap_common]
    selected = boosted + common

    region_to_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in selected:
        region_to_rows[row["region_id"]].append(row)

    # Сортируем регионы по количеству patches (богатые — в train), плюс
    # принудительно держим в train регионы с runway/taxiway (airport) — это
    # самый ценный класс для авиа-домена, и у нас всего один такой регион.
    def _region_priority(region_id: str) -> tuple:
        is_airport = "airport" in region_id.lower() or "svo" in region_id.lower()
        return (0 if is_airport else 1, -len(region_to_rows[region_id]), region_id)

    region_ids = sorted(region_to_rows.keys(), key=_region_priority)
    n = len(region_ids)
    n_train = max(1, int(n * 0.7))
    n_val = max(1, int(n * 0.15)) if n - n_train >= 2 else max(0, n - n_train - 1)
    train_regions = set(region_ids[:n_train])
    val_regions = set(region_ids[n_train : n_train + n_val])
    test_regions = set(region_ids) - train_regions - val_regions

    for row in selected:
        region_id = row["region_id"]
        if region_id in train_regions:
            row["split"] = "train"
        elif region_id in val_regions:
            row["split"] = "val"
        else:
            row["split"] = "test"

    out_path = root / "meta_patches.csv"
    if selected:
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(selected[0].keys()))
            writer.writeheader()
            writer.writerows(selected)

    split_path = root / "splits.json"
    save_json(
        split_path,
        {
            "train_regions": sorted(train_regions),
            "val_regions": sorted(val_regions),
            "test_regions": sorted(test_regions),
            "patch_size": patch_size,
            "selection_strategy": "rare-classes-boost",
            "used_clean_meta": bool(allowed_tile_ids is not None),
        },
    )

    aug_path = root / "segformer_augmentations.json"
    save_json(
        aug_path,
        {
            "train": [
                {"name": "RandomBrightnessContrast", "p": 0.5, "brightness_limit": 0.2, "contrast_limit": 0.2},
                {"name": "HueSaturationValue", "p": 0.4, "hue_shift_limit": 10, "sat_shift_limit": 20, "val_shift_limit": 10},
                {"name": "RandomFog", "p": 0.15, "fog_coef_lower": 0.05, "fog_coef_upper": 0.25},
                {"name": "MotionBlur", "p": 0.2, "blur_limit": 5},
                {"name": "RandomShadow", "p": 0.2},
                {"name": "HorizontalFlip", "p": 0.5},
                {"name": "VerticalFlip", "p": 0.1},
                {"name": "RandomRotate90", "p": 0.3},
            ],
            "val": [{"name": "NoOp"}],
            "notes": "Погодно-сезонные и цветовые аугментации для повышения устойчивости к съемке с самолёта.",
        },
    )

    return out_path


def ensure_structure(root: Path) -> None:
    for p in [
        root / "images" / "tiles",
        root / "images" / "patches_512",
        root / "masks" / "tiles",
        root / "masks" / "patches_512",
        root / "vectors",
    ]:
        ensure_dir(p)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Сборка Overture + satellite датасета для SegFormer")
    parser.add_argument("--config", required=True, help="Путь к JSON-конфигу")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["download", "rasterize", "qc", "patches", "all"],
        help="Этап пайплайна",
    )
    parser.add_argument("--min-non-bg-ratio", type=float, default=0.01, help="Минимальная доля не-фона")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_json(config_path)

    root = Path(config["dataset_root"]).resolve()
    ensure_structure(root)

    class_map_path = root / "class_map.json"
    save_json(
        class_map_path,
        {
            "classes": config["class_names"],
            "mapping_rules": config["class_mapping"],
            "raster_priority": config["raster_priority"],
        },
    )

    if args.stage in {"download", "all"}:
        download_satellite_tiles(config, root)

    if args.stage in {"rasterize", "all"}:
        rasterize_masks(config, root)

    if args.stage in {"qc", "all"}:
        run_quality_checks(root, min_non_bg_ratio=args.min_non_bg_ratio)

    if args.stage in {"patches", "all"}:
        generate_patches_and_split(root, min_non_bg_ratio=args.min_non_bg_ratio)


if __name__ == "__main__":
    main()
