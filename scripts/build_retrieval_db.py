"""Этап 3.5, шаг 1: собрать reference-базу спутниковых тайлов для retrieval.

Скачивает сетку тайлов ``zoom`` (по умолчанию z=14, ~1.4 км/тайл на 55°N),
покрывающую bbox миссии, прогоняет каждый через encoder retriever и сохраняет
на диск набор (descriptors, lat/lon центров, tile_ids).

Выход: data/retrieval/db_z{zoom}.{npz, json}

Значения по умолчанию дают покрытие около 30 км × 30 км вокруг точки взлёта
GP010269, чего хватает на весь полёт с запасом.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import mercantile
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Обход проблемы сертификатов в macOS Python (см. эксперимент verify_undistort).
# torch.hub использует urllib, который учитывает SSL_CERT_FILE.
try:
    import certifi  # type: ignore
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except Exception:
    pass

from src.map_loader import MapDownloader
from src.retriever import ReferenceDatabase, Retriever


def _tile_center_latlon(tx: int, ty: int, z: int) -> tuple[float, float]:
    """Центр slippy-map тайла (z, x, y) в lat/lon."""
    bounds = mercantile.bounds(mercantile.Tile(tx, ty, z))
    return (
        0.5 * (bounds.south + bounds.north),
        0.5 * (bounds.west + bounds.east),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--centre-lat", type=float, default=55.086025,
                   help="centre of the area to cover (mission start by default)")
    p.add_argument("--centre-lon", type=float, default=38.149033)
    p.add_argument("--zoom", type=int, default=14,
                   help="reference-DB zoom level. z=14 ~ 1.4 km/tile at 55°N")
    p.add_argument("--radius-tiles", type=int, default=11,
                   help="±radius_tiles around centre. 11 -> 23x23 = 529 tiles "
                        "covering ~32 km × 32 km at z=14")
    p.add_argument("--output", type=Path, default=Path("data/retrieval"))
    p.add_argument("--db-name", type=str, default=None,
                   help="default = db_z<zoom>")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-workers", type=int, default=8,
                   help="parallel tile downloads")
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    db_path = args.output / (args.db_name or f"db_z{args.zoom}")

    centre_tile = mercantile.tile(args.centre_lon, args.centre_lat, args.zoom)
    print(f"[db] centre  : ({args.centre_lat}, {args.centre_lon})")
    print(f"[db] z={args.zoom}, tile=({centre_tile.x}, {centre_tile.y})")
    grid = (2 * args.radius_tiles + 1)
    print(f"[db] grid    : {grid}x{grid} = {grid*grid} tiles "
          f"(~{grid * 1.4:.0f} km × {grid * 1.4:.0f} km coverage at z=14)")
    print(f"[db] output  : {db_path}.{{npz,json}}")
    print()

    # 1. Генерируем список координат тайлов.
    tile_list = []
    for dy in range(-args.radius_tiles, args.radius_tiles + 1):
        for dx in range(-args.radius_tiles, args.radius_tiles + 1):
            tx = centre_tile.x + dx
            ty = centre_tile.y + dy
            tile_list.append((tx, ty, args.zoom))

    # 2. Параллельно скачиваем через MapDownloader.
    print(f"[db] downloading {len(tile_list)} tiles...")
    loader = MapDownloader(zoom=args.zoom, max_workers=args.max_workers)
    t0 = time.time()
    raw = loader.download_tiles_parallel(tile_list)  # {(tx,ty,z): PIL.Image or None}
    dl_dt = time.time() - t0
    failed = sum(1 for v in raw.values() if v is None)
    print(f"[db] downloaded in {dl_dt:.1f}s  failed={failed}/{len(tile_list)}")

    # 3. Строим дескрипторы.
    print(f"[db] loading {Retriever.__name__} ...")
    retr = Retriever(device=args.device)
    print(f"[db] device  : {retr.device}")

    valid_tiles = [(tx, ty, z) for (tx, ty, z) in tile_list if raw[(tx, ty, z)] is not None]
    print(f"[db] encoding {len(valid_tiles)} valid tiles in batches of {args.batch_size} ...")
    descriptors = np.empty((len(valid_tiles), 768), dtype=np.float32)
    centers_ll = np.empty((len(valid_tiles), 2), dtype=np.float64)
    tile_ids: list[str] = []
    t0 = time.time()
    for batch_start in range(0, len(valid_tiles), args.batch_size):
        batch = valid_tiles[batch_start:batch_start + args.batch_size]
        imgs = []
        for (tx, ty, z) in batch:
            pil = raw[(tx, ty, z)]
            arr = np.array(pil)  # RGB
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            elif arr.shape[2] == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            else:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            imgs.append(arr)
        feats = retr.encode_batch(imgs)
        for i, (tx, ty, z) in enumerate(batch):
            j = batch_start + i
            descriptors[j] = feats[i]
            lat, lon = _tile_center_latlon(tx, ty, z)
            centers_ll[j, 0] = lat
            centers_ll[j, 1] = lon
            tile_ids.append(f"{z}/{tx}/{ty}")
        if batch_start // args.batch_size % 10 == 0:
            done = batch_start + len(batch)
            print(f"  encoded {done}/{len(valid_tiles)}  ({done/(time.time()-t0):.1f} tiles/s)")
    print(f"[db] encoded in {time.time()-t0:.1f}s")

    # 4. Save.
    db = ReferenceDatabase(
        descriptors=descriptors,
        centers_latlon=centers_ll,
        tile_ids=tile_ids,
        zoom=args.zoom,
        model_name=retr.model_name,
        input_size=retr.input_size,
    )
    db.save(db_path)
    sz_mb = (db_path.with_suffix(".npz").stat().st_size) / 1024 / 1024
    print(f"[db] saved {db_path}.npz ({sz_mb:.2f} MB) + .json")
    bbox_lats = centers_ll[:, 0]
    bbox_lons = centers_ll[:, 1]
    print(f"[db] bbox    : lat [{bbox_lats.min():.4f}, {bbox_lats.max():.4f}]  "
          f"lon [{bbox_lons.min():.4f}, {bbox_lons.max():.4f}]")


if __name__ == "__main__":
    main()
