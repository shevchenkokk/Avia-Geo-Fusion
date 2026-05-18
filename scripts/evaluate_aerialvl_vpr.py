"""Оценка DINOv2-поиска на VPR-разделе датасета AerialVL.

Это первый мост между AerialVL и текущим кодом проекта. Скрипт использует уже
существующий кодировщик ``src.retriever.Retriever`` и считает геолокационные
метрики по координатам, которые AerialVL хранит прямо в именах файлов.

Типовой порядок запуска:
  1. Сначала выполнить ``scripts/prepare_aerialvl.py`` и получить манифесты.
  2. Затем запустить этот скрипт для одного из уровней ``map_database/level_*``.

На выходе сохраняются:
  - ``summary.json`` и ``summary.md`` с метриками Recall@K@R;
  - ``predictions.csv`` с top-1 ответом для каждого query-изображения;
  - опциональный кэш базы поиска в формате ``ReferenceDatabase`` проекта.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# На macOS torch.hub иногда падает из-за системных сертификатов Python.
# Явно подставляем certifi, чтобы загрузка DINOv2 была воспроизводимой.
try:
    import certifi  # type: ignore

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except Exception:
    pass

from src.retriever import ReferenceDatabase, Retriever


@dataclass(frozen=True)
class MapTile:
    index: int
    level: str
    path: Path
    rel_path: str
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float
    center_lat: float
    center_lon: float


@dataclass(frozen=True)
class QueryImage:
    index: int
    query_set: str
    path: Path
    rel_path: str
    lat: float
    lon: float


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_image_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")
    return img


def _dist_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Быстрая локальная оценка расстояния между двумя WGS84-точками в метрах."""
    mean_lat = math.radians(0.5 * (lat1 + lat2))
    dy = (lat2 - lat1) * 111_320.0
    dx = (lon2 - lon1) * 111_320.0 * math.cos(mean_lat)
    return math.hypot(dx, dy)


def _inside(tile: MapTile, lat: float, lon: float) -> bool:
    return tile.lat_min <= lat <= tile.lat_max and tile.lon_min <= lon <= tile.lon_max


def _load_tiles(dataset_root: Path, manifest_dir: Path, level: str) -> list[MapTile]:
    rows = _read_csv(manifest_dir / "vpr_map_tiles.csv")
    out: list[MapTile] = []
    for row in rows:
        if row["level"] != level:
            continue
        rel_path = row["path"]
        out.append(
            MapTile(
                index=len(out),
                level=row["level"],
                path=dataset_root / rel_path,
                rel_path=rel_path,
                lon_min=float(row["lon_min"]),
                lat_min=float(row["lat_min"]),
                lon_max=float(row["lon_max"]),
                lat_max=float(row["lat_max"]),
                center_lat=float(row["center_lat"]),
                center_lon=float(row["center_lon"]),
            )
        )
    return out


def _load_queries(
    dataset_root: Path, manifest_dir: Path, query_set: str | None
) -> list[QueryImage]:
    rows = _read_csv(manifest_dir / "vpr_queries.csv")
    out: list[QueryImage] = []
    for row in rows:
        if query_set is not None and row["query_set"] != query_set:
            continue
        rel_path = row["path"]
        out.append(
            QueryImage(
                index=len(out),
                query_set=row["query_set"],
                path=dataset_root / rel_path,
                rel_path=rel_path,
                lat=float(row["lat"]),
                lon=float(row["lon"]),
            )
        )
    return out


def _build_or_load_db(
    retriever: Retriever,
    tiles: list[MapTile],
    db_path: Path,
    batch_size: int,
    force_rebuild: bool,
) -> ReferenceDatabase:
    if (
        not force_rebuild
        and db_path.with_suffix(".npz").exists()
        and db_path.with_suffix(".json").exists()
    ):
        db = ReferenceDatabase.load(db_path)
        if len(db) != len(tiles):
            raise ValueError(
                f"В кэше {len(db)} записей, а в manifest-файле {len(tiles)} тайлов; "
                "пересоберите базу с --force-rebuild-db"
            )
        retriever.load_database(db_path)
        return db

    db_path.parent.mkdir(parents=True, exist_ok=True)
    descriptors = np.empty((len(tiles), 768), dtype=np.float32)
    centers_latlon = np.empty((len(tiles), 2), dtype=np.float64)
    tile_ids: list[str] = []

    t0 = time.time()
    for batch_start in range(0, len(tiles), batch_size):
        batch = tiles[batch_start : batch_start + batch_size]
        imgs = [_read_image_bgr(tile.path) for tile in batch]
        feats = retriever.encode_batch(imgs)
        for i, tile in enumerate(batch):
            j = batch_start + i
            descriptors[j] = feats[i]
            centers_latlon[j, 0] = tile.center_lat
            centers_latlon[j, 1] = tile.center_lon
            tile_ids.append(tile.rel_path)
        done = batch_start + len(batch)
        print(
            f"[aerialvl] закодировано gallery-тайлов {done}/{len(tiles)} "
            f"({done / max(time.time() - t0, 1e-6):.2f} img/s)"
        )

    db = ReferenceDatabase(
        descriptors=descriptors,
        centers_latlon=centers_latlon,
        tile_ids=tile_ids,
        zoom=-1,
        model_name=retriever.model_name,
        input_size=retriever.input_size,
    )
    db.save(db_path)
    retriever.load_database(db_path)
    return db


def _percent(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# AerialVL: VPR-валидация",
        "",
        f"Статус: {'PASS' if payload['passed'] else 'FAIL'}",
        "",
        "## Датасет",
        "",
        f"- level: `{payload['level']}`",
        f"- query_set: `{payload['query_set']}`",
        f"- gallery_tiles: `{payload['gallery_tiles']}`",
        f"- queries: `{payload['queries']}`",
        "",
        "## Ошибка top-1",
        "",
        f"- median_top1_error_m: `{payload['median_top1_error_m']:.2f}`",
        f"- p95_top1_error_m: `{payload['p95_top1_error_m']:.2f}`",
        f"- mean_top1_error_m: `{payload['mean_top1_error_m']:.2f}`",
        f"- top1_inside_tile_pct: `{payload['top1_inside_tile_pct']:.2f}`",
        "",
        "## Recall@K внутри заданного радиуса",
        "",
        "| Метрика | Значение, % |",
        "|---|---:|",
    ]
    for key, value in sorted(payload["recall"].items()):
        lines.append(f"| {key} | {value:.2f} |")
    lines.extend(
        [
            "",
            "## Файлы",
            "",
            f"- predictions: `{payload['predictions_csv']}`",
            f"- retrieval_db: `{payload['retrieval_db']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="путь к распакованному AerialVL",
    )
    parser.add_argument(
        "--manifest-dir", type=Path, default=Path("data/aerialvl/manifests")
    )
    parser.add_argument(
        "--level",
        default="level_1",
        help="уровень map_database, который берём как gallery",
    )
    parser.add_argument(
        "--query-set",
        default=None,
        help="опционально: конкретная папка query_images_*",
    )
    parser.add_argument("--output", type=Path, default=Path("results/aerialvl_vpr"))
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="путь к кэшу ReferenceDatabase без .npz/.json суффикса",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument(
        "--radii-m", type=float, nargs="+", default=[25.0, 50.0, 100.0, 200.0]
    )
    parser.add_argument(
        "--max-queries", type=int, default=0, help="0 означает использовать все query"
    )
    parser.add_argument("--force-rebuild-db", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    manifest_dir = args.manifest_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    tiles = _load_tiles(dataset_root, manifest_dir, args.level)
    queries = _load_queries(dataset_root, manifest_dir, args.query_set)
    if args.max_queries > 0:
        queries = queries[: args.max_queries]
    if not tiles:
        parser.error(
            f"для level={args.level!r} в {manifest_dir} не найдено VPR map tiles"
        )
    if not queries:
        parser.error(f"в {manifest_dir} не найдено VPR query-изображений")

    db_path = args.db_path or (output / f"aerialvl_{args.level}_dinov2_db")
    print(f"[aerialvl] gallery tiles : {len(tiles)}")
    print(f"[aerialvl] queries       : {len(queries)}")
    print(f"[aerialvl] db            : {db_path}.{{npz,json}}")

    retriever = Retriever(device=args.device)
    db = _build_or_load_db(
        retriever=retriever,
        tiles=tiles,
        db_path=db_path,
        batch_size=max(1, args.batch_size),
        force_rebuild=args.force_rebuild_db,
    )
    if retriever.database is None:
        retriever.load_database(db_path)

    max_k = max(args.top_k)
    predictions: list[dict[str, Any]] = []
    top1_errors: list[float] = []
    top1_inside = 0
    recall_counts = {
        f"R@{k}_{int(radius)}m": 0 for k in args.top_k for radius in args.radii_m
    }

    t0 = time.time()
    for qi, query in enumerate(queries, start=1):
        img = _read_image_bgr(query.path)
        descriptor = retriever.encode(img)
        ranked = db.query(descriptor, top_k=max_k)
        ranked_tiles = [(tiles[idx], score) for idx, score in ranked]
        ranked_errors = [
            _dist_m(query.lat, query.lon, tile.center_lat, tile.center_lon)
            for tile, _ in ranked_tiles
        ]

        top_tile, top_score = ranked_tiles[0]
        top1_error = ranked_errors[0]
        inside = _inside(top_tile, query.lat, query.lon)
        top1_errors.append(top1_error)
        top1_inside += int(inside)

        for k in args.top_k:
            k_errors = ranked_errors[:k]
            for radius in args.radii_m:
                key = f"R@{k}_{int(radius)}m"
                if min(k_errors) <= radius:
                    recall_counts[key] += 1

        predictions.append(
            {
                "query_index": query.index,
                "query_set": query.query_set,
                "query_path": query.rel_path,
                "query_lat": query.lat,
                "query_lon": query.lon,
                "top1_tile_index": top_tile.index,
                "top1_tile_path": top_tile.rel_path,
                "top1_center_lat": top_tile.center_lat,
                "top1_center_lon": top_tile.center_lon,
                "top1_score": top_score,
                "top1_error_m": top1_error,
                "top1_inside_tile": int(inside),
            }
        )

        if qi % 50 == 0 or qi == len(queries):
            print(
                f"[aerialvl] обработано query {qi}/{len(queries)} "
                f"({qi / max(time.time() - t0, 1e-6):.2f} q/s)"
            )

    err = np.asarray(top1_errors, dtype=np.float64)
    recall_pct = {
        key: _percent(value, len(queries)) for key, value in recall_counts.items()
    }
    predictions_csv = output / "predictions.csv"
    summary_json = output / "summary.json"
    summary_md = output / "summary.md"
    _write_predictions(predictions_csv, predictions)

    payload: dict[str, Any] = {
        "passed": True,
        "dataset_root": str(dataset_root),
        "manifest_dir": str(manifest_dir),
        "level": args.level,
        "query_set": args.query_set or "all",
        "gallery_tiles": len(tiles),
        "queries": len(queries),
        "retrieval_db": str(db_path),
        "predictions_csv": str(predictions_csv),
        "median_top1_error_m": float(np.median(err)),
        "p95_top1_error_m": float(np.percentile(err, 95)),
        "mean_top1_error_m": float(np.mean(err)),
        "top1_inside_tile_pct": _percent(top1_inside, len(queries)),
        "recall": recall_pct,
    }
    summary_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_summary_md(summary_md, payload)
    print(f"[aerialvl] summary -> {summary_md}")
    print(f"[aerialvl] json    -> {summary_json}")


if __name__ == "__main__":
    main()
