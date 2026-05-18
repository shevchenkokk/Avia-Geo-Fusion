"""Подготовка лёгких манифестов для датасета AerialVL.

Скрипт не копирует тяжёлые изображения. Он проходит по распакованному каталогу
AerialVL, вытаскивает координаты WGS84 из имён файлов и сохраняет несколько CSV,
которые дальше удобно использовать в экспериментах.

Поддерживаются две структуры, описанные авторами AerialVL:

VAL:
  geo_referenced_map/@large_map@LT_lon@LT_lat@RB_lon@RB_lat@.tif
  long_trajtr/<sequence>/@UTC@lon@lat@.png
  short_trajtr/<sequence>/@UTC@lon@lat@.png

VPR:
  map_database/level_*/@map@LB_lon@LB_lat@RT_lon@RT_lat@.png
  query_images/query_images_*/@lon@lat@.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _fields_from_at_name(path: Path) -> list[str]:
    """Разбивает имя вида @поле@поле@ на непустые части."""
    return [part for part in path.stem.split("@") if part]


def _float(value: str, *, path: Path, field: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"Не удалось прочитать {field}={value!r} в имени файла {path}"
        ) from exc


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _iter_images(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def scan_vpr(root: Path) -> tuple[list[dict], list[dict]]:
    """Собирает строки для тайлов карты и query-кадров из VPR-раздела."""
    map_rows: list[dict] = []
    map_root = root / "map_database"
    if map_root.exists():
        for level_dir in sorted(p for p in map_root.iterdir() if p.is_dir()):
            level = level_dir.name
            for path in _iter_images(level_dir):
                fields = _fields_from_at_name(path)
                if len(fields) != 5:
                    continue
                name, lb_lon_s, lb_lat_s, rt_lon_s, rt_lat_s = fields
                lb_lon = _float(lb_lon_s, path=path, field="left_bottom_lon")
                lb_lat = _float(lb_lat_s, path=path, field="left_bottom_lat")
                rt_lon = _float(rt_lon_s, path=path, field="right_top_lon")
                rt_lat = _float(rt_lat_s, path=path, field="right_top_lat")
                lon_min, lon_max = min(lb_lon, rt_lon), max(lb_lon, rt_lon)
                lat_min, lat_max = min(lb_lat, rt_lat), max(lb_lat, rt_lat)
                map_rows.append(
                    {
                        "level": level,
                        "name": name,
                        "path": _rel(path, root),
                        "lon_min": lon_min,
                        "lat_min": lat_min,
                        "lon_max": lon_max,
                        "lat_max": lat_max,
                        "center_lat": 0.5 * (lat_min + lat_max),
                        "center_lon": 0.5 * (lon_min + lon_max),
                        "heading_convention": "east",
                    }
                )

    query_rows: list[dict] = []
    query_root = root / "query_images"
    if query_root.exists():
        for query_dir in sorted(p for p in query_root.iterdir() if p.is_dir()):
            query_set = query_dir.name
            for path in _iter_images(query_dir):
                fields = _fields_from_at_name(path)
                if len(fields) != 2:
                    continue
                lon_s, lat_s = fields
                lon = _float(lon_s, path=path, field="lon")
                lat = _float(lat_s, path=path, field="lat")
                query_rows.append(
                    {
                        "query_set": query_set,
                        "path": _rel(path, root),
                        "lat": lat,
                        "lon": lon,
                        "heading_convention": "east",
                    }
                )
    return map_rows, query_rows


def scan_val(root: Path) -> tuple[list[dict], list[dict]]:
    """Собирает строки для geo-referenced maps и кадров траекторий из VAL-раздела."""
    map_rows: list[dict] = []
    geo_map_root = root / "geo_referenced_map"
    if geo_map_root.exists():
        for path in _iter_images(geo_map_root):
            fields = _fields_from_at_name(path)
            if len(fields) != 5:
                continue
            name, lt_lon_s, lt_lat_s, rb_lon_s, rb_lat_s = fields
            lt_lon = _float(lt_lon_s, path=path, field="left_top_lon")
            lt_lat = _float(lt_lat_s, path=path, field="left_top_lat")
            rb_lon = _float(rb_lon_s, path=path, field="right_bottom_lon")
            rb_lat = _float(rb_lat_s, path=path, field="right_bottom_lat")
            lon_min, lon_max = min(lt_lon, rb_lon), max(lt_lon, rb_lon)
            lat_min, lat_max = min(lt_lat, rb_lat), max(lt_lat, rb_lat)
            map_rows.append(
                {
                    "name": name,
                    "path": _rel(path, root),
                    "lon_min": lon_min,
                    "lat_min": lat_min,
                    "lon_max": lon_max,
                    "lat_max": lat_max,
                    "center_lat": 0.5 * (lat_min + lat_max),
                    "center_lon": 0.5 * (lon_min + lon_max),
                    "heading_convention": "north",
                }
            )

    frame_rows: list[dict] = []
    for split in ("long_trajtr", "short_trajtr"):
        split_root = root / split
        if not split_root.exists():
            continue
        for seq_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
            for path in _iter_images(seq_dir):
                fields = _fields_from_at_name(path)
                if len(fields) != 3:
                    continue
                utc_s, lon_s, lat_s = fields
                lon = _float(lon_s, path=path, field="lon")
                lat = _float(lat_s, path=path, field="lat")
                frame_rows.append(
                    {
                        "split": split,
                        "sequence": seq_dir.name,
                        "path": _rel(path, root),
                        "utc_timestamp": utc_s,
                        "lat": lat,
                        "lon": lon,
                        "heading_convention": "east",
                    }
                )
    return map_rows, frame_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="путь к распакованному каталогу AerialVL",
    )
    parser.add_argument("--output", type=Path, default=Path("data/aerialvl/manifests"))
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    if not root.exists():
        parser.error(f"каталог --dataset-root не найден: {root}")
    args.output.mkdir(parents=True, exist_ok=True)

    vpr_maps, vpr_queries = scan_vpr(root)
    val_maps, val_frames = scan_val(root)

    _write_csv(
        args.output / "vpr_map_tiles.csv",
        vpr_maps,
        [
            "level",
            "name",
            "path",
            "lon_min",
            "lat_min",
            "lon_max",
            "lat_max",
            "center_lat",
            "center_lon",
            "heading_convention",
        ],
    )
    _write_csv(
        args.output / "vpr_queries.csv",
        vpr_queries,
        [
            "query_set",
            "path",
            "lat",
            "lon",
            "heading_convention",
        ],
    )
    _write_csv(
        args.output / "val_maps.csv",
        val_maps,
        [
            "name",
            "path",
            "lon_min",
            "lat_min",
            "lon_max",
            "lat_max",
            "center_lat",
            "center_lon",
            "heading_convention",
        ],
    )
    _write_csv(
        args.output / "val_frames.csv",
        val_frames,
        [
            "split",
            "sequence",
            "path",
            "utc_timestamp",
            "lat",
            "lon",
            "heading_convention",
        ],
    )

    summary = {
        "dataset_root": str(root),
        "vpr_map_tiles": len(vpr_maps),
        "vpr_queries": len(vpr_queries),
        "val_maps": len(val_maps),
        "val_frames": len(val_frames),
        "outputs": {
            "vpr_map_tiles": str(args.output / "vpr_map_tiles.csv"),
            "vpr_queries": str(args.output / "vpr_queries.csv"),
            "val_maps": str(args.output / "val_maps.csv"),
            "val_frames": str(args.output / "val_frames.csv"),
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
