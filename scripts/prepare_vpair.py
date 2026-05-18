"""Подготовка CSV-манифестов для VPAIR sample/full dataset.

Ожидаемая структура после ``scripts/download_vpair_sample.py`` или ручной
распаковки:

  poses_query.txt
  poses_reference_view.txt
  queries/*.png
  reference_views/*.png
  distractors/*.png                # опционально

Pose-файлы VPAIR содержат ECEF-координаты ``x,y,z`` и углы ``roll,pitch,yaw``.
Для совместимости с существующим ``ReferenceDatabase`` дополнительно считается
WGS84 ``lat,lon,alt``. Основная метрика VPAIR при этом остаётся метрической по
ECEF, потому что query/reference пары заданы в одной 3D-системе координат.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

# Константы WGS84.
_A = 6378137.0
_F = 1.0 / 298.257223563
_B = _A * (1.0 - _F)
_E2 = 1.0 - (_B * _B) / (_A * _A)
_EP2 = (_A * _A - _B * _B) / (_B * _B)


def _ecef_to_wgs84(x: float, y: float, z: float) -> tuple[float, float, float]:
    """ECEF в метрах -> WGS84 lat/lon в градусах и эллипсоидальная высота в метрах."""
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    theta = math.atan2(z * _A, p * _B)
    sin_t = math.sin(theta)
    cos_t = math.cos(theta)
    lat = math.atan2(
        z + _EP2 * _B * sin_t * sin_t * sin_t,
        p - _E2 * _A * cos_t * cos_t * cos_t,
    )
    sin_lat = math.sin(lat)
    n = _A / math.sqrt(1.0 - _E2 * sin_lat * sin_lat)
    alt = p / max(math.cos(lat), 1e-12) - n
    return math.degrees(lat), math.degrees(lon), alt


def _read_pose_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"VPAIR pose file not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _image_exists(root: Path, rel_path: str) -> bool:
    path = root / rel_path
    return path.exists() and path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def _row_from_pose(
    row: dict[str, str],
    root: Path,
    index: int,
    role: str,
    include_missing: bool,
) -> dict[str, Any] | None:
    rel_path = row["filepath"]
    if not include_missing and not _image_exists(root, rel_path):
        return None

    x = float(row["x"])
    y = float(row["y"])
    z = float(row["z"])
    lat, lon, alt = _ecef_to_wgs84(x, y, z)
    image_id = Path(rel_path).stem

    out: dict[str, Any] = {
        "index": index,
        "role": role,
        "image_id": image_id,
        "path": rel_path,
        "x_ecef_m": x,
        "y_ecef_m": y,
        "z_ecef_m": z,
        "lat": lat,
        "lon": lon,
        "alt_ellipsoid_m": alt,
        "undulation_m": float(row.get("undulation", "nan")),
        "roll_rad": float(row.get("roll", "nan")),
        "pitch_rad": float(row.get("pitch", "nan")),
        "yaw_rad": float(row.get("yaw", "nan")),
        "landcover": row.get("landcover", ""),
        "heading_convention": "vpair_ecef",
    }
    return out


def _scan_distractors(root: Path) -> list[dict[str, Any]]:
    distractor_root = root / "distractors"
    if not distractor_root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(
        p
        for p in distractor_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ):
        # У distractors координата зашита в имени как UTM-like east/north, но без
        # полного pose-файла. Для retrieval-базы они допустимы как hard negatives;
        # метрическую ошибку по ним не считаем.
        rows.append(
            {
                "index": len(rows),
                "role": "distractor",
                "image_id": path.stem,
                "path": _rel(path, root),
                "x_ecef_m": float("nan"),
                "y_ecef_m": float("nan"),
                "z_ecef_m": float("nan"),
                "lat": float("nan"),
                "lon": float("nan"),
                "alt_ellipsoid_m": float("nan"),
                "undulation_m": float("nan"),
                "roll_rad": float("nan"),
                "pitch_rad": float("nan"),
                "yaw_rad": float("nan"),
                "landcover": "",
                "heading_convention": "unknown",
            }
        )
    return rows


POSE_FIELDNAMES = [
    "index",
    "role",
    "image_id",
    "path",
    "x_ecef_m",
    "y_ecef_m",
    "z_ecef_m",
    "lat",
    "lon",
    "alt_ellipsoid_m",
    "undulation_m",
    "roll_rad",
    "pitch_rad",
    "yaw_rad",
    "landcover",
    "heading_convention",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="каталог VPAIR sample/full после распаковки",
    )
    parser.add_argument("--output", type=Path, default=Path("data/vpair/manifests"))
    parser.add_argument(
        "--include-missing-images",
        action="store_true",
        help="не выкидывать строки pose-файлов, если картинка отсутствует локально",
    )
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    if not root.exists():
        parser.error(f"каталог --dataset-root не найден: {root}")

    query_raw = _read_pose_csv(root / "poses_query.txt")
    ref_raw = _read_pose_csv(root / "poses_reference_view.txt")

    query_rows: list[dict[str, Any]] = []
    for row in query_raw:
        item = _row_from_pose(
            row,
            root,
            len(query_rows),
            "query",
            include_missing=args.include_missing_images,
        )
        if item is not None:
            query_rows.append(item)

    reference_rows: list[dict[str, Any]] = []
    for row in ref_raw:
        item = _row_from_pose(
            row,
            root,
            len(reference_rows),
            "reference",
            include_missing=args.include_missing_images,
        )
        if item is not None:
            reference_rows.append(item)

    distractor_rows = _scan_distractors(root)

    _write_csv(args.output / "queries.csv", query_rows, POSE_FIELDNAMES)
    _write_csv(args.output / "references.csv", reference_rows, POSE_FIELDNAMES)
    _write_csv(args.output / "distractors.csv", distractor_rows, POSE_FIELDNAMES)

    summary = {
        "dataset_root": str(root),
        "queries": len(query_rows),
        "references": len(reference_rows),
        "distractors": len(distractor_rows),
        "missing_queries_skipped": len(query_raw) - len(query_rows),
        "missing_references_skipped": len(ref_raw) - len(reference_rows),
    }
    (args.output / "summary.json").write_text(
        __import__("json").dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[vpair] queries    : {summary['queries']}")
    print(f"[vpair] references : {summary['references']}")
    print(f"[vpair] distractors: {summary['distractors']}")
    print(f"[vpair] manifests  : {args.output}")


if __name__ == "__main__":
    main()
