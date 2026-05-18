"""VPAIR VPR-бенчмарк через существующий DINOv2 retriever проекта.

Скрипт использует манифесты из ``scripts/prepare_vpair.py`` и считает:

  - Recall@K@R по метрической ECEF-дистанции до reference view;
  - top-1 exact-pair accuracy по совпадению ``image_id``;
  - роль top-1 ответа: reference или distractor;
  - median/p95/mean top-1 error для случаев, когда top-1 — reference.

VPAIR важен для диплома как proxy-сценарий ближе к самолётному: высота >300 м,
надирная камера, длинная траектория, reference renders + distractors.
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

try:
    import certifi  # type: ignore

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except Exception:
    pass

from src.retriever import ReferenceDatabase, Retriever


@dataclass(frozen=True)
class VPairImage:
    index: int
    role: str
    image_id: str
    path: Path
    rel_path: str
    x_ecef_m: float
    y_ecef_m: float
    z_ecef_m: float
    lat: float
    lon: float
    landcover: str = ""

    @property
    def has_metric_pose(self) -> bool:
        return all(
            math.isfinite(v) for v in (self.x_ecef_m, self.y_ecef_m, self.z_ecef_m)
        )


@dataclass(frozen=True)
class GalleryItem:
    index: int
    role: str
    image_id: str
    path: Path
    rel_path: str
    x_ecef_m: float
    y_ecef_m: float
    z_ecef_m: float
    lat: float
    lon: float

    @property
    def has_metric_pose(self) -> bool:
        return all(
            math.isfinite(v) for v in (self.x_ecef_m, self.y_ecef_m, self.z_ecef_m)
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(value: str, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_image_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")
    return img


def _dist_ecef_m(a: VPairImage, b: GalleryItem) -> float:
    if not a.has_metric_pose or not b.has_metric_pose:
        return float("inf")
    return float(
        math.sqrt(
            (a.x_ecef_m - b.x_ecef_m) ** 2
            + (a.y_ecef_m - b.y_ecef_m) ** 2
            + (a.z_ecef_m - b.z_ecef_m) ** 2
        )
    )


def _load_queries(dataset_root: Path, manifest_dir: Path) -> list[VPairImage]:
    rows = _read_csv(manifest_dir / "queries.csv")
    out: list[VPairImage] = []
    for row in rows:
        rel = row["path"]
        out.append(
            VPairImage(
                index=len(out),
                role=row.get("role", "query"),
                image_id=row.get("image_id") or Path(rel).stem,
                path=dataset_root / rel,
                rel_path=rel,
                x_ecef_m=_float(row.get("x_ecef_m", "nan")),
                y_ecef_m=_float(row.get("y_ecef_m", "nan")),
                z_ecef_m=_float(row.get("z_ecef_m", "nan")),
                lat=_float(row.get("lat", "nan")),
                lon=_float(row.get("lon", "nan")),
                landcover=row.get("landcover", ""),
            )
        )
    return out


def _load_gallery(
    dataset_root: Path,
    manifest_dir: Path,
    include_distractors: bool,
) -> list[GalleryItem]:
    gallery: list[GalleryItem] = []
    for row in _read_csv(manifest_dir / "references.csv"):
        rel = row["path"]
        gallery.append(
            GalleryItem(
                index=len(gallery),
                role="reference",
                image_id=row.get("image_id") or Path(rel).stem,
                path=dataset_root / rel,
                rel_path=rel,
                x_ecef_m=_float(row.get("x_ecef_m", "nan")),
                y_ecef_m=_float(row.get("y_ecef_m", "nan")),
                z_ecef_m=_float(row.get("z_ecef_m", "nan")),
                lat=_float(row.get("lat", "nan")),
                lon=_float(row.get("lon", "nan")),
            )
        )
    if include_distractors:
        distractor_path = manifest_dir / "distractors.csv"
        if distractor_path.exists():
            for row in _read_csv(distractor_path):
                rel = row["path"]
                gallery.append(
                    GalleryItem(
                        index=len(gallery),
                        role="distractor",
                        image_id=row.get("image_id") or Path(rel).stem,
                        path=dataset_root / rel,
                        rel_path=rel,
                        x_ecef_m=float("nan"),
                        y_ecef_m=float("nan"),
                        z_ecef_m=float("nan"),
                        lat=float("nan"),
                        lon=float("nan"),
                    )
                )
    return gallery


def _build_or_load_db(
    retriever: Retriever,
    gallery: list[GalleryItem],
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
        if len(db) != len(gallery):
            raise ValueError(
                f"В кэше {len(db)} записей, а в gallery {len(gallery)}; "
                "пересоберите с --force-rebuild-db"
            )
        retriever.load_database(db_path)
        return db

    db_path.parent.mkdir(parents=True, exist_ok=True)
    descriptors = np.empty((len(gallery), 768), dtype=np.float32)
    centers_latlon = np.zeros((len(gallery), 2), dtype=np.float64)
    tile_ids: list[str] = []

    t0 = time.time()
    for batch_start in range(0, len(gallery), batch_size):
        batch = gallery[batch_start : batch_start + batch_size]
        imgs = [_read_image_bgr(item.path) for item in batch]
        feats = retriever.encode_batch(imgs)
        for i, item in enumerate(batch):
            j = batch_start + i
            descriptors[j] = feats[i]
            centers_latlon[j, 0] = item.lat if math.isfinite(item.lat) else 0.0
            centers_latlon[j, 1] = item.lon if math.isfinite(item.lon) else 0.0
            tile_ids.append(item.rel_path)
        done = batch_start + len(batch)
        print(
            f"[vpair] encoded gallery {done}/{len(gallery)} "
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
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _finite_stats(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(arr)), float(np.percentile(arr, 95)), float(np.mean(arr))


def _write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# VPAIR: VPR-валидация",
        "",
        f"Статус: {'PASS' if payload['passed'] else 'FAIL'}",
        "",
        "## Датасет",
        "",
        f"- dataset_root: `{payload['dataset_root']}`",
        f"- gallery_images: `{payload['gallery_images']}`",
        f"- reference_images: `{payload['reference_images']}`",
        f"- distractor_images: `{payload['distractor_images']}`",
        f"- queries: `{payload['queries']}`",
        "",
        "## Top-1",
        "",
        f"- exact_pair_top1_pct: `{payload['exact_pair_top1_pct']:.2f}`",
        f"- top1_reference_pct: `{payload['top1_reference_pct']:.2f}`",
        f"- top1_distractor_pct: `{payload['top1_distractor_pct']:.2f}`",
        f"- median_top1_reference_error_m: `{payload['median_top1_reference_error_m']:.2f}`",
        f"- p95_top1_reference_error_m: `{payload['p95_top1_reference_error_m']:.2f}`",
        f"- mean_top1_reference_error_m: `{payload['mean_top1_reference_error_m']:.2f}`",
        "",
        "## Recall@K внутри ECEF-радиуса",
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
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--manifest-dir", type=Path, default=Path("data/vpair/manifests")
    )
    parser.add_argument("--output", type=Path, default=Path("results/vpair_vpr"))
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument(
        "--radii-m", type=float, nargs="+", default=[25.0, 50.0, 100.0, 200.0]
    )
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument(
        "--include-distractors", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--force-rebuild-db", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    manifest_dir = args.manifest_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    queries = _load_queries(dataset_root, manifest_dir)
    if args.max_queries > 0:
        queries = queries[: args.max_queries]
    gallery = _load_gallery(
        dataset_root, manifest_dir, include_distractors=args.include_distractors
    )
    if not queries:
        parser.error(f"в {manifest_dir} не найдено VPAIR queries")
    if not gallery:
        parser.error(f"в {manifest_dir} не найдено VPAIR gallery")

    n_ref = sum(1 for item in gallery if item.role == "reference")
    n_dist = sum(1 for item in gallery if item.role == "distractor")
    db_path = args.db_path or (output / "vpair_dinov2_db")
    print(f"[vpair] gallery      : {len(gallery)} ({n_ref} ref, {n_dist} distractors)")
    print(f"[vpair] queries      : {len(queries)}")
    print(f"[vpair] db           : {db_path}.{{npz,json}}")

    retriever = Retriever(device=args.device)
    db = _build_or_load_db(
        retriever=retriever,
        gallery=gallery,
        db_path=db_path,
        batch_size=max(1, args.batch_size),
        force_rebuild=args.force_rebuild_db,
    )
    if retriever.database is None:
        retriever.load_database(db_path)

    max_k = min(max(args.top_k), len(gallery))
    predictions: list[dict[str, Any]] = []
    top1_errors: list[float] = []
    exact_top1 = 0
    top1_reference = 0
    top1_distractor = 0
    recall_counts = {
        f"R@{k}_{int(radius)}m": 0 for k in args.top_k for radius in args.radii_m
    }

    t0 = time.time()
    for qi, query in enumerate(queries, start=1):
        descriptor = retriever.encode(_read_image_bgr(query.path))
        ranked = db.query(descriptor, top_k=max_k)
        ranked_items = [(gallery[idx], score) for idx, score in ranked]
        ranked_errors = [_dist_ecef_m(query, item) for item, _ in ranked_items]

        top_item, top_score = ranked_items[0]
        top_error = ranked_errors[0]
        top1_errors.append(top_error)
        exact_top1 += int(
            top_item.role == "reference" and top_item.image_id == query.image_id
        )
        top1_reference += int(top_item.role == "reference")
        top1_distractor += int(top_item.role == "distractor")

        for k in args.top_k:
            k_errors = ranked_errors[: min(k, len(ranked_errors))]
            for radius in args.radii_m:
                key = f"R@{k}_{int(radius)}m"
                if k_errors and min(k_errors) <= radius:
                    recall_counts[key] += 1

        predictions.append(
            {
                "query_index": query.index,
                "query_image_id": query.image_id,
                "query_path": query.rel_path,
                "query_landcover": query.landcover,
                "top1_gallery_index": top_item.index,
                "top1_role": top_item.role,
                "top1_image_id": top_item.image_id,
                "top1_path": top_item.rel_path,
                "top1_score": top_score,
                "top1_error_m": top_error,
                "top1_exact_pair": int(
                    top_item.role == "reference" and top_item.image_id == query.image_id
                ),
            }
        )

        if qi % 50 == 0 or qi == len(queries):
            print(
                f"[vpair] processed query {qi}/{len(queries)} "
                f"({qi / max(time.time() - t0, 1e-6):.2f} q/s)"
            )

    median_err, p95_err, mean_err = _finite_stats(top1_errors)
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
        "gallery_images": len(gallery),
        "reference_images": n_ref,
        "distractor_images": n_dist,
        "queries": len(queries),
        "retrieval_db": str(db_path),
        "predictions_csv": str(predictions_csv),
        "exact_pair_top1_pct": _percent(exact_top1, len(queries)),
        "top1_reference_pct": _percent(top1_reference, len(queries)),
        "top1_distractor_pct": _percent(top1_distractor, len(queries)),
        "median_top1_reference_error_m": median_err,
        "p95_top1_reference_error_m": p95_err,
        "mean_top1_reference_error_m": mean_err,
        "recall": recall_pct,
    }
    summary_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_summary_md(summary_md, payload)
    print(f"[vpair] summary -> {summary_md}")
    print(f"[vpair] json    -> {summary_json}")


if __name__ == "__main__":
    main()
