"""Очистка Overture-RU датасета перед перетреном SegFormer.

Дропаем три типа мусорных тайлов:
  1. monoculture: >= mono-threshold пикселей в одном классе (по умолчанию 95%);
     такие тайлы доминируют в karelia_forest / krasnodar_fields / rostov_open
     (медиана 100% vegetation) и смещают prior.
  2. smeared:    помечены qc_report как `smeared_tiles` (rasterization провалилась).
  3. low-fg:     non_background_ratio < min-foreground-share (по умолчанию 0.05).

На выходе пишем meta_filtered.csv с тем же набором колонок, что и meta.csv,
плюс колонку `kept_reason`. SegFormer-конфиг должен указывать на этот файл.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--input-csv", default="meta.csv")
    p.add_argument("--output-csv", default="meta_filtered.csv")
    p.add_argument("--mono-threshold", type=float, default=0.95,
                   help="Drop tiles where any single class share >= this")
    p.add_argument("--min-foreground-share", type=float, default=0.05,
                   help="Drop tiles with non_background_ratio below this")
    p.add_argument("--also-drop-likely-shifted", action="store_true",
                   help="Additionally drop tiles in qc_report.likely_shifted (legacy heuristic)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root)
    in_path = root / args.input_csv
    out_path = root / args.output_csv

    qc = json.loads((root / "qc_report.json").read_text())
    smeared_ids = {x["tile_id"] for x in qc.get("smeared_tiles", [])}
    shifted_ids = {x["tile_id"] for x in qc.get("likely_shifted", [])}

    drops = defaultdict(int)
    kept_rows = []
    region_counts = defaultdict(lambda: defaultdict(int))

    with in_path.open() as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if "kept_reason" not in fieldnames:
            fieldnames.append("kept_reason")

        for row in reader:
            tid = row["tile_id"]
            reg = row["region_id"]
            region_counts[reg]["total"] += 1

            try:
                hist = json.loads(row["pixel_hist_json"])
            except Exception:
                drops["bad_hist"] += 1
                region_counts[reg]["dropped"] += 1
                continue
            total = sum(hist.values())
            if total <= 0:
                drops["empty"] += 1
                region_counts[reg]["dropped"] += 1
                continue
            shares = {int(k): v / total for k, v in hist.items()}
            max_share = max(shares.values())
            fg_share = 1.0 - shares.get(0, 0.0)

            if max_share >= args.mono_threshold:
                drops["monoculture"] += 1
                region_counts[reg]["dropped"] += 1
                continue
            if fg_share < args.min_foreground_share:
                drops["low_foreground"] += 1
                region_counts[reg]["dropped"] += 1
                continue
            if tid in smeared_ids:
                drops["smeared"] += 1
                region_counts[reg]["dropped"] += 1
                continue
            if args.also_drop_likely_shifted and tid in shifted_ids:
                drops["likely_shifted"] += 1
                region_counts[reg]["dropped"] += 1
                continue

            row["kept_reason"] = (
                f"max_class_share={max_share:.2f},fg_share={fg_share:.2f}"
            )
            kept_rows.append(row)
            region_counts[reg]["kept"] += 1

    with out_path.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(kept_rows)

    print(f"Wrote {out_path} with {len(kept_rows)} kept rows")
    print("\nDrops:")
    for k, v in sorted(drops.items(), key=lambda kv: -kv[1]):
        print(f"  {k:>20}: {v}")

    print("\nPer-region (kept / total):")
    for reg in sorted(region_counts):
        c = region_counts[reg]
        print(f"  {reg:<28} {c['kept']:>5} / {c['total']:>5}  ({c['kept']/max(1,c['total'])*100:5.1f}%)")


if __name__ == "__main__":
    main()
