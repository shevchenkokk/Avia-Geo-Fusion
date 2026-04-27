from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_bar(path: Path, title: str, labels: list[str], values: list[float], color: str = "#2c7fb8") -> None:
    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, color=color)
    plt.title(title)
    plt.ylabel("Count")
    plt.xticks(rotation=25, ha="right")

    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2.0, b.get_height(), f"{int(v)}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_qc_chart(path: Path, qc: dict) -> None:
    labels = [
        "missing_files",
        "size_mismatch",
        "likely_shifted",
        "empty_tiles",
        "smeared_tiles",
        "low_structural",
        "rejected_tiles",
        "duplicate_images_groups",
        "duplicate_masks_groups",
    ]
    values = [
        len(qc.get("missing_files", [])),
        len(qc.get("size_mismatch", [])),
        len(qc.get("likely_shifted", [])),
        len(qc.get("empty_tiles", [])),
        len(qc.get("smeared_tiles", [])),
        len(qc.get("low_structural", [])),
        len(qc.get("rejected_tiles", [])),
        len(qc.get("duplicate_images", [])),
        len(qc.get("duplicate_masks", [])),
    ]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, color="#d95f02")
    plt.title("QC Findings")
    plt.ylabel("Count")
    plt.xticks(rotation=25, ha="right")

    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2.0, b.get_height(), str(v), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def parse_hist(meta_rows: list[dict[str, str]]) -> dict[int, int]:
    hist = defaultdict(int)
    for row in meta_rows:
        raw = row.get("pixel_hist_json", "{}")
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for k, v in d.items():
            hist[int(k)] += int(v)
    return dict(sorted(hist.items()))


def make_markdown(
    report_path: Path,
    dataset_root: Path,
    n_tiles: int,
    n_patches: int,
    split_counts: dict[str, int],
    region_counts: dict[str, int],
    class_hist: dict[int, int],
    qc: dict,
) -> None:
    total_px = sum(class_hist.values())
    fg_px = total_px - class_hist.get(0, 0)
    fg_ratio = fg_px / max(1, total_px)

    lines = []
    lines.append("# Defense Report: Overture Dataset (Starter, Zoom 17)")
    lines.append("")
    lines.append("## 1) Что собрано")
    lines.append(f"- Корень датасета: {dataset_root}")
    lines.append(f"- Всего пар tile image/mask: {n_tiles}")
    lines.append(f"- Всего патчей 512x512: {n_patches}")
    lines.append("")
    lines.append("## 2) Географическое покрытие")
    for region, cnt in sorted(region_counts.items()):
        lines.append(f"- {region}: {cnt} tiles")
    lines.append("")
    lines.append("## 3) Классы и схема")
    lines.append("- 0: background")
    lines.append("- 1: water")
    lines.append("- 2: vegetation")
    lines.append("- 3: buildings")
    lines.append("- 4: roads_runway_taxiway")
    lines.append("")
    lines.append("Пиксельная статистика по tile masks:")
    for cid, px in class_hist.items():
        frac = px / max(1, total_px)
        lines.append(f"- class {cid}: {px} px ({frac:.4%})")
    lines.append(f"- foreground ratio (class > 0): {fg_ratio:.4%}")
    lines.append("")
    lines.append("## 4) Split для обучения")
    for split, cnt in split_counts.items():
        lines.append(f"- {split}: {cnt}")
    lines.append("")
    lines.append("## 5) QC")
    lines.append(f"- total_pairs: {qc.get('total_pairs', 0)}")
    lines.append(f"- missing_files: {len(qc.get('missing_files', []))}")
    lines.append(f"- size_mismatch: {len(qc.get('size_mismatch', []))}")
    lines.append(f"- likely_shifted: {len(qc.get('likely_shifted', []))}")
    lines.append(f"- empty_tiles: {len(qc.get('empty_tiles', []))}")
    lines.append(f"- smeared_tiles: {len(qc.get('smeared_tiles', []))}")
    lines.append(f"- low_structural: {len(qc.get('low_structural', []))}")
    lines.append(f"- rejected_tiles: {len(qc.get('rejected_tiles', []))}")
    lines.append(f"- duplicate_images_groups: {len(qc.get('duplicate_images', []))}")
    lines.append(f"- duplicate_masks_groups: {len(qc.get('duplicate_masks', []))}")
    lines.append("")
    lines.append("## 6) Что показать на защите")
    lines.append("- preview_mosaic.png: примеры image / color mask / overlay")
    lines.append("- region_coverage.png: сколько тайлов по каждому региону")
    lines.append("- class_distribution_tiles.png: дисбаланс классов")
    lines.append("- split_counts.png: train/val/test")
    lines.append("- qc_summary.png: результаты контроля качества")
    lines.append("")
    lines.append("## 7) Где лежат артефакты")
    lines.append("- results/defense_dataset_starter/")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    dataset_root = Path("data/overture_ru_dataset_starter")
    out_root = Path("results/defense_dataset_starter")
    out_root.mkdir(parents=True, exist_ok=True)

    meta_path = dataset_root / "meta.csv"
    patches_path = dataset_root / "meta_patches.csv"
    qc_path = dataset_root / "qc_report.json"

    if not meta_path.exists() or not patches_path.exists() or not qc_path.exists():
        raise FileNotFoundError("Expected meta.csv, meta_patches.csv and qc_report.json in dataset root")

    meta_rows = load_csv(meta_path)
    patch_rows = load_csv(patches_path)
    qc = json.loads(qc_path.read_text(encoding="utf-8"))

    region_counts = Counter(r.get("region_id", "unknown") for r in meta_rows)
    split_counts = Counter(r.get("split", "unknown") for r in patch_rows)
    class_hist = parse_hist(meta_rows)

    # Charts
    save_bar(
        out_root / "region_coverage.png",
        "Tiles by Region",
        list(region_counts.keys()),
        [region_counts[k] for k in region_counts.keys()],
        color="#1b9e77",
    )

    save_bar(
        out_root / "split_counts.png",
        "Patches Split (train/val/test)",
        list(split_counts.keys()),
        [split_counts[k] for k in split_counts.keys()],
        color="#7570b3",
    )

    cls_labels = [f"{cid}" for cid in class_hist.keys()]
    cls_values = [class_hist[cid] for cid in class_hist.keys()]
    save_bar(
        out_root / "class_distribution_tiles.png",
        "Class Pixel Distribution (tile masks)",
        cls_labels,
        cls_values,
        color="#66a61e",
    )

    save_qc_chart(out_root / "qc_summary.png", qc)

    # Copy/mention preview mosaic if exists
    mosaic_src = dataset_root / "quality_preview" / "preview_mosaic.png"
    if mosaic_src.exists():
        mosaic_dst = out_root / "preview_mosaic.png"
        mosaic_dst.write_bytes(mosaic_src.read_bytes())

    make_markdown(
        out_root / "defense_report.md",
        dataset_root,
        n_tiles=len(meta_rows),
        n_patches=len(patch_rows),
        split_counts=dict(split_counts),
        region_counts=dict(region_counts),
        class_hist=class_hist,
        qc=qc,
    )

    print("Defense pack created:", out_root)


if __name__ == "__main__":
    main()
