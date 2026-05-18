"""Скачивание и выборочная распаковка публичного sample-поднабора VPAIR.

VPAIR sample опубликован на Zenodo как один ZIP (~0.86 ГБ). Скрипт скачивает
архив, а затем извлекает только нужные для VPR-валидации файлы:

  - ``poses_query.txt`` и ``poses_reference_view.txt``;
  - ``queries/*.png``;
  - ``reference_views/*.png``;
  - опционально ``distractors/*.png``;
  - depth ``.npy`` по умолчанию пропускается, потому что для текущего retriever
    benchmark он не нужен.

После распаковки можно выполнить ``scripts/prepare_vpair.py``.
"""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

import requests

ZENODO_SAMPLE_URL = (
    "https://zenodo.org/records/6473989/files/vpair_sample.zip?download=1"
)
CHUNK_SIZE = 8 * 1024 * 1024


def _human_mb(value: int | float) -> str:
    return f"{float(value) / 1024.0 / 1024.0:.1f} МБ"


def _download(url: str, target: Path, force: bool = False) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    headers: dict[str, str] = {}
    mode = "wb"
    existing = target.stat().st_size if target.exists() else 0
    if target.exists() and not force and existing > 0:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"

    with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
        # Некоторые хранилища игнорируют Range и возвращают 200 вместо 206.
        if resp.status_code == 200 and mode == "ab":
            mode = "wb"
            existing = 0
        resp.raise_for_status()
        total_header = resp.headers.get("Content-Length")
        total = int(total_header) + existing if total_header else None

        done = existing
        t0 = time.time()
        last_log = t0
        with target.open(mode) as f:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - last_log > 2.0:
                    speed = (done - existing) / max(now - t0, 1e-6)
                    suffix = f"/{_human_mb(total)}" if total else ""
                    print(
                        f"[vpair-download] {_human_mb(done)}{suffix} "
                        f"{_human_mb(speed)}/s",
                        flush=True,
                    )
                    last_log = now
    print(
        f"[vpair-download] archive ready: {target} ({_human_mb(target.stat().st_size)})"
    )


def _strip_top_level(name: str) -> str:
    parts = Path(name).parts
    if len(parts) >= 2 and parts[0] == "vpair_sample":
        return str(Path(*parts[1:]))
    return name


def _want_member(name: str, include_distractors: bool, include_depth: bool) -> bool:
    rel = _strip_top_level(name)
    if not rel or rel.endswith("/"):
        return False
    path = Path(rel)
    if rel in {"poses_query.txt", "poses_reference_view.txt"}:
        return True
    if (
        len(path.parts) >= 2
        and path.parts[0] == "queries"
        and path.suffix.lower() == ".png"
    ):
        return True
    if len(path.parts) >= 2 and path.parts[0] == "reference_views":
        if path.suffix.lower() == ".png":
            return True
        if (
            include_depth
            and path.suffix.lower() == ".npy"
            and path.name.startswith("depth_")
        ):
            return True
    if include_distractors and len(path.parts) >= 2 and path.parts[0] == "distractors":
        return path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    return False


def _extract_selected(
    archive: Path,
    output_root: Path,
    include_distractors: bool,
    include_depth: bool,
    skip_existing: bool,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        selected = [
            info
            for info in zf.infolist()
            if _want_member(info.filename, include_distractors, include_depth)
        ]
        if not selected:
            raise RuntimeError(f"В архиве {archive} не найдено файлов VPAIR")
        total = sum(info.file_size for info in selected)
        print(
            f"[vpair-download] files to extract: {len(selected)} ({_human_mb(total)})"
        )
        done = 0
        t0 = time.time()
        for i, info in enumerate(selected, start=1):
            rel = _strip_top_level(info.filename)
            target = output_root / rel
            if (
                skip_existing
                and target.exists()
                and target.stat().st_size == info.file_size
            ):
                done += info.file_size
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, target.open("wb") as dst:
                    while True:
                        chunk = src.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        dst.write(chunk)
                done += info.file_size
            if i % 25 == 0 or i == len(selected):
                speed = done / max(time.time() - t0, 1e-6)
                print(
                    f"[vpair-download] extracted {i}/{len(selected)} "
                    f"{_human_mb(done)}/{_human_mb(total)} "
                    f"{_human_mb(speed)}/s",
                    flush=True,
                )
    print(f"[vpair-download] dataset root: {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=ZENODO_SAMPLE_URL)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/downloads/vpair_sample.zip"),
        help="куда сохранить ZIP",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/external/vpair_sample"),
        help="каталог датасета после strip top-level",
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--include-distractors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="извлекать distractors/*.png для более строгого VPR benchmark",
    )
    parser.add_argument(
        "--include-depth",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="извлекать reference_views/depth_*.npy; для текущего DINOv2 VPR не нужно",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    try:
        _download(args.url, args.archive, force=args.force_download)
        _extract_selected(
            archive=args.archive,
            output_root=args.output_root,
            include_distractors=args.include_distractors,
            include_depth=args.include_depth,
            skip_existing=args.skip_existing,
        )
    except KeyboardInterrupt:
        print("\n[vpair-download] interrupted", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
