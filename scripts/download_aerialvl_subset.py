"""Выборочная загрузка небольшого VAL-поднабора AerialVL из split ZIP-архива.

На Tsinghua Cloud датасет VAL лежит как многотомный архив `VAL.zip.001`...
`VAL.zip.027`. Целиком он занимает около 56 ГБ, но для первого прогона полного
пайплайна нам нужна только папка `geo_referenced_map` и одна последовательность
из `short_trajtr` или `long_trajtr`.

Скрипт читает центральный каталог ZIP через HTTP Range, находит нужные файлы и
скачивает только их сжатые данные. Затем каждый файл распаковывается отдельно.
Так можно получить 100-200 кадров для smoke-проверки, не скачивая весь архив.
"""

from __future__ import annotations

import argparse
import json
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import requests

TSINGHUA_TOKEN = "68c3a4ed24cc40f1a7da"
TSINGHUA_ROOT = "https://cloud.tsinghua.edu.cn"
VAL_DIR = "/images/VAL/"
TAIL_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class PartInfo:
    name: str
    path: str
    size: int
    start: int
    end: int


@dataclass(frozen=True)
class ZipEntry:
    name: str
    method: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int


def _dirents(path: str) -> list[dict]:
    url = f"{TSINGHUA_ROOT}/api/v2.1/share-links/{TSINGHUA_TOKEN}/dirents/"
    resp = requests.get(url, params={"path": path}, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    if "dirent_list" not in payload:
        raise RuntimeError(f"Неожиданный ответ Seafile API: {payload}")
    return payload["dirent_list"]


def _val_parts() -> list[PartInfo]:
    items = _dirents(VAL_DIR)
    parts_raw = [
        item
        for item in items
        if not item.get("is_dir") and item["file_name"].startswith("VAL.zip.")
    ]
    parts_raw.sort(key=lambda item: item["file_name"])
    out: list[PartInfo] = []
    cursor = 0
    for item in parts_raw:
        size = int(item["size"])
        out.append(
            PartInfo(
                name=item["file_name"],
                path=item["file_path"],
                size=size,
                start=cursor,
                end=cursor + size,
            )
        )
        cursor += size
    if not out:
        raise RuntimeError("Не нашёл VAL.zip.* в Tsinghua Cloud")
    return out


def _file_url(path: str) -> str:
    quoted = quote(path, safe="/")
    return f"{TSINGHUA_ROOT}/d/{TSINGHUA_TOKEN}/files/?p={quoted}&dl=1"


def _download_part_range(part: PartInfo, start_in_part: int, end_in_part: int) -> bytes:
    if start_in_part < 0 or end_in_part > part.size or end_in_part <= start_in_part:
        raise ValueError(
            f"Некорректный диапазон для {part.name}: {start_in_part}:{end_in_part}"
        )
    headers = {"Range": f"bytes={start_in_part}-{end_in_part - 1}"}
    resp = requests.get(_file_url(part.path), headers=headers, timeout=180)
    resp.raise_for_status()
    data = resp.content
    expected = end_in_part - start_in_part
    if len(data) != expected:
        raise RuntimeError(
            f"{part.name}: ожидал {expected} байт, получил {len(data)} байт"
        )
    return data


def _download_absolute_range(parts: list[PartInfo], start: int, size: int) -> bytes:
    end = start + size
    chunks: list[bytes] = []
    for part in parts:
        if part.end <= start or part.start >= end:
            continue
        local_start = max(start, part.start) - part.start
        local_end = min(end, part.end) - part.start
        chunks.append(_download_part_range(part, local_start, local_end))
    data = b"".join(chunks)
    if len(data) != size:
        raise RuntimeError(
            f"Диапазон {start}:{end}: ожидал {size} байт, получил {len(data)}"
        )
    return data


def _parse_zip64_eocd(tail: bytes) -> tuple[int, int, int]:
    pos = tail.rfind(b"PK\x06\x06")
    if pos < 0:
        raise RuntimeError("Не нашёл ZIP64 EOCD в хвосте архива")
    fixed = tail[pos : pos + 56]
    if len(fixed) < 56:
        raise RuntimeError("ZIP64 EOCD обрезан")
    values = struct.unpack("<4sQ2H2L4Q", fixed)
    entries_total = int(values[7])
    cd_size = int(values[8])
    cd_offset = int(values[9])
    return entries_total, cd_size, cd_offset


def _apply_zip64_extra(
    extra: bytes, comp: int, uncomp: int, offset: int
) -> tuple[int, int, int]:
    cursor = 0
    while cursor + 4 <= len(extra):
        header_id, data_size = struct.unpack("<HH", extra[cursor : cursor + 4])
        body = extra[cursor + 4 : cursor + 4 + data_size]
        if header_id == 0x0001:
            r = 0
            if uncomp == 0xFFFFFFFF and r + 8 <= len(body):
                uncomp = struct.unpack("<Q", body[r : r + 8])[0]
                r += 8
            if comp == 0xFFFFFFFF and r + 8 <= len(body):
                comp = struct.unpack("<Q", body[r : r + 8])[0]
                r += 8
            if offset == 0xFFFFFFFF and r + 8 <= len(body):
                offset = struct.unpack("<Q", body[r : r + 8])[0]
        cursor += 4 + data_size
    return int(comp), int(uncomp), int(offset)


def _parse_central_directory(cd: bytes) -> list[ZipEntry]:
    entries: list[ZipEntry] = []
    cursor = 0
    while cursor < len(cd):
        if cd[cursor : cursor + 4] != b"PK\x01\x02":
            raise RuntimeError(
                f"Некорректная запись central directory на смещении {cursor}"
            )
        fixed = cd[cursor : cursor + 46]
        values = struct.unpack("<4s6H3L5H2L", fixed)
        method = int(values[4])
        comp = int(values[8])
        uncomp = int(values[9])
        name_len = int(values[10])
        extra_len = int(values[11])
        comment_len = int(values[12])
        offset = int(values[16])
        name_start = cursor + 46
        name_end = name_start + name_len
        extra_end = name_end + extra_len
        name = cd[name_start:name_end].decode("utf-8", errors="replace")
        extra = cd[name_end:extra_end]
        comp, uncomp, offset = _apply_zip64_extra(extra, comp, uncomp, offset)
        entries.append(
            ZipEntry(
                name=name,
                method=method,
                compressed_size=comp,
                uncompressed_size=uncomp,
                local_header_offset=offset,
            )
        )
        cursor = extra_end + comment_len
    return entries


def _load_entries(parts: list[PartInfo], cache_path: Path | None) -> list[ZipEntry]:
    if cache_path is not None and cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return [ZipEntry(**item) for item in payload["entries"]]

    total_size = parts[-1].end
    last = parts[-1]
    tail_size = min(TAIL_BYTES, last.size)
    tail = _download_part_range(last, last.size - tail_size, last.size)
    _, cd_size, cd_offset = _parse_zip64_eocd(tail)
    cd = _download_absolute_range(parts, cd_offset, cd_size)
    entries = _parse_central_directory(cd)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "total_size": total_size,
                    "entries": [entry.__dict__ for entry in entries],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return entries


def _selected_entries(
    entries: Iterable[ZipEntry],
    sequence: str,
    max_frames: int,
) -> list[ZipEntry]:
    seq_prefix = sequence.strip("/") + "/"
    selected: list[ZipEntry] = []
    frame_count = 0
    for entry in entries:
        if entry.name.endswith("/"):
            continue
        if entry.name.startswith("geo_referenced_map/"):
            selected.append(entry)
            continue
        if entry.name.startswith(seq_prefix):
            if max_frames <= 0 or frame_count < max_frames:
                selected.append(entry)
                frame_count += 1
    selected.sort(key=lambda item: item.local_header_offset)
    return selected


def _read_compressed_payload(parts: list[PartInfo], entry: ZipEntry) -> bytes:
    local = _download_absolute_range(parts, entry.local_header_offset, 30)
    if local[:4] != b"PK\x03\x04":
        raise RuntimeError(f"{entry.name}: некорректный local header")
    values = struct.unpack("<4s5H3L2H", local)
    name_len = int(values[9])
    extra_len = int(values[10])
    data_offset = entry.local_header_offset + 30 + name_len + extra_len
    return _download_absolute_range(parts, data_offset, entry.compressed_size)


def _decompress(entry: ZipEntry, payload: bytes) -> bytes:
    if entry.method == 0:
        return payload
    if entry.method == 8:
        return zlib.decompress(payload, -15)
    raise RuntimeError(f"{entry.name}: неподдерживаемый ZIP-метод {entry.method}")


def _write_entry(output_root: Path, entry: ZipEntry, content: bytes) -> None:
    target = output_root / entry.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)


def _human_mb(value: int) -> str:
    return f"{value / 1024 / 1024:.1f} МБ"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/external/aerialvl")
    )
    parser.add_argument(
        "--sequence",
        default="short_trajtr/2023-03-11-11-48-35",
        help="какую VAL-последовательность скачать",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=150,
        help="сколько кадров скачать из последовательности; 0 означает все кадры",
    )
    parser.add_argument(
        "--cache", type=Path, default=Path("data/aerialvl/val_zip_index.json")
    )
    parser.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    print("[aerialvl-download] читаю список частей VAL.zip.*")
    parts = _val_parts()
    total_size = parts[-1].end
    print(
        f"[aerialvl-download] частей: {len(parts)}, общий размер: {_human_mb(total_size)}"
    )

    print("[aerialvl-download] читаю индекс ZIP")
    entries = _load_entries(parts, args.cache)
    selected = _selected_entries(entries, args.sequence, args.max_frames)
    if not selected:
        raise RuntimeError(f"Не нашёл файлы для {args.sequence!r}")
    total_comp = sum(entry.compressed_size for entry in selected)
    print(f"[aerialvl-download] файлов к скачиванию: {len(selected)}")
    print(f"[aerialvl-download] сжатый объём: {_human_mb(total_comp)}")

    done_bytes = 0
    t0 = time.time()
    for i, entry in enumerate(selected, start=1):
        target = args.output_root / entry.name
        if (
            args.skip_existing
            and target.exists()
            and target.stat().st_size == entry.uncompressed_size
        ):
            done_bytes += entry.compressed_size
            continue
        payload = _read_compressed_payload(parts, entry)
        content = _decompress(entry, payload)
        if len(content) != entry.uncompressed_size:
            raise RuntimeError(
                f"{entry.name}: после распаковки ожидал {entry.uncompressed_size}, получил {len(content)}"
            )
        _write_entry(args.output_root, entry, content)
        done_bytes += entry.compressed_size
        if i % 10 == 0 or i == len(selected):
            dt = max(time.time() - t0, 1e-6)
            print(
                f"[aerialvl-download] {i}/{len(selected)}  "
                f"{_human_mb(done_bytes)}/{_human_mb(total_comp)}  "
                f"{done_bytes / 1024 / 1024 / dt:.2f} МБ/с"
            )

    print(f"[aerialvl-download] готово: {args.output_root}")


if __name__ == "__main__":
    main()
