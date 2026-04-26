from __future__ import annotations

from pathlib import Path


def discover_videos(search_roots: list[Path] | None = None) -> list[Path]:
    roots = search_roots or [Path("data/videos"), Path(".")]
    seen: set[str] = set()
    videos: list[Path] = []

    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for pattern in ("*.MP4", "*.mp4"):
            for video_path in sorted(root.glob(pattern)):
                key = str(video_path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                videos.append(video_path)

    videos.sort(key=lambda p: (p.name.lower(), str(p)))
    return videos


def pick_default_video(preferred_names: tuple[str, ...] = ("GP010269.MP4", "GOPR0269.MP4")) -> Path | None:
    videos = discover_videos()
    if not videos:
        return None

    by_name = {p.name: p for p in videos}
    for preferred_name in preferred_names:
        if preferred_name in by_name:
            return by_name[preferred_name]

    return videos[0]