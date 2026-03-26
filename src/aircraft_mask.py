"""Stage 1.2: aircraft-body mask provider for the online pipeline.

The ``AircraftMaskTracker`` returns a per-frame binary mask covering the
visible aircraft body so callers can suppress keypoints that would
otherwise lock onto the fuselage / landing gear / strut.

Current strategy (Step A): **nearest-anchor lookup**. Returns the seed
mask from the temporally closest anchor produced by Stage 0b. This is
what ``scripts/stage0_smoketest.py`` already validated as visually
tracking the aircraft body across 30 s windows on this footage.

Future steps (not yet wired):
  * Step B — sparse OF propagation between anchors (mask edge points
    tracked frame-to-frame, with a fallback to nearest anchor when
    tracking confidence drops).
  * Step C — periodic SAM3 refresh on the current frame in a parallel
    thread, every ~5 s, to catch viewpoint changes the OF chain has
    drifted away from.

The API surface is intentionally minimal so Steps B/C can replace the
internals without breaking callers::

    tracker = AircraftMaskTracker.from_index(Path("data/masks/anchors"))
    mask = tracker.mask_for_frame(frame_idx, frame_shape=(1080, 1920))

The returned mask is uint8, shape ``frame_shape``, values in {0, 255}.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class _Anchor:
    index: int
    frame_index: int
    timestamp_seconds: float
    mask_path: Path


class AircraftMaskTracker:
    """Per-frame aircraft mask provider, currently nearest-anchor."""

    def __init__(self, anchors: Sequence[_Anchor]) -> None:
        if not anchors:
            raise ValueError("AircraftMaskTracker requires at least one anchor")
        self._anchors: list[_Anchor] = sorted(anchors, key=lambda a: a.frame_index)
        self._mask_cache: dict[Path, np.ndarray] = {}

    @classmethod
    def from_index(cls, anchors_dir: Path) -> "AircraftMaskTracker":
        idx_path = Path(anchors_dir) / "index.json"
        payload = json.loads(idx_path.read_text(encoding="utf-8"))
        anchors = [
            _Anchor(
                index=int(a["anchor_index"]),
                frame_index=int(a["frame_index"]),
                timestamp_seconds=float(a["timestamp_seconds"]),
                mask_path=Path(anchors_dir) / a["mask"],
            )
            for a in payload["anchors"]
        ]
        return cls(anchors)

    def num_anchors(self) -> int:
        return len(self._anchors)

    def _nearest(self, frame_idx: int) -> _Anchor:
        # бинарный поиск по отсортированному frame_index
        lo, hi = 0, len(self._anchors) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._anchors[mid].frame_index < frame_idx:
                lo = mid + 1
            else:
                hi = mid
        candidates = {lo}
        if lo > 0:
            candidates.add(lo - 1)
        if lo < len(self._anchors) - 1:
            candidates.add(lo + 1)
        return min(
            (self._anchors[i] for i in candidates),
            key=lambda a: abs(a.frame_index - frame_idx),
        )

    def _load(self, anchor: _Anchor, target_shape: Tuple[int, int]) -> np.ndarray:
        h, w = target_shape
        cached = self._mask_cache.get(anchor.mask_path)
        if cached is not None and cached.shape[:2] == (h, w):
            return cached
        m = cv2.imread(str(anchor.mask_path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            m = np.zeros((h, w), dtype=np.uint8)
        elif m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        # Нормируем к {0, 255}
        m = (m > 127).astype(np.uint8) * 255
        self._mask_cache[anchor.mask_path] = m
        return m

    def mask_for_frame(
        self,
        frame_idx: int,
        frame_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Return uint8 mask for the given frame index, sized to frame_shape (H, W)."""
        anchor = self._nearest(frame_idx)
        return self._load(anchor, frame_shape)

    def anchor_for_frame(self, frame_idx: int) -> _Anchor:
        """Returns the anchor used by ``mask_for_frame`` (for diagnostics/HUD)."""
        return self._nearest(frame_idx)


def apply_mask_to_image(img: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Zero out pixels under ``mask`` (uint8 0/255). Returns a copy if mask
    is non-empty, otherwise returns the input unchanged.

    Used as a pre-filter into LightGlue/LoFTR/ORB: a textureless black
    region under the keypoint detector produces no features, so no
    keypoints can land on the aircraft body.
    """
    if mask is None or not mask.any():
        return img
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = img.copy()
    if out.ndim == 2:
        out[mask > 0] = 0
    else:
        out[mask > 0] = (0, 0, 0)
    return out


def filter_keypoints_by_mask(
    pts: np.ndarray,
    mask: Optional[np.ndarray],
) -> np.ndarray:
    """Return a boolean array of shape ``(len(pts),)`` selecting keypoints
    that are *outside* the mask (i.e. safe to keep). If mask is None or
    empty, all pts are kept.
    """
    n = 0 if pts is None else len(pts)
    if n == 0 or mask is None or not mask.any():
        return np.ones(n, dtype=bool)
    h, w = mask.shape[:2]
    xs = np.clip(pts[:, 0].astype(np.int32), 0, w - 1)
    ys = np.clip(pts[:, 1].astype(np.int32), 0, h - 1)
    return mask[ys, xs] == 0
