"""Маска корпуса самолёта для основного пайплайна.

``AircraftMaskTracker`` для каждого кадра возвращает бинарную маску видимой
части самолёта. По ней можно отфильтровать ключевые точки, которые иначе
цеплялись бы за фюзеляж, шасси или стойки.

Текущая схема: **поиск ближайшего якоря + распространение из шага B**.
Если передан сам кадр, трекер переносит предыдущую якорную маску разреженным
оптическим потоком Лукаса-Канаде. Если уверенность падает, берётся ближайшая
якорная маска.

Интерфейс специально оставлен небольшим, чтобы шаги B/C можно было заменить
внутри без правок в вызывающем коде::

    tracker = AircraftMaskTracker.from_index(Path("data/masks/anchors"))
    mask = tracker.mask_for_frame(frame_idx, frame_shape=(1080, 1920), frame=frame)

Возвращается маска uint8 размера ``frame_shape`` со значениями {0, 255}.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class _Anchor:
    index: int
    frame_index: int
    timestamp_seconds: float
    mask_path: Path
    frame_path: Optional[Path] = None


@dataclass(frozen=True)
class MaskDiagnostics:
    frame_idx: int
    anchor_index: int
    anchor_frame_index: int
    frame_delta: int
    method: str
    confidence: float
    num_seed_points: int = 0
    num_tracked_points: int = 0
    median_flow_px: float = 0.0
    coverage: float = 0.0
    reject_reason: str = ""


@dataclass
class _PropagationState:
    frame_idx: int
    gray: np.ndarray
    mask: np.ndarray
    points: np.ndarray
    anchor: _Anchor
    confidence: float = 1.0
    diagnostics: MaskDiagnostics = field(default_factory=lambda: MaskDiagnostics(
        frame_idx=-1,
        anchor_index=-1,
        anchor_frame_index=-1,
        frame_delta=0,
        method="uninitialized",
        confidence=0.0,
    ))


class AircraftMaskTracker:
    """Покадровая маска самолёта с переносом по оптическому потоку."""

    def __init__(
        self,
        anchors: Sequence[_Anchor],
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        if not anchors:
            raise ValueError("AircraftMaskTracker requires at least one anchor")
        self._anchors: list[_Anchor] = sorted(anchors, key=lambda a: a.frame_index)
        self.metadata: dict[str, Any] = metadata or {}
        self._mask_cache: dict[Path, np.ndarray] = {}
        self._frame_cache: dict[Path, np.ndarray] = {}
        self._state: Optional[_PropagationState] = None
        self.last_diagnostics = MaskDiagnostics(
            frame_idx=-1,
            anchor_index=-1,
            anchor_frame_index=-1,
            frame_delta=0,
            method="uninitialized",
            confidence=0.0,
        )
        self.lk_params = dict(
            winSize=(31, 31),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.max_step_frames = 45
        self.min_seed_points = 16
        self.min_track_points = 8
        self.min_track_ratio = 0.35
        self.max_fb_error_px = 1.5
        self.max_median_flow_px = 80.0
        self.coverage_min = 0.01
        self.coverage_max = 0.25

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
                frame_path=Path(anchors_dir) / a["frame"] if a.get("frame") else None,
            )
            for a in payload["anchors"]
        ]
        return cls(anchors, metadata=payload)

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

    def _load_anchor_gray(self, anchor: _Anchor, target_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        if anchor.frame_path is None:
            return None
        cached = self._frame_cache.get(anchor.frame_path)
        h, w = target_shape
        if cached is not None and cached.shape[:2] == (h, w):
            return cached
        frame = cv2.imread(str(anchor.frame_path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            return None
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        self._frame_cache[anchor.frame_path] = frame
        return frame

    def _seed_points_from_mask(self, mask: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        edge_pts = []
        for contour in contours:
            if len(contour) == 0:
                continue
            step = max(1, len(contour) // 120)
            edge_pts.append(contour[::step, 0, :])
        pts = np.vstack(edge_pts).astype(np.float32) if edge_pts else np.empty((0, 2), dtype=np.float32)

        x, y, w, h = cv2.boundingRect(mask)
        roi = mask[y:y + h, x:x + w]
        dist = cv2.distanceTransform((roi > 0).astype(np.uint8), cv2.DIST_L2, 3)
        if dist.size:
            ys, xs = np.where(dist >= max(2.0, float(dist.max()) * 0.25))
            if len(xs) > 0:
                step = max(1, len(xs) // 120)
                inside = np.column_stack([xs[::step] + x, ys[::step] + y]).astype(np.float32)
                pts = np.vstack([pts, inside]) if len(pts) else inside

        if len(pts) == 0:
            return np.empty((0, 1, 2), dtype=np.float32)
        _, unique_idx = np.unique(np.round(pts).astype(np.int32), axis=0, return_index=True)
        pts = pts[np.sort(unique_idx)]
        return pts.reshape(-1, 1, 2).astype(np.float32)

    def _seed_points(self, gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=240,
            qualityLevel=0.01,
            minDistance=8,
            mask=(mask > 0).astype(np.uint8) * 255,
        )
        if points is None or len(points) < self.min_seed_points:
            return self._seed_points_from_mask(mask)
        return points.astype(np.float32)

    def _warp_mask(
        self,
        mask: np.ndarray,
        prev_points: np.ndarray,
        curr_points: np.ndarray,
        target_shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        if len(prev_points) < 3 or len(curr_points) < 3:
            return None
        affine, inliers = cv2.estimateAffinePartial2D(
            prev_points.astype(np.float32),
            curr_points.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=1000,
            confidence=0.99,
        )
        if affine is None or inliers is None or int(inliers.sum()) < self.min_track_points:
            return None
        h, w = target_shape
        warped = cv2.warpAffine(
            mask,
            affine,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped = (warped > 127).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        return cv2.morphologyEx(warped, cv2.MORPH_CLOSE, kernel, iterations=1)

    def _anchor_state(
        self,
        anchor: _Anchor,
        frame_idx: int,
        frame_shape: Tuple[int, int],
    ) -> _PropagationState:
        mask = self._load(anchor, frame_shape)
        gray = self._load_anchor_gray(anchor, frame_shape)
        if gray is None:
            gray = np.zeros(frame_shape, dtype=np.uint8)
        points = self._seed_points(gray, mask)
        diag = MaskDiagnostics(
            frame_idx=frame_idx,
            anchor_index=anchor.index,
            anchor_frame_index=anchor.frame_index,
            frame_delta=frame_idx - anchor.frame_index,
            method="anchor",
            confidence=1.0,
            num_seed_points=len(points),
            num_tracked_points=len(points),
            coverage=_mask_coverage(mask),
        )
        return _PropagationState(
            frame_idx=anchor.frame_index,
            gray=gray,
            mask=mask,
            points=points,
            anchor=anchor,
            confidence=1.0,
            diagnostics=diag,
        )

    def _fallback(
        self,
        frame_idx: int,
        gray: np.ndarray,
        reason: str,
    ) -> np.ndarray:
        anchor = self._nearest(frame_idx)
        mask = self._load(anchor, gray.shape[:2])
        points = self._seed_points(gray, mask)
        self._state = _PropagationState(
            frame_idx=frame_idx,
            gray=gray,
            mask=mask,
            points=points,
            anchor=anchor,
            confidence=0.0,
        )
        self.last_diagnostics = MaskDiagnostics(
            frame_idx=frame_idx,
            anchor_index=anchor.index,
            anchor_frame_index=anchor.frame_index,
            frame_delta=frame_idx - anchor.frame_index,
            method="fallback_anchor",
            confidence=0.0,
            num_seed_points=len(points),
            num_tracked_points=0,
            coverage=_mask_coverage(mask),
            reject_reason=reason,
        )
        return mask

    def _should_restart_from_anchor(self, frame_idx: int, anchor: _Anchor) -> bool:
        if self._state is None:
            return True
        if self._state.anchor.index > anchor.index:
            return True
        if self._state.anchor.index < anchor.index and frame_idx >= anchor.frame_index:
            return True
        if frame_idx < self._state.frame_idx:
            return True
        if frame_idx - self._state.frame_idx > self.max_step_frames:
            return True
        return False

    def _propagate_from_state(
        self,
        state: _PropagationState,
        frame_idx: int,
        gray: np.ndarray,
    ) -> Optional[_PropagationState]:
        seed_count = len(state.points)
        if seed_count < self.min_seed_points:
            return None
        curr_pts, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
            state.gray, gray, state.points, None, **self.lk_params,
        )
        if curr_pts is None or st_fwd is None:
            return None
        back_pts, st_bwd, _ = cv2.calcOpticalFlowPyrLK(
            gray, state.gray, curr_pts, None, **self.lk_params,
        )
        if back_pts is None or st_bwd is None:
            return None
        fb_err = np.linalg.norm(back_pts.reshape(-1, 2) - state.points.reshape(-1, 2), axis=1)
        ok = (st_fwd.ravel() == 1) & (st_bwd.ravel() == 1) & (fb_err < self.max_fb_error_px)
        tracked_count = int(ok.sum())
        if tracked_count < self.min_track_points:
            return None
        ratio = tracked_count / max(seed_count, 1)
        if ratio < self.min_track_ratio:
            return None
        prev_kept = state.points.reshape(-1, 2)[ok]
        curr_kept = curr_pts.reshape(-1, 2)[ok]
        flow = curr_kept - prev_kept
        median_flow = float(np.median(np.linalg.norm(flow, axis=1))) if len(flow) else 0.0
        if median_flow > self.max_median_flow_px:
            return None
        mask = self._warp_mask(state.mask, prev_kept, curr_kept, gray.shape[:2])
        if mask is None:
            return None
        coverage = _mask_coverage(mask)
        if not (self.coverage_min <= coverage <= self.coverage_max):
            return None
        points = self._seed_points(gray, mask)
        confidence = float(np.clip(ratio * math.exp(-median_flow / 120.0), 0.0, 1.0))
        diag = MaskDiagnostics(
            frame_idx=frame_idx,
            anchor_index=state.anchor.index,
            anchor_frame_index=state.anchor.frame_index,
            frame_delta=frame_idx - state.anchor.frame_index,
            method="of_propagated",
            confidence=confidence,
            num_seed_points=seed_count,
            num_tracked_points=tracked_count,
            median_flow_px=median_flow,
            coverage=coverage,
        )
        return _PropagationState(
            frame_idx=frame_idx,
            gray=gray,
            mask=mask,
            points=points,
            anchor=state.anchor,
            confidence=confidence,
            diagnostics=diag,
        )

    def mask_for_frame(
        self,
        frame_idx: int,
        frame_shape: Tuple[int, int],
        frame: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Вернуть uint8-маску для заданного кадра в размере frame_shape (H, W)."""
        if frame is not None:
            return self.update(frame_idx, frame)
        anchor = self._nearest(frame_idx)
        mask = self._load(anchor, frame_shape)
        self.last_diagnostics = MaskDiagnostics(
            frame_idx=frame_idx,
            anchor_index=anchor.index,
            anchor_frame_index=anchor.frame_index,
            frame_delta=frame_idx - anchor.frame_index,
            method="nearest_anchor",
            confidence=1.0,
            coverage=_mask_coverage(mask),
        )
        return mask

    def update(self, frame_idx: int, frame: np.ndarray) -> np.ndarray:
        """Вернуть маску текущего кадра, по возможности перенесённую от ближайшего якоря."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        anchor = self._nearest(frame_idx)
        if self._should_restart_from_anchor(frame_idx, anchor):
            base = self._anchor_state(anchor, frame_idx, gray.shape[:2])
            if anchor.frame_index == frame_idx:
                self._state = _PropagationState(
                    frame_idx=frame_idx,
                    gray=gray,
                    mask=base.mask,
                    points=base.points,
                    anchor=anchor,
                    confidence=1.0,
                    diagnostics=base.diagnostics,
                )
                self.last_diagnostics = base.diagnostics
                return base.mask
            propagated = self._propagate_from_state(base, frame_idx, gray)
        else:
            propagated = self._propagate_from_state(self._state, frame_idx, gray)

        if propagated is None:
            return self._fallback(frame_idx, gray, "propagation_low_confidence")
        self._state = propagated
        self.last_diagnostics = propagated.diagnostics
        return propagated.mask

    def anchor_for_frame(self, frame_idx: int) -> _Anchor:
        """Вернуть ближайший якорь для диагностики и HUD."""
        return self._nearest(frame_idx)


def _resolve_index_path(value: str | Path, *, repo_root: Path) -> Path:
    """Разрешить путь из index.json относительно cwd или корня репозитория."""
    path = Path(value).expanduser()
    if path.is_absolute():
        base = path
    else:
        candidates = [Path.cwd() / path, repo_root / path]
        base = next((c for c in candidates if c.exists()), candidates[0])
    try:
        return base.resolve(strict=True)
    except FileNotFoundError:
        return base.resolve(strict=False)


def _same_source_path(a: str | Path, b: str | Path, *, repo_root: Path) -> bool:
    return _resolve_index_path(a, repo_root=repo_root) == _resolve_index_path(
        b,
        repo_root=repo_root,
    )


def load_aircraft_mask_tracker_for_video(
    anchors_dir: Path,
    video_path: Path,
    *,
    allow_video_mismatch: bool = False,
    warn: Optional[Callable[[str], None]] = print,
) -> Optional[AircraftMaskTracker]:
    """Загрузить anchor-трекер только если index соответствует текущему видео.

    Anchor-маски — это калибровочные данные конкретной камеры и платформы.
    Молча применить маски GP010269 к другому полёту хуже, чем идти без маски,
    поэтому production-загрузчики должны использовать этот helper, а не вызывать
    ``AircraftMaskTracker.from_index`` напрямую.
    """
    anchors_dir = Path(anchors_dir)
    idx_path = anchors_dir / "index.json"
    if not idx_path.exists():
        return None

    payload = json.loads(idx_path.read_text(encoding="utf-8"))
    source_video = payload.get("video")
    if not source_video:
        if not allow_video_mismatch:
            if warn:
                warn(
                    f"[aircraft-mask] в {idx_path} нет поля 'video'; "
                    "запускаемся БЕЗ anchor-маски самолёта"
                )
            return None
    else:
        repo_root = Path(__file__).resolve().parents[1]
        if not _same_source_path(source_video, video_path, repo_root=repo_root):
            if not allow_video_mismatch:
                if warn:
                    warn(
                        "[aircraft-mask] anchor-маски от другого видео: "
                        f"index={source_video} current={video_path}; "
                        "запускаемся БЕЗ anchor-маски самолёта"
                    )
                return None
            if warn:
                warn(
                    "[aircraft-mask] ПРЕДУПРЕЖДЕНИЕ: принудительно используем anchor-маски "
                    f"от другого видео: index={source_video} current={video_path}"
                )

    return AircraftMaskTracker.from_index(anchors_dir)


def _mask_coverage(mask: np.ndarray) -> float:
    if mask is None or mask.size == 0:
        return 0.0
    return float((mask > 0).sum() / mask.size)


def apply_mask_to_image(img: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Обнулить пиксели под ``mask`` (uint8 0/255).

    Если маска непустая, возвращается копия изображения. Иначе входное
    изображение возвращается без изменений.

    Используется как предварительный фильтр перед LightGlue/LoFTR/ORB:
    чёрная область без текстуры не даёт детектору признаков, поэтому
    ключевые точки не попадают на корпус самолёта.
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
    """Вернуть булев массив формы ``(len(pts),)`` для отбора точек вне маски.

    Если маска не задана или пустая, сохраняются все точки.
    """
    n = 0 if pts is None else len(pts)
    if n == 0 or mask is None or not mask.any():
        return np.ones(n, dtype=bool)
    h, w = mask.shape[:2]
    xs = np.clip(pts[:, 0].astype(np.int32), 0, w - 1)
    ys = np.clip(pts[:, 1].astype(np.int32), 0, h - 1)
    return mask[ys, xs] == 0
