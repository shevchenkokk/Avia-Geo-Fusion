"""Онлайн-маска окклюзий от самолёта для realtime-запуска.

Офлайн anchor-маски из ``data/masks/anchors`` — это профиль конкретной камеры
и носителя, а не универсальная сегментация. Этот модуль добавляет рабочий
онлайн-вариант: периодически переинициализирует маску через SAM и между этими
запусками переносит её оптическим потоком Лукаса-Канаде. Если SAM недоступен
или ничего не нашёл, вызывающий код получает ``None`` либо совместимый
резервный профиль платформы, но не устаревшую маску от другого видео.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from src.aircraft_mask import AircraftMaskTracker, MaskDiagnostics


@dataclass(frozen=True)
class SamOcclusionResult:
    mask: Optional[np.ndarray]
    coverage: float
    confidence: float
    detections: list[dict]
    reject_reason: str = ""


class SamAircraftSegmenter:
    """SAM3-сегментатор видимых частей самолёта по текстовым подсказкам."""

    def __init__(
        self,
        *,
        model_id: str = "facebook/sam3",
        prompts: Sequence[str] = (
            "aircraft",
            "airplane wing",
            "aircraft fuselage",
            "landing gear",
            "wing strut",
            "cockpit frame",
        ),
        device: str = "auto",
        score_threshold: float = 0.20,
        mask_threshold: float = 0.50,
        dilation_px: int = 18,
        close_radius_px: int = 5,
        min_coverage: float = 0.002,
        max_coverage: float = 0.45,
        left_only_frac: float = 1.0,
        max_box_right_frac: float = 1.0,
    ) -> None:
        self.model_id = model_id
        self.prompts = [p.strip() for p in prompts if p.strip()]
        self.device_request = device
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.dilation_px = dilation_px
        self.close_radius_px = close_radius_px
        self.min_coverage = min_coverage
        self.max_coverage = max_coverage
        self.left_only_frac = left_only_frac
        self.max_box_right_frac = max_box_right_frac

        self._torch = None
        self._processor = None
        self._model = None
        self._device = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        # Импортируем здесь, чтобы ошибка зависимостей появилась только при
        # реальном включении online-SAM, а обычный pipeline не тащил лишнее.
        from PIL import Image  # noqa: F401
        from transformers import Sam3Model, Sam3Processor

        if self.device_request != "auto":
            device = torch.device(self.device_request)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        elif (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        dtype = torch.float16 if device.type == "cuda" else torch.float32
        processor = Sam3Processor.from_pretrained(self.model_id)
        try:
            model = Sam3Model.from_pretrained(self.model_id, dtype=dtype)
        except TypeError:
            model = Sam3Model.from_pretrained(self.model_id, torch_dtype=dtype)
        model = model.to(device)
        model.eval()

        self._torch = torch
        self._processor = processor
        self._model = model
        self._device = device

    def segment(self, frame_bgr: np.ndarray) -> SamOcclusionResult:
        self._ensure_model()
        assert self._torch is not None
        assert self._processor is not None
        assert self._model is not None
        assert self._device is not None

        from PIL import Image

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        merged = np.zeros((h, w), dtype=np.uint8)
        all_detections: list[dict] = []
        best_score = 0.0

        for prompt in self.prompts:
            inputs = self._processor(images=pil, text=prompt, return_tensors="pt")
            inputs = inputs.to(self._device)
            with self._torch.no_grad():
                outputs = self._model(**inputs)
            results = self._processor.post_process_instance_segmentation(
                outputs,
                threshold=self.score_threshold,
                mask_threshold=self.mask_threshold,
                target_sizes=[(h, w)],
            )[0]
            prompt_mask, detections = self._union_prompt_masks(
                prompt,
                results.get("masks", []),
                results.get("scores", []),
                results.get("boxes", []),
                image_shape=(h, w),
            )
            merged = np.maximum(merged, prompt_mask)
            all_detections.extend(detections)
            for det in detections:
                if det.get("accepted"):
                    best_score = max(best_score, float(det.get("score", 0.0)))

        merged = self._close(merged, self.close_radius_px)
        merged = self._dilate(merged, self.dilation_px)
        coverage = _mask_coverage(merged)
        accepted = [d for d in all_detections if d.get("accepted")]
        if not accepted:
            return SamOcclusionResult(
                mask=None,
                coverage=coverage,
                confidence=0.0,
                detections=all_detections,
                reject_reason="no_sam_detection",
            )
        if coverage < self.min_coverage:
            return SamOcclusionResult(
                mask=None,
                coverage=coverage,
                confidence=best_score,
                detections=all_detections,
                reject_reason="mask_too_small",
            )
        if coverage > self.max_coverage:
            return SamOcclusionResult(
                mask=None,
                coverage=coverage,
                confidence=best_score,
                detections=all_detections,
                reject_reason="mask_too_large",
            )
        return SamOcclusionResult(
            mask=merged,
            coverage=coverage,
            confidence=best_score,
            detections=all_detections,
        )

    def _union_prompt_masks(
        self,
        prompt: str,
        masks,
        scores,
        boxes,
        *,
        image_shape: tuple[int, int],
    ) -> tuple[np.ndarray, list[dict]]:
        h, w = image_shape
        union = np.zeros((h, w), dtype=np.uint8)
        detections: list[dict] = []

        masks_np = _to_numpy(masks)
        scores_np = _to_numpy(scores)
        boxes_np = _to_numpy(boxes)
        if masks_np.ndim == 4:
            masks_np = masks_np[0]
        if scores_np.ndim == 2:
            scores_np = scores_np[0]
        if boxes_np.ndim == 3:
            boxes_np = boxes_np[0]
        if masks_np.ndim == 2:
            masks_np = masks_np.reshape(1, *masks_np.shape)
        scores_np = np.atleast_1d(scores_np)
        if boxes_np.ndim == 1 and boxes_np.size == 4:
            boxes_np = boxes_np.reshape(1, 4)
        if masks_np.size == 0 or scores_np.size == 0 or boxes_np.size == 0:
            return union, detections

        for i, (mask_i, score_i, box_i) in enumerate(zip(masks_np, scores_np, boxes_np)):
            score = float(score_i)
            box = np.asarray(box_i, dtype=np.float32).reshape(-1)
            if len(box) < 4:
                continue
            x1, y1, x2, y2 = (float(v) for v in box[:4])
            cx_frac = 0.5 * (x1 + x2) / max(1.0, float(w))
            x2_frac = x2 / max(1.0, float(w))
            det = {
                "prompt": prompt,
                "index": int(i),
                "score": score,
                "box": [x1, y1, x2, y2],
                "cx_frac": cx_frac,
                "x2_frac": x2_frac,
            }
            if score < self.score_threshold:
                det.update(accepted=False, reason="low_score")
                detections.append(det)
                continue
            if cx_frac > self.left_only_frac:
                det.update(accepted=False, reason="right_of_roi_prior")
                detections.append(det)
                continue
            if x2_frac > self.max_box_right_frac:
                det.update(accepted=False, reason="box_extends_past_roi")
                detections.append(det)
                continue

            mask_bin = np.asarray(mask_i).astype(np.uint8)
            if mask_bin.shape != (h, w):
                mask_bin = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)
            union[mask_bin.astype(bool)] = 255
            det.update(accepted=True)
            detections.append(det)

        return union, detections

    @staticmethod
    def _dilate(mask: np.ndarray, radius_px: int) -> np.ndarray:
        if radius_px <= 0:
            return mask
        ksize = 2 * radius_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        return cv2.dilate(mask, kernel)

    @staticmethod
    def _close(mask: np.ndarray, radius_px: int) -> np.ndarray:
        if radius_px <= 0:
            return mask
        ksize = 2 * radius_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


@dataclass
class _OnlineMaskState:
    frame_idx: int
    gray: np.ndarray
    mask: np.ndarray
    points: np.ndarray
    confidence: float


class OnlineAircraftOcclusionMasker:
    """Периодическое обновление через SAM и перенос маски optical flow."""

    def __init__(
        self,
        *,
        sam_segmenter: Optional[SamAircraftSegmenter] = None,
        platform_tracker: Optional[AircraftMaskTracker] = None,
        sam_refresh_frames: int = 60,
        min_seed_points: int = 16,
        min_track_points: int = 8,
        min_track_ratio: float = 0.35,
        max_fb_error_px: float = 1.5,
        max_median_flow_px: float = 80.0,
        coverage_min: float = 0.001,
        coverage_max: float = 0.45,
        refresh_on_confidence_below: float = 0.30,
    ) -> None:
        self.sam_segmenter = sam_segmenter
        self.platform_tracker = platform_tracker
        self.sam_refresh_frames = max(1, int(sam_refresh_frames))
        self.min_seed_points = min_seed_points
        self.min_track_points = min_track_points
        self.min_track_ratio = min_track_ratio
        self.max_fb_error_px = max_fb_error_px
        self.max_median_flow_px = max_median_flow_px
        self.coverage_min = coverage_min
        self.coverage_max = coverage_max
        self.refresh_on_confidence_below = refresh_on_confidence_below
        self.lk_params = dict(
            winSize=(31, 31),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        self._state: Optional[_OnlineMaskState] = None
        self._last_sam_frame: Optional[int] = None
        self._sam_disabled_reason = ""
        self.last_diagnostics = MaskDiagnostics(
            frame_idx=-1,
            anchor_index=-1,
            anchor_frame_index=-1,
            frame_delta=0,
            method="uninitialized",
            confidence=0.0,
        )

    def num_anchors(self) -> int:
        return self.platform_tracker.num_anchors() if self.platform_tracker else 0

    def mask_for_frame(
        self,
        frame_idx: int,
        frame_shape: tuple[int, int],
        frame: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        if frame is None:
            return self._platform_mask(frame_idx, frame_shape, frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if self._should_run_sam(frame_idx):
            mask = self._refresh_with_sam(frame_idx, frame, gray)
            if mask is not None:
                return mask

        if self._state is not None:
            propagated = self._propagate(frame_idx, gray)
            if propagated is not None:
                self._state = propagated
                self.last_diagnostics = MaskDiagnostics(
                    frame_idx=frame_idx,
                    anchor_index=-1,
                    anchor_frame_index=self._last_sam_frame or frame_idx,
                    frame_delta=frame_idx - (self._last_sam_frame or frame_idx),
                    method="online_of",
                    confidence=propagated.confidence,
                    num_seed_points=len(propagated.points),
                    num_tracked_points=len(propagated.points),
                    coverage=_mask_coverage(propagated.mask),
                )
                return propagated.mask

        return self._platform_mask(frame_idx, frame_shape, frame)

    def _should_run_sam(self, frame_idx: int) -> bool:
        if self.sam_segmenter is None or self._sam_disabled_reason:
            return False
        if self._state is None or self._last_sam_frame is None:
            return True
        if abs(frame_idx - self._last_sam_frame) >= self.sam_refresh_frames:
            return True
        return self.last_diagnostics.confidence < self.refresh_on_confidence_below

    def _refresh_with_sam(
        self,
        frame_idx: int,
        frame: np.ndarray,
        gray: np.ndarray,
    ) -> Optional[np.ndarray]:
        assert self.sam_segmenter is not None
        try:
            result = self.sam_segmenter.segment(frame)
        except Exception as exc:
            self._sam_disabled_reason = f"{type(exc).__name__}: {exc}"
            self.last_diagnostics = MaskDiagnostics(
                frame_idx=frame_idx,
                anchor_index=-1,
                anchor_frame_index=frame_idx,
                frame_delta=0,
                method="sam_disabled",
                confidence=0.0,
                reject_reason=self._sam_disabled_reason,
            )
            return None

        if result.mask is None:
            self.last_diagnostics = MaskDiagnostics(
                frame_idx=frame_idx,
                anchor_index=-1,
                anchor_frame_index=frame_idx,
                frame_delta=0,
                method="sam_no_mask",
                confidence=result.confidence,
                coverage=result.coverage,
                reject_reason=result.reject_reason,
            )
            return None

        points = self._seed_points(gray, result.mask)
        self._state = _OnlineMaskState(
            frame_idx=frame_idx,
            gray=gray,
            mask=result.mask,
            points=points,
            confidence=result.confidence,
        )
        self._last_sam_frame = frame_idx
        self.last_diagnostics = MaskDiagnostics(
            frame_idx=frame_idx,
            anchor_index=-1,
            anchor_frame_index=frame_idx,
            frame_delta=0,
            method="online_sam",
            confidence=result.confidence,
            num_seed_points=len(points),
            num_tracked_points=len(points),
            coverage=result.coverage,
        )
        return result.mask

    def _platform_mask(
        self,
        frame_idx: int,
        frame_shape: tuple[int, int],
        frame: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if self.platform_tracker is None:
            self.last_diagnostics = MaskDiagnostics(
                frame_idx=frame_idx,
                anchor_index=-1,
                anchor_frame_index=-1,
                frame_delta=0,
                method="none",
                confidence=1.0,
            )
            return None
        mask = self.platform_tracker.mask_for_frame(frame_idx, frame_shape, frame=frame)
        self.last_diagnostics = self.platform_tracker.last_diagnostics
        return mask

    def _seed_points(self, gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=240,
            qualityLevel=0.01,
            minDistance=8,
            mask=(mask > 0).astype(np.uint8) * 255,
        )
        if points is not None and len(points) >= self.min_seed_points:
            return points.astype(np.float32)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        edge_pts = []
        for contour in contours:
            if len(contour) == 0:
                continue
            step = max(1, len(contour) // 120)
            edge_pts.append(contour[::step, 0, :])
        if not edge_pts:
            return np.empty((0, 1, 2), dtype=np.float32)
        pts = np.vstack(edge_pts).astype(np.float32)
        _, unique_idx = np.unique(np.round(pts).astype(np.int32), axis=0, return_index=True)
        pts = pts[np.sort(unique_idx)]
        return pts.reshape(-1, 1, 2).astype(np.float32)

    def _propagate(
        self,
        frame_idx: int,
        gray: np.ndarray,
    ) -> Optional[_OnlineMaskState]:
        assert self._state is not None
        state = self._state
        seed_count = len(state.points)
        if seed_count < self.min_seed_points:
            return None
        curr_pts, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
            state.gray,
            gray,
            state.points,
            None,
            **self.lk_params,
        )
        if curr_pts is None or st_fwd is None:
            return None
        back_pts, st_bwd, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            state.gray,
            curr_pts,
            None,
            **self.lk_params,
        )
        if back_pts is None or st_bwd is None:
            return None
        fb_err = np.linalg.norm(
            back_pts.reshape(-1, 2) - state.points.reshape(-1, 2),
            axis=1,
        )
        ok = (
            (st_fwd.ravel() == 1)
            & (st_bwd.ravel() == 1)
            & (fb_err < self.max_fb_error_px)
        )
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

        affine, inliers = cv2.estimateAffinePartial2D(
            prev_kept.astype(np.float32),
            curr_kept.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=1000,
            confidence=0.99,
        )
        if affine is None or inliers is None or int(inliers.sum()) < self.min_track_points:
            return None
        h, w = gray.shape[:2]
        mask = cv2.warpAffine(
            state.mask,
            affine,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        mask = (mask > 127).astype(np.uint8) * 255
        coverage = _mask_coverage(mask)
        if not (self.coverage_min <= coverage <= self.coverage_max):
            return None
        points = self._seed_points(gray, mask)
        confidence = float(np.clip(ratio * math.exp(-median_flow / 120.0), 0.0, 1.0))
        return _OnlineMaskState(
            frame_idx=frame_idx,
            gray=gray,
            mask=mask,
            points=points,
            confidence=confidence,
        )


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _mask_coverage(mask: Optional[np.ndarray]) -> float:
    if mask is None or mask.size == 0:
        return 0.0
    return float((mask > 0).sum() / mask.size)
