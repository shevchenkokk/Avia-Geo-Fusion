"""Детектор облаков / заграждения / смаза.

Кадр пригоден для матчинга только при наличии геометрической текстуры.
Попадание в облако, иней на линзе или засветка дают почти однородное
серо-белое поле; матчер «изобретает» совпадения в микровариациях контраста
и портит состояние EKF мусором.

Четыре метрики на уменьшенном кадре (~1 мс на кадр):
  1. Глобальное СКО (sigma_I).      Облако: 5–15.    Чисто: >30.
  2. Энтропия гистограммы (бит).    Облако: <5.      Чисто: 6–7.5.
  3. Плотность рёбер (|grad|>thr).  Облако: 1–2%.    Чисто: 10–20%.
  4. Доля блоков с малой дисперсией. Облако: ~0.85.  Чисто: <0.5.

Кадр флагируется при ≥3 из 4 голосов. Гистерезис (entry/exit streak)
подавляет одиночные выбросы; стейт-машина пайплайна видит чистые фронты.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class ObstructionMetrics:
    std: float
    entropy: float
    edge_density: float
    low_var_patch_frac: float
    vote_std: bool
    vote_entropy: bool
    vote_edge: bool
    vote_patch: bool
    votes: int

    def as_dict(self) -> dict:
        return {
            "obstr_std": self.std,
            "obstr_entropy": self.entropy,
            "obstr_edge_density": self.edge_density,
            "obstr_low_var_patch_frac": self.low_var_patch_frac,
            "obstr_votes": self.votes,
        }


@dataclass
class ObstructionResult:
    metrics: ObstructionMetrics
    raw_obstructed: bool          # голосование по текущему кадру
    is_obstructed: bool           # флаг после гистерезиса (на него реагирует пайплайн)
    streak_obstructed: int
    streak_clear: int


class ObstructionDetector:
    """Кадровый детектор облаков / смаза / блика с гистерезисом.

    Пороги подобраны для GP010269.MP4 (750 м AGL, сельхоз-ландшафт).
    Для другой оптики / высот / сцен требуется перекалибровка через
    scripts/verify_obstruction_detector.py.
    """

    def __init__(
        self,
        downsample_size: tuple[int, int] = (256, 144),
        std_threshold: float = 16.0,
        entropy_threshold: float = 5.8,
        edge_density_threshold: float = 0.05,
        edge_grad_threshold: float = 25.0,
        patch_grid: tuple[int, int] = (8, 8),
        patch_std_threshold: float = 12.0,
        low_var_patch_frac_threshold: float = 0.80,
        votes_required: int = 3,
        entry_streak: int = 2,
        exit_streak: int = 2,
    ) -> None:
        self.downsample_size = downsample_size
        self.std_threshold = std_threshold
        self.entropy_threshold = entropy_threshold
        self.edge_density_threshold = edge_density_threshold
        self.edge_grad_threshold = edge_grad_threshold
        self.patch_grid = patch_grid
        self.patch_std_threshold = patch_std_threshold
        self.low_var_patch_frac_threshold = low_var_patch_frac_threshold
        self.votes_required = votes_required
        self.entry_streak = entry_streak
        self.exit_streak = exit_streak

        self._streak_obstr = 0
        self._streak_clear = 0
        self._is_obstructed = False

    def reset(self) -> None:
        self._streak_obstr = 0
        self._streak_clear = 0
        self._is_obstructed = False

    @property
    def is_obstructed(self) -> bool:
        return self._is_obstructed

    def detect(
        self,
        frame: np.ndarray,
        aircraft_mask: Optional[np.ndarray] = None,
    ) -> ObstructionResult:
        """Вычисляет метрики и обновляет гистерезис для одного кадра.

        ``aircraft_mask`` — та же uint8 0/255 маска фюзеляжа что и в матчере.
        Замаскированные пиксели исключаются из расчёта, чтобы однородный фюзеляж
        не смещал метрики облака.
        """
        metrics = self._compute_metrics(frame, aircraft_mask)
        raw = metrics.votes >= self.votes_required

        if raw:
            self._streak_obstr += 1
            self._streak_clear = 0
            if not self._is_obstructed and self._streak_obstr >= self.entry_streak:
                self._is_obstructed = True
        else:
            self._streak_clear += 1
            self._streak_obstr = 0
            if self._is_obstructed and self._streak_clear >= self.exit_streak:
                self._is_obstructed = False

        return ObstructionResult(
            metrics=metrics,
            raw_obstructed=raw,
            is_obstructed=self._is_obstructed,
            streak_obstructed=self._streak_obstr,
            streak_clear=self._streak_clear,
        )

    def _compute_metrics(
        self,
        frame: np.ndarray,
        aircraft_mask: Optional[np.ndarray],
    ) -> ObstructionMetrics:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        small = cv2.resize(gray, self.downsample_size, interpolation=cv2.INTER_AREA)

        # Маска валидных пикселей: исключаем фюзеляж, чтобы его однородность
        # не тянула глобальное СКО и patch-дисперсию в сторону «заграждение».
        if aircraft_mask is not None:
            small_mask = cv2.resize(
                aircraft_mask, self.downsample_size, interpolation=cv2.INTER_NEAREST
            )
            valid = small_mask == 0
        else:
            valid = np.ones_like(small, dtype=bool)

        valid_pix = small[valid]
        if valid_pix.size < 64:
            # Почти весь кадр — фюзеляж: нельзя решить. Матчер позже отсеет по too_few_matches.
            return ObstructionMetrics(
                std=float("nan"), entropy=float("nan"),
                edge_density=float("nan"), low_var_patch_frac=float("nan"),
                vote_std=False, vote_entropy=False, vote_edge=False, vote_patch=False,
                votes=0,
            )

        std = float(np.std(valid_pix))

        hist = np.bincount(valid_pix.astype(np.int64), minlength=256).astype(np.float64)
        p = hist / hist.sum()
        nz = p > 0
        entropy = float(-(p[nz] * np.log2(p[nz])).sum())

        gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(gx, gy)
        edge_pixels = (grad_mag > self.edge_grad_threshold) & valid
        edge_density = float(edge_pixels.sum()) / float(valid.sum())

        gx_n, gy_n = self.patch_grid
        h, w = small.shape
        patch_h = h // gy_n
        patch_w = w // gx_n
        low_count = 0
        total_cells = 0
        for iy in range(gy_n):
            for ix in range(gx_n):
                y0, y1 = iy * patch_h, (iy + 1) * patch_h
                x0, x1 = ix * patch_w, (ix + 1) * patch_w
                cell = small[y0:y1, x0:x1]
                cell_valid = valid[y0:y1, x0:x1]
                if cell_valid.sum() < 16:
                    continue
                cell_std = float(np.std(cell[cell_valid]))
                total_cells += 1
                if cell_std < self.patch_std_threshold:
                    low_count += 1
        low_var_patch_frac = low_count / total_cells if total_cells > 0 else float("nan")

        vote_std = std < self.std_threshold
        vote_entropy = entropy < self.entropy_threshold
        vote_edge = edge_density < self.edge_density_threshold
        vote_patch = (
            np.isfinite(low_var_patch_frac)
            and low_var_patch_frac > self.low_var_patch_frac_threshold
        )
        votes = int(vote_std) + int(vote_entropy) + int(vote_edge) + int(vote_patch)

        return ObstructionMetrics(
            std=std,
            entropy=entropy,
            edge_density=edge_density,
            low_var_patch_frac=float(low_var_patch_frac),
            vote_std=bool(vote_std),
            vote_entropy=bool(vote_entropy),
            vote_edge=bool(vote_edge),
            vote_patch=bool(vote_patch),
            votes=votes,
        )
