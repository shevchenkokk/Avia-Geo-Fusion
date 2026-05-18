"""Автомат состояний трека: TRACK / WEAK / RELOCALIZE / BOOTSTRAP.

Адаптивное расписание каналов абсолютной локализации (appearance / structural)
по горизонтальной σ_pos и времени с последней принятой фиксации по карте.

Мотивация: бортовой пайплайн на Jetson Orin Nano не может позволить
запускать XFeat + DINOv2 retriever + structural NCC каждый кадр —
суммарный бюджет ~250-450мс при целевых 33мс для 30 fps. В режиме TRACK
EKF уже хорошо знает позицию по VO, и appearance нужен только изредка,
чтобы сдерживать дрейф. В RELOCALIZE наоборот нужны частые фиксации от
обоих каналов, чтобы быстрее восстановить захват.

Режимы:
  - BOOTSTRAP: до первой инициализации EKF из cluster centroid.
  - TRACK    : σ_pos < sigma_track_m И gap < timeout_track_s.
  - WEAK     : σ_pos < sigma_weak_m  И gap < timeout_weak_s.
  - RELOCALIZE: всё остальное.

Расписание подобрано под Orin Nano с учётом профилей XFeat ~100мс,
structural NCC ~200мс. На более мощном железе (Orin AGX, x86 GPU) можно
сократить периоды через ChannelSchedule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TrackMode(str, Enum):
    BOOTSTRAP = "bootstrap"
    TRACK = "track"
    WEAK = "weak"
    RELOCALIZE = "relocalize"


@dataclass
class ChannelSchedule:
    """Период запуска каналов (секунды). 0.0 → каждый кадр."""
    appearance_period_s: float
    structural_period_s: float


_DEFAULT_SCHEDULES = {
    TrackMode.BOOTSTRAP:  ChannelSchedule(appearance_period_s=1.0, structural_period_s=2.0),
    TrackMode.TRACK:      ChannelSchedule(appearance_period_s=4.0, structural_period_s=8.0),
    TrackMode.WEAK:       ChannelSchedule(appearance_period_s=1.0, structural_period_s=2.0),
    TrackMode.RELOCALIZE: ChannelSchedule(appearance_period_s=0.5, structural_period_s=1.0),
}


@dataclass
class TrackState:
    """Адаптивное расписание каналов абсолютной локализации."""

    sigma_track_m: float = 60.0
    sigma_weak_m: float = 200.0
    timeout_track_s: float = 5.0
    timeout_weak_s: float = 30.0

    mode: TrackMode = TrackMode.BOOTSTRAP
    last_accept_t_s: Optional[float] = None
    last_appearance_run_t_s: Optional[float] = None
    last_structural_run_t_s: Optional[float] = None
    schedules: dict[TrackMode, ChannelSchedule] = field(
        default_factory=lambda: dict(_DEFAULT_SCHEDULES)
    )

    def record_accept(self, t_sec: float) -> None:
        """Зафиксировать принятый map-fix на момент t_sec."""
        self.last_accept_t_s = t_sec

    def recompute_mode(
        self,
        t_sec: float,
        sigma_pos_m: float,
        bootstrap_done: bool,
    ) -> TrackMode:
        """Пересчитать режим. Не меняет last_accept."""
        if not bootstrap_done:
            self.mode = TrackMode.BOOTSTRAP
            return self.mode
        gap_s = (t_sec - self.last_accept_t_s) if self.last_accept_t_s is not None else 1e9
        if sigma_pos_m < self.sigma_track_m and gap_s < self.timeout_track_s:
            self.mode = TrackMode.TRACK
        elif sigma_pos_m < self.sigma_weak_m and gap_s < self.timeout_weak_s:
            self.mode = TrackMode.WEAK
        else:
            self.mode = TrackMode.RELOCALIZE
        return self.mode

    def update(
        self,
        t_sec: float,
        sigma_pos_m: float,
        accepted_this_frame: bool,
        bootstrap_done: bool,
    ) -> TrackMode:
        """Удобная обёртка: record_accept + recompute_mode."""
        if accepted_this_frame:
            self.record_accept(t_sec)
        return self.recompute_mode(t_sec, sigma_pos_m, bootstrap_done)

    def should_run_appearance(self, t_sec: float, obstructed: bool) -> bool:
        if obstructed:
            return False
        period = self.schedules[self.mode].appearance_period_s
        if period <= 0.0:
            return True
        if self.last_appearance_run_t_s is None:
            return True
        return (t_sec - self.last_appearance_run_t_s) >= period

    def should_run_structural(self, t_sec: float, obstructed: bool) -> bool:
        if obstructed:
            return False
        period = self.schedules[self.mode].structural_period_s
        if period <= 0.0:
            return True
        if self.last_structural_run_t_s is None:
            return True
        return (t_sec - self.last_structural_run_t_s) >= period

    def mark_appearance_ran(self, t_sec: float) -> None:
        self.last_appearance_run_t_s = t_sec

    def mark_structural_ran(self, t_sec: float) -> None:
        self.last_structural_run_t_s = t_sec
