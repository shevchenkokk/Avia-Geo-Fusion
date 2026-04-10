"""Расширенный фильтр Калмана для оценки состояния самолёта (этап 2.4).

Вектор состояния (9 компонент) в локальной СК ENU:

    x = [x_e, y_n, z_u, vx_e, vy_n, vz_u, yaw, yaw_rate, scale_bias]

``scale_bias`` компенсирует мультипликативную ошибку VO: истинная скорость ≈
scale_bias * raw_vo_velocity. Наблюдаем только через рассогласование карта–VO
между двумя последовательными фиксациями (этап 2.5).

Соглашения:
  - ``yaw`` — курс от севера, по часовой стрелке (авиационный).
  - тело x = вперёд (по курсу), тело y = вправо.
  - ENU: x = восток, y = север, z = вверх.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.frame_bridge import FrameBridge


def wrap_pi(angle: float) -> float:
    return float((angle + math.pi) % (2 * math.pi) - math.pi)


@dataclass
class UpdateResult:
    accepted: bool
    mahalanobis2: float
    innovation: np.ndarray
    channel: str = ""


class StateFilter:
    """EKF на 9-мерном состоянии ENU.

    Начальная ковариация намеренно широкая (км для позиции, π для курса),
    чтобы первая фиксация карты доминировала над приором.
    """

    DIM = 9
    IDX_X_E = 0
    IDX_Y_N = 1
    IDX_Z_U = 2
    IDX_VX_E = 3
    IDX_VY_N = 4
    IDX_VZ_U = 5
    IDX_YAW = 6
    IDX_YAW_RATE = 7
    IDX_SCALE_BIAS = 8

    def __init__(
        self,
        bridge: FrameBridge,
        # СКО шумов процесса (в секунду).
        q_pos_mps: float = 0.0,            # интеграция; шум позиции приходит от скорости
        q_vel_mps2: float = 1.5,           # ветер и турбулентность (уровень ускорения)
        q_vz_mps2: float = 0.5,            # вертикальное ускорение (меньше; крейсер горизонтален)
        q_yaw_radps: float = math.radians(2.0),    # дрейф курса из-за шума угловой скорости VO
        q_yaw_rate_radps2: float = math.radians(5.0),  # джиттер угловой скорости
        q_scale_per_sec: float = 1e-3,             # очень медленный дрейф; смещение в основном статично
        # Начальная неопределённость (1-sigma).
        sigma0_pos_m: float = 5_000.0,
        sigma0_z_m: float = 100.0,
        sigma0_vel_mps: float = 50.0,
        sigma0_yaw_rad: float = math.pi,
        sigma0_yaw_rate_radps: float = math.radians(30.0),
        sigma0_scale: float = 0.3,                 # 30% неизвестно до первого сравнения карта–VO
        # Пороги ворот Махаланобиса. χ²(k=1) при 99.9% ≈ 10.83.
        # Используем чуть более мягкие значения, чтобы не отбрасывать
        # достоверные, но зашумлённые фиксации в переходных режимах.
        gate_chi2_pos: float = 25.0,
        gate_chi2_vel: float = 16.0,
        gate_chi2_alt: float = 16.0,
        gate_chi2_heading: float = 9.0,
        gate_chi2_scale: float = 25.0,    # мягче: соотношение карта–VO шумное
        heading_prior_yaw_rate_threshold_radps: float = math.radians(2.0),
    ) -> None:
        self.bridge = bridge
        self.q_pos = q_pos_mps
        self.q_vel = q_vel_mps2
        self.q_vz = q_vz_mps2
        self.q_yaw = q_yaw_radps
        self.q_yaw_rate = q_yaw_rate_radps2
        self.q_scale = q_scale_per_sec
        self.gate_chi2 = dict(
            pos=gate_chi2_pos,
            vel=gate_chi2_vel,
            alt=gate_chi2_alt,
            heading=gate_chi2_heading,
            scale=gate_chi2_scale,
        )
        self.heading_prior_threshold = heading_prior_yaw_rate_threshold_radps

        self.x = np.zeros(self.DIM)
        self.x[self.IDX_SCALE_BIAS] = 1.0
        self.P = np.diag([
            sigma0_pos_m ** 2, sigma0_pos_m ** 2, sigma0_z_m ** 2,
            sigma0_vel_mps ** 2, sigma0_vel_mps ** 2, sigma0_vel_mps ** 2,
            sigma0_yaw_rad ** 2, sigma0_yaw_rate_radps ** 2,
            sigma0_scale ** 2,
        ])
        self.initialized = False
        self.last_dt = 0.0
        self.last_update: Optional[UpdateResult] = None
        # Буфер этапа 2.5: сырое смещение VO и последняя принятая фиксация карты.
        # Сбрасывается при каждом принятом update_map_position; используется
        # для вычисления обновления scale_bias.
        self._vo_disp_enu_raw = np.zeros(2)
        self._last_map_pos_enu: Optional[np.ndarray] = None
        self._scale_updates_applied = 0

    def initialize_from_wgs84(
        self,
        lat: float, lon: float, alt_msl: float,
        yaw: float = 0.0,
        sigma_pos_m: float = 50.0,
        sigma_yaw_rad: float = math.pi / 2,
    ) -> None:
        x_e, y_n, z_u = self.bridge.wgs84_to_enu(lat, lon, alt_msl)
        self.x[self.IDX_X_E] = x_e
        self.x[self.IDX_Y_N] = y_n
        self.x[self.IDX_Z_U] = z_u
        self.x[self.IDX_VX_E] = 0.0
        self.x[self.IDX_VY_N] = 0.0
        self.x[self.IDX_VZ_U] = 0.0
        self.x[self.IDX_YAW] = wrap_pi(yaw)
        self.x[self.IDX_YAW_RATE] = 0.0
        self.P[self.IDX_X_E, self.IDX_X_E] = sigma_pos_m ** 2
        self.P[self.IDX_Y_N, self.IDX_Y_N] = sigma_pos_m ** 2
        self.P[self.IDX_YAW, self.IDX_YAW] = sigma_yaw_rad ** 2
        self.initialized = True

    def predict(self, dt: float) -> None:
        if dt <= 0.0:
            return
        self.last_dt = dt

        F = np.eye(self.DIM)
        F[self.IDX_X_E, self.IDX_VX_E] = dt
        F[self.IDX_Y_N, self.IDX_VY_N] = dt
        F[self.IDX_Z_U, self.IDX_VZ_U] = dt
        F[self.IDX_YAW, self.IDX_YAW_RATE] = dt

        self.x = F @ self.x
        self.x[self.IDX_YAW] = wrap_pi(self.x[self.IDX_YAW])

        # Дискретный шум процесса. Модель постоянного ускорения: позиция
        # накапливает (q_vel*dt²)²/3; используем диагональное приближение.
        q_pos_var = (self.q_pos * dt) ** 2 + (self.q_vel * dt * dt) ** 2 / 3.0
        q_vel_var = (self.q_vel * dt) ** 2
        q_z_var = (self.q_pos * dt) ** 2 + (self.q_vz * dt * dt) ** 2 / 3.0
        q_vz_var = (self.q_vz * dt) ** 2
        q_yaw_var = (self.q_yaw * dt) ** 2 + (self.q_yaw_rate * dt * dt) ** 2 / 3.0
        q_yaw_rate_var = (self.q_yaw_rate * dt) ** 2
        q_scale_var = (self.q_scale * dt) ** 2
        Q = np.diag([
            q_pos_var, q_pos_var, q_z_var,
            q_vel_var, q_vel_var, q_vz_var,
            q_yaw_var, q_yaw_rate_var,
            q_scale_var,
        ])
        self.P = F @ self.P @ F.T + Q
        # Симметризация — борьба с накопленной асимметрией конечной точности.
        self.P = 0.5 * (self.P + self.P.T)

    def _gated_update(
        self,
        H: np.ndarray,
        z: np.ndarray,
        R: np.ndarray,
        gate_chi2: float,
        channel: str,
        wrap_idx: tuple[int, ...] = (),
    ) -> UpdateResult:
        """Линейное обновление измерения с воротами Махаланобиса.

        ``wrap_idx`` — индексы компонент невязки, требующих wrap_pi (курс).
        """
        y = z - H @ self.x
        for i in wrap_idx:
            y[i] = wrap_pi(y[i])
        S = H @ self.P @ H.T + R
        try:
            S_inv_y = np.linalg.solve(S, y)
        except np.linalg.LinAlgError:
            return UpdateResult(False, float("inf"), y, channel)
        d2 = float(y @ S_inv_y)
        if d2 > gate_chi2:
            return UpdateResult(False, d2, y, channel)
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T
        self.x = self.x + K @ y
        if self.IDX_YAW in wrap_idx or self.IDX_YAW < self.DIM:
            self.x[self.IDX_YAW] = wrap_pi(self.x[self.IDX_YAW])
        # Форма Джозефа: P = (I-KH)P(I-KH)ᵀ + KRKᵀ — симметрична и PSD-устойчива.
        I = np.eye(self.DIM)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return UpdateResult(True, d2, y, channel)

    def update_map_position(
        self,
        x_e: float, y_n: float,
        sigma_xy_m: float = 30.0,
        min_disp_for_scale_m: float = 30.0,
    ) -> UpdateResult:
        """Обновление по фиксации карты (линейное, с воротами).

        При принятии и наличии предыдущей фиксации с достаточным накопленным
        VO-смещением дополнительно запускает ``_update_scale_from_disp_pair``
        (этап 2.5).
        """
        H = np.zeros((2, self.DIM))
        H[0, self.IDX_X_E] = 1.0
        H[1, self.IDX_Y_N] = 1.0
        z = np.array([x_e, y_n], dtype=np.float64)
        R = np.eye(2) * (sigma_xy_m ** 2)
        res = self._gated_update(H, z, R, self.gate_chi2["pos"], channel="map")
        if res.accepted:
            if self._last_map_pos_enu is not None:
                map_disp = np.array([x_e, y_n]) - self._last_map_pos_enu
                vo_disp = self._vo_disp_enu_raw.copy()
                if (np.linalg.norm(vo_disp) >= min_disp_for_scale_m
                        and np.linalg.norm(map_disp) >= min_disp_for_scale_m):
                    # СКО разности смещений: две фиксации карты дают σ_xy каждая
                    # (независимо), поэтому σ смещения = √2 * σ_xy.
                    self._update_scale_from_disp_pair(
                        map_disp_xy=map_disp,
                        vo_disp_xy_raw=vo_disp,
                        sigma_disp_m=math.sqrt(2.0) * sigma_xy_m,
                    )
            self._last_map_pos_enu = np.array([x_e, y_n], dtype=np.float64)
            self._vo_disp_enu_raw = np.zeros(2)
        self.last_update = res
        return res

    def _update_scale_from_disp_pair(
        self,
        map_disp_xy: np.ndarray,
        vo_disp_xy_raw: np.ndarray,
        sigma_disp_m: float,
    ) -> UpdateResult:
        """Обновление scale_bias по паре смещений карта–VO (этап 2.5).

        Модель измерения:
            z = map_disp_xy
            h(state) = scale_bias * vo_disp_xy_raw
            ∂h/∂scale_bias = vo_disp_xy_raw

        Шум VO-интеграции (5% от модуля смещения за 1σ) добавляется
        к ковариации измерения, чтобы длинные отрезки не переоценивались.
        """
        disp_mag = float(np.linalg.norm(vo_disp_xy_raw))
        sigma_vo_m = 0.05 * disp_mag
        sigma_total = math.hypot(sigma_disp_m, sigma_vo_m)

        H = np.zeros((2, self.DIM))
        H[:, self.IDX_SCALE_BIAS] = vo_disp_xy_raw
        z = map_disp_xy
        R = np.eye(2) * (sigma_total ** 2)
        res = self._gated_update(H, z, R, self.gate_chi2["scale"], channel="scale")
        if res.accepted:
            self._scale_updates_applied += 1
        return res

    def update_map_position_wgs84(
        self, lat: float, lon: float,
        sigma_xy_m: float = 30.0,
    ) -> UpdateResult:
        x_e, y_n, _ = self.bridge.wgs84_to_enu(lat, lon, self.bridge.alt0_msl)
        return self.update_map_position(x_e, y_n, sigma_xy_m=sigma_xy_m)

    def update_of_velocity(
        self,
        v_body_xy: np.ndarray,
        yaw_rate_radps: float,
        dt: float,
        sigma_v_mps: float = 2.0,
        sigma_yaw_rate_radps: float = math.radians(1.0),
    ) -> UpdateResult:
        """Линейное обновление по (vx_e, vy_n, yaw_rate).

        Скорость в СК тела поворачивается в ENU через текущую оценку курса.
        ``v_body_xy`` — СЫРОЙ выход VO; внутри умножается на scale_bias перед
        формированием невязки, чтобы неверный масштаб не портил состояние
        скорости.

        ``dt`` используется только для накопления сырого VO-смещения для
        пути scale_bias (этап 2.5) и не влияет на само измерение скорости.
        """
        yaw = float(self.x[self.IDX_YAW])
        s, c = math.sin(yaw), math.cos(yaw)
        vx_b_raw, vy_b_raw = float(v_body_xy[0]), float(v_body_xy[1])
        scale = float(self.x[self.IDX_SCALE_BIAS])
        vx_b = scale * vx_b_raw
        vy_b = scale * vy_b_raw
        vx_e = vx_b * s + vy_b * c
        vy_n = vx_b * c - vy_b * s
        # Когда scale_bias не сошёлся, VO-скорость точно известна лишь как
        # направление. Раздуваем R на (σ_scale * |v_raw|)² — по мере сходимости
        # scale_bias σ_scale уменьшается и R возвращается к шумовому полу.
        # Угловая скорость остаётся нетронутой: scale_bias на неё не влияет.
        sigma_scale = math.sqrt(self.P[self.IDX_SCALE_BIAS, self.IDX_SCALE_BIAS])
        v_raw_norm = math.hypot(vx_b_raw, vy_b_raw)
        sigma_v_eff_sq = sigma_v_mps ** 2 + (sigma_scale * v_raw_norm) ** 2
        H = np.zeros((3, self.DIM))
        H[0, self.IDX_VX_E] = 1.0
        H[1, self.IDX_VY_N] = 1.0
        H[2, self.IDX_YAW_RATE] = 1.0
        z = np.array([vx_e, vy_n, yaw_rate_radps], dtype=np.float64)
        R = np.diag([sigma_v_eff_sq, sigma_v_eff_sq, sigma_yaw_rate_radps ** 2])
        res = self._gated_update(H, z, R, self.gate_chi2["vel"], channel="of")
        # Накапливаем СЫРОЕ (немасштабированное) ENU-смещение для пути scale_bias —
        # даже если обновление скорости отклонено воротами Махаланобиса.
        # Смысл scale_bias — поглощать систематическое смещение VO, которое
        # по построению проявляется как устойчивая невязка; фильтрация накопителя
        # по res.accepted заглушила бы именно этот сигнал.
        if dt > 0.0:
            vx_e_raw = vx_b_raw * s + vy_b_raw * c
            vy_n_raw = vx_b_raw * c - vy_b_raw * s
            self._vo_disp_enu_raw += np.array([vx_e_raw, vy_n_raw]) * dt
        self.last_update = res
        return res

    def update_altitude(
        self, z_msl: float, sigma_h_m: float = 100.0,
    ) -> UpdateResult:
        """Мягкая привязка z_u. ``z_msl`` переводится в ENU через bridge."""
        z_u = z_msl - self.bridge.alt0_msl
        H = np.zeros((1, self.DIM))
        H[0, self.IDX_Z_U] = 1.0
        z = np.array([z_u], dtype=np.float64)
        R = np.array([[sigma_h_m ** 2]])
        res = self._gated_update(H, z, R, self.gate_chi2["alt"], channel="alt")
        self.last_update = res
        return res

    def maybe_update_heading_prior(
        self, sigma_heading_rad: float = math.radians(15.0),
    ) -> UpdateResult:
        """Приор курса по вектору скорости, применяется только при прямолинейном полёте.

        На прямом крейсере курс задан вектором скорости (yaw = atan2(vx_e, vy_n)).
        Моделируем как мягкое прямое измерение yaw через линеаризованное h(x);
        Якобиан берётся только по yaw, без обратного распространения через скорость.
        """
        if abs(self.x[self.IDX_YAW_RATE]) > self.heading_prior_threshold:
            res = UpdateResult(False, 0.0, np.zeros(1), channel="heading_skip")
            self.last_update = res
            return res
        vx_e = float(self.x[self.IDX_VX_E])
        vy_n = float(self.x[self.IDX_VY_N])
        speed = math.hypot(vx_e, vy_n)
        if speed < 5.0:
            # При малой скорости atan2 шумит — курс по скорости бессмысленен.
            res = UpdateResult(False, 0.0, np.zeros(1), channel="heading_lowspeed")
            self.last_update = res
            return res
        z_yaw = math.atan2(vx_e, vy_n)
        H = np.zeros((1, self.DIM))
        H[0, self.IDX_YAW] = 1.0
        z = np.array([z_yaw])
        R = np.array([[sigma_heading_rad ** 2]])
        res = self._gated_update(
            H, z, R, self.gate_chi2["heading"], channel="heading",
            wrap_idx=(0,),
        )
        self.last_update = res
        return res

    def position_wgs84(self) -> tuple[float, float, float]:
        x_e = float(self.x[self.IDX_X_E])
        y_n = float(self.x[self.IDX_Y_N])
        z_u = float(self.x[self.IDX_Z_U])
        return self.bridge.enu_to_wgs84(x_e, y_n, z_u)

    def position_enu(self) -> tuple[float, float, float]:
        return (
            float(self.x[self.IDX_X_E]),
            float(self.x[self.IDX_Y_N]),
            float(self.x[self.IDX_Z_U]),
        )

    def position_sigma_m(self) -> float:
        """1σ горизонтальная неопределённость позиции (м). Норма по следу."""
        return float(math.sqrt(
            self.P[self.IDX_X_E, self.IDX_X_E]
            + self.P[self.IDX_Y_N, self.IDX_Y_N]
        ))

    def heading_deg(self) -> float:
        return math.degrees(self.x[self.IDX_YAW])

    def speed(self) -> float:
        return float(math.hypot(self.x[self.IDX_VX_E], self.x[self.IDX_VY_N]))

    def scale_bias(self) -> float:
        return float(self.x[self.IDX_SCALE_BIAS])

    def scale_bias_sigma(self) -> float:
        return float(math.sqrt(self.P[self.IDX_SCALE_BIAS, self.IDX_SCALE_BIAS]))

    def n_scale_updates(self) -> int:
        return self._scale_updates_applied
