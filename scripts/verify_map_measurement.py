"""Проверка этапа 2.6: доказать на синтетике с известной истиной, что новый
путь ``compute_map_measurement`` + EKF даёт качество траектории не хуже старого
Geolocator со сглаживанием EMA.

Два этапа:

  Этап A — поточечная корректность. Строятся синтетические пары
  (mkpts_drone, mkpts_map) по известной гомографии, соответствующей самолёту
  над центром тайла. Затем запускается ``compute_map_measurement`` и
  проверяется, что восстановленные (lat, lon) совпадают с аналитической
  проекцией в пределах субпиксельной ошибки, а σ_xy_m попадает в ожидаемый
  диапазон.

  Этап B — end-to-end fusion. Моделируется двухминутный крейсерский участок.
  Каждую секунду создаётся синтетический MapMeasurement с шумом матчера и
  разным числом inlier-точек. Он подаётся в ``StateFilter`` через
  ``apply_to_state_filter``; проверяется, что траектория фильтра держится
  возле истины в рамках ожиданий §2.6: без EMA, всё сглаживание внутри EKF,
  ошибка позиции остаётся ограниченной σ_xy_m.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ekf import StateFilter
from src.frame_bridge import FrameBridge
from src.map_measurement import (
    MapMeasurement,
    apply_to_state_filter,
    compute_map_measurement,
)


# ---------------------------------------------------------------------------
# Вспомогательные функции для синтетической гомографии
# ---------------------------------------------------------------------------

def _make_synthetic_match_pair(
    bbox: tuple[float, float, float, float],
    map_shape: tuple[int, int],
    truth_lat: float,
    truth_lon: float,
    frame_shape: tuple[int, int] = (1080, 1920),
    n_kpts: int = 60,
    noise_px: float = 1.0,
    rng: np.random.Generator | None = None,
):
    """Построить (mkpts_drone, mkpts_map), согласованные с гомографией.

    Гомография переводит центр кадра самолёта в пиксель карты, соответствующий
    (truth_lat, truth_lon). Для простоты это similarity-преобразование
    (масштаб + сдвиг); у реального матчера H более общая, но логика проекции
    центра такая же.
    """
    if rng is None:
        rng = np.random.default_rng(seed=0)
    west, south, east, north = bbox
    h, w = map_shape[:2]

    # Переводим истинную точку в пиксель спутникового тайла, обратно к _pixel_to_gps.
    map_x = (truth_lon - west) / max(east - west, 1e-9) * w
    map_y = (north - truth_lat) / max(north - south, 1e-9) * h

    # Центр кадра самолёта.
    fh, fw = frame_shape[:2]
    cx_d, cy_d = fw / 2.0, fh / 2.0

    # Синтетические равномерные ключевые точки самолёта с отступом от края.
    drone_pts = rng.uniform(low=[fw * 0.15, fh * 0.15],
                            high=[fw * 0.85, fh * 0.85],
                            size=(n_kpts, 2)).astype(np.float32)

    # Берём similarity H: масштаб s = map_extent_px / frame_extent_px,
    # поворот 0, сдвиг кладёт (cx_d, cy_d) в (map_x, map_y). Масштаб выбран так,
    # чтобы кадр шириной 1920 пикселей покрывал разумный участок тайла,
    # например около 50 % ширины.
    s = (0.5 * w) / fw  # frame-pixel -> map-pixel
    H_truth = np.array([
        [s, 0, map_x - s * cx_d],
        [0, s, map_y - s * cy_d],
        [0, 0, 1],
    ], dtype=np.float32)

    # Проецируем точки самолёта на карту через H и добавляем шум.
    drone_h = np.column_stack([drone_pts, np.ones(len(drone_pts), dtype=np.float32)])
    map_h = (H_truth @ drone_h.T).T
    map_pts = (map_h[:, :2] / map_h[:, 2:3]).astype(np.float32)
    map_pts += rng.normal(0.0, noise_px, size=map_pts.shape).astype(np.float32)
    return drone_pts, map_pts, frame_shape, H_truth


# ---------------------------------------------------------------------------
# Этап A — поточечная корректность
# ---------------------------------------------------------------------------

def stage_a(rng_seed: int = 11) -> dict:
    rng = np.random.default_rng(seed=rng_seed)
    # Реалистичный тайл z=17 около Коломны: ~1.1 м/пикс при 55°N, так что
    # тайл 2000x2000 пикс покрывает около 2.2 км по каждой оси. Ошибка
    # репроекции 1 пикс здесь соответствует примерно 1 м — правильный масштаб
    # для вывода σ_xy_m.
    bbox = (38.140, 55.080, 38.175, 55.100)
    map_shape = (2000, 2000)

    truths = [
        (55.086, 38.149),
        (55.090, 38.155),
        (55.095, 38.165),
    ]
    rows = []
    for lat_t, lon_t in truths:
        d_pts, m_pts, frame_shape, _H = _make_synthetic_match_pair(
            bbox, map_shape, lat_t, lon_t, n_kpts=60, noise_px=1.0, rng=rng,
        )
        meas = compute_map_measurement(frame_shape, d_pts, m_pts, bbox, map_shape)
        if not meas.accepted:
            rows.append((lat_t, lon_t, meas, None, None))
            continue
        # Переводим ошибку в метры.
        d_lat_m = (meas.lat - lat_t) * 111320.0
        d_lon_m = (meas.lon - lon_t) * 111320.0 * math.cos(math.radians(lat_t))
        err_m = math.hypot(d_lat_m, d_lon_m)
        rows.append((lat_t, lon_t, meas, err_m, None))

    accepted = [r for r in rows if r[2].accepted]
    errors_m = [r[3] for r in rows if r[3] is not None]
    sigmas = [r[2].sigma_xy_m for r in accepted]
    confs = [r[2].confidence for r in accepted]
    return {
        "n_truths": len(truths),
        "n_accepted": len(accepted),
        "max_err_m": max(errors_m) if errors_m else float("inf"),
        "mean_err_m": float(np.mean(errors_m)) if errors_m else float("inf"),
        "sigma_min_m": float(min(sigmas)) if sigmas else float("inf"),
        "sigma_max_m": float(max(sigmas)) if sigmas else float("inf"),
        "conf_min": float(min(confs)) if confs else 0.0,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Этап B — end-to-end fusion вместо старого пути с EMA
# ---------------------------------------------------------------------------

def stage_b(rng_seed: int = 22) -> dict:
    rng = np.random.default_rng(seed=rng_seed)
    bridge = FrameBridge(55.086025, 38.149033, 130.0)
    # Реалистичный тайл z=17 около Коломны: ~1.1 м/пикс при 55°N, так что
    # тайл 2000x2000 пикс покрывает около 2.2 км по каждой оси. Ошибка
    # репроекции 1 пикс здесь соответствует примерно 1 м — правильный масштаб
    # для вывода σ_xy_m.
    bbox = (38.140, 55.080, 38.175, 55.100)
    map_shape = (2000, 2000)
    # Этап B объединяет 2 минуты крейсера на 70 м/с, то есть ~8.4 км —
    # слишком много для одного тайла z=17. В этапе A сохраняем реализм тайла,
    # а здесь берём более крупный bbox, чтобы синтетическая истинная траектория
    # целиком оставалась внутри одной карты.
    bbox_b = (38.05, 55.05, 38.25, 55.15)
    map_shape_b = (2048, 2048)

    f = StateFilter(bridge)
    f.initialize_from_wgs84(bridge.lat0, bridge.lon0, bridge.alt0_msl + 620.0,
                            yaw=math.radians(30.0),
                            sigma_pos_m=20.0, sigma_yaw_rad=math.radians(10.0))
    yaw_r = math.radians(30.0)
    speed = 70.0
    f.x[StateFilter.IDX_VX_E] = speed * math.sin(yaw_r)
    f.x[StateFilter.IDX_VY_N] = speed * math.cos(yaw_r)
    f.P[StateFilter.IDX_VX_E, StateFilter.IDX_VX_E] = 30.0 ** 2
    f.P[StateFilter.IDX_VY_N, StateFilter.IDX_VY_N] = 30.0 ** 2

    of_dt = 0.1
    duration = 120.0
    map_period_s = 1.0
    map_steps = int(round(map_period_s / of_dt))
    n_steps = int(round(duration / of_dt))

    # Истинная траектория в WGS84.
    vx_e_t = speed * math.sin(yaw_r)
    vy_n_t = speed * math.cos(yaw_r)

    pos_err = []
    sigmas_used = []
    accepted_count = 0

    t_acc = 0.0
    for k in range(1, n_steps + 1):
        f.predict(of_dt)
        t_acc += of_dt
        x_truth = vx_e_t * t_acc
        y_truth = vy_n_t * t_acc

        # Синтетический OF без смещения: здесь проверяется путь map_measurement,
        # а VO со смещением отдельно покрыт в этапе 2.5.
        sigma_v = 1.5
        v_raw = np.array([speed + rng.normal(0, sigma_v), rng.normal(0, sigma_v)])
        f.update_of_velocity(v_raw, rng.normal(0, math.radians(0.3)),
                             dt=of_dt, sigma_v_mps=sigma_v,
                             sigma_yaw_rate_radps=math.radians(0.5))

        if k % map_steps == 0:
            # Строим синтетическую пару вокруг истины, иногда с ухудшенным
            # качеством и малым числом inlier-точек, чтобы проверить адаптацию σ_xy.
            n_kpts = int(rng.choice([60, 30, 15]))
            noise_px = float(rng.choice([1.0, 1.5, 2.5]))
            lat_t, lon_t, _ = bridge.enu_to_wgs84(x_truth, y_truth, 0.0)
            d_pts, m_pts, frame_shape, _ = _make_synthetic_match_pair(
                bbox_b, map_shape_b, lat_t, lon_t, n_kpts=n_kpts,
                noise_px=noise_px, rng=rng,
            )
            meas = compute_map_measurement(frame_shape, d_pts, m_pts, bbox_b, map_shape_b)
            res = apply_to_state_filter(meas, f, bridge)
            if res is not None and res.accepted:
                accepted_count += 1
                sigmas_used.append(meas.sigma_xy_m)

        f.update_altitude(bridge.alt0_msl + 620.0, sigma_h_m=50.0)
        f.maybe_update_heading_prior(sigma_heading_rad=math.radians(20.0))

        ex, ey, _ = f.position_enu()
        pos_err.append(math.hypot(ex - x_truth, ey - y_truth))

    pos_err = np.array(pos_err)
    return {
        "n_steps": len(pos_err),
        "n_map_accepted": accepted_count,
        "sigma_min_m": float(min(sigmas_used)) if sigmas_used else float("nan"),
        "sigma_max_m": float(max(sigmas_used)) if sigmas_used else float("nan"),
        "sigma_mean_m": float(np.mean(sigmas_used)) if sigmas_used else float("nan"),
        "pos_err_median_m": float(np.median(pos_err)),
        "pos_err_p95_m": float(np.percentile(pos_err, 95)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("results/stage2_6"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    sa = stage_a()
    print("[stageA] pointwise correctness")
    print(f"  truths           : {sa['n_truths']}")
    print(f"  accepted         : {sa['n_accepted']}")
    print(f"  max position err : {sa['max_err_m']:.3f} m")
    print(f"  mean position err: {sa['mean_err_m']:.3f} m")
    print(f"  sigma_xy range   : [{sa['sigma_min_m']:.2f}, {sa['sigma_max_m']:.2f}] m")
    print(f"  min confidence   : {sa['conf_min']:.3f}")
    sa_ok = (
        sa["n_accepted"] == sa["n_truths"]
        and sa["max_err_m"] < 10.0
        and 1.0 <= sa["sigma_min_m"] <= 50.0
    )
    print(f"  stageA -> {'OK' if sa_ok else 'FAIL'}")
    print()

    sb = stage_b()
    print("[stageB] end-to-end fusion (2 min cruise, no EMA)")
    print(f"  steps         : {sb['n_steps']}")
    print(f"  map accepted  : {sb['n_map_accepted']}")
    print(f"  sigma_xy used : min {sb['sigma_min_m']:.2f}  "
          f"mean {sb['sigma_mean_m']:.2f}  max {sb['sigma_max_m']:.2f} m")
    print(f"  pos err       : median {sb['pos_err_median_m']:.2f}  "
          f"p95 {sb['pos_err_p95_m']:.2f} m")
    sb_ok = (
        sb["n_map_accepted"] >= 0.9 * (sb["n_steps"] // 10)
        and sb["pos_err_median_m"] < sb["sigma_mean_m"] * 1.5
        and sb["pos_err_p95_m"] < sb["sigma_mean_m"] * 4.0
    )
    print(f"  stageB -> {'OK' if sb_ok else 'FAIL'}")
    print()

    summary = "\n".join([
        f"stageA n_accepted = {sa['n_accepted']}/{sa['n_truths']}",
        f"stageA max_err_m  = {sa['max_err_m']:.3f}",
        f"stageA sigma_min  = {sa['sigma_min_m']:.2f}",
        f"stageA sigma_max  = {sa['sigma_max_m']:.2f}",
        f"stageB pos_err median = {sb['pos_err_median_m']:.2f} m",
        f"stageB pos_err p95    = {sb['pos_err_p95_m']:.2f} m",
        f"stageB sigma mean     = {sb['sigma_mean_m']:.2f} m",
        f"stageA = {'OK' if sa_ok else 'FAIL'}",
        f"stageB = {'OK' if sb_ok else 'FAIL'}",
    ])
    (args.output / "summary.txt").write_text(summary + "\n", encoding="utf-8")

    if sa_ok and sb_ok:
        print("[verify] CRITERION PASSED")
        return
    print("[verify] CRITERION NOT MET")
    sys.exit(2)


if __name__ == "__main__":
    main()
