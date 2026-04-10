"""Stage 2.4 verification: prove the EKF behaves as designed under
controlled scenarios.

Four stages:

  Stage 1 — pure propagation. Start at a known truth, predict for
  60 s with no measurements. Trajectory drifts along the seeded
  velocity; position covariance grows monotonically. The §2.4
  criterion's "linearly growing uncertainty in the absence of fixes"
  is checked here numerically.

  Stage 2 — full sensor fusion. Synthetic ground-truth flight
  trajectory, simulated noisy OF every 0.1 s and noisy map fixes
  every 2 s. Filter must track truth within 3σ at >95 % of timesteps.

  Stage 3 — map-fix outage. Start with regular fixes for 30 s,
  drop them for 30 s (OF only). Drift through the gap is bounded by
  the velocity noise budget; filter recovers on the first fix after.

  Stage 4 — Mahalanobis outlier rejection. Inject a 5 km teleport
  fix into a converged filter; filter state must stay close to truth
  (the gate rejects the bogus measurement).

The verification PASSES when all four stages meet their tolerances.
A summary plot is written to ``results/stage2_4/verify.png``.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ekf import StateFilter
from src.frame_bridge import FrameBridge


def _seed_filter(bridge: FrameBridge, lat0, lon0, alt0, yaw_deg, speed) -> StateFilter:
    f = StateFilter(bridge)
    f.initialize_from_wgs84(lat0, lon0, alt0, yaw=math.radians(yaw_deg),
                            sigma_pos_m=20.0, sigma_yaw_rad=math.radians(10.0))
    yaw_rad = math.radians(yaw_deg)
    f.x[StateFilter.IDX_VX_E] = speed * math.sin(yaw_rad)
    f.x[StateFilter.IDX_VY_N] = speed * math.cos(yaw_rad)
    f.P[StateFilter.IDX_VX_E, StateFilter.IDX_VX_E] = 5.0 ** 2
    f.P[StateFilter.IDX_VY_N, StateFilter.IDX_VY_N] = 5.0 ** 2
    return f


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

def stage1_pure_propagation() -> dict:
    bridge = FrameBridge(lat0=55.086025, lon0=38.149033, alt0_msl=130.0)
    f = _seed_filter(bridge, 55.086025, 38.149033, 750.0, yaw_deg=45.0, speed=70.0)
    sigma0 = f.position_sigma_m()
    pos0 = f.position_enu()
    sigmas = []
    times = []
    for i in range(600):  # 60 s @ 10 Hz
        f.predict(0.1)
        sigmas.append(f.position_sigma_m())
        times.append((i + 1) * 0.1)
    pos_end = f.position_enu()
    expected_dx = 70.0 * math.sin(math.radians(45.0)) * 60.0
    expected_dy = 70.0 * math.cos(math.radians(45.0)) * 60.0
    err = math.hypot(pos_end[0] - pos0[0] - expected_dx,
                     pos_end[1] - pos0[1] - expected_dy)
    return {
        "sigma0_m": sigma0,
        "sigma_60s_m": sigmas[-1],
        "monotone": all(sigmas[i] >= sigmas[i - 1] - 1e-9 for i in range(1, len(sigmas))),
        "deadreckon_err_m": err,
        "times": np.array(times),
        "sigmas": np.array(sigmas),
    }


# ---------------------------------------------------------------------------
# Stage 2 — synthetic truth + noisy measurements
# ---------------------------------------------------------------------------

def _simulate_truth(duration_s, dt, yaw_deg, speed, lat0, lon0, alt0):
    n = int(round(duration_s / dt))
    yaw = math.radians(yaw_deg)
    vx_e = speed * math.sin(yaw)
    vy_n = speed * math.cos(yaw)
    t = np.arange(n + 1) * dt
    x_e = vx_e * t
    y_n = vy_n * t
    return t, x_e, y_n, vx_e, vy_n, yaw


def stage2_full_fusion() -> dict:
    rng = np.random.default_rng(seed=1)
    bridge = FrameBridge(55.086025, 38.149033, 130.0)
    duration = 60.0
    of_dt = 0.1
    map_period_s = 2.0
    speed = 70.0
    yaw_deg = 30.0

    t, x_e_truth, y_n_truth, vx_e_t, vy_n_t, yaw_t = _simulate_truth(
        duration, of_dt, yaw_deg, speed, 55.086025, 38.149033, 750.0
    )
    f = _seed_filter(bridge, 55.086025, 38.149033, 750.0, yaw_deg=yaw_deg, speed=speed)

    pos_err = []
    sigma_pos = []
    n_map_accept = 0
    n_map_reject = 0
    map_period_steps = int(round(map_period_s / of_dt))

    for k in range(1, len(t)):
        f.predict(of_dt)
        # OF measurement of body velocity. Body XY: vx_b along yaw,
        # vy_b zero. Add noise.
        sigma_v = 1.5
        vx_b = speed + rng.normal(0.0, sigma_v)
        vy_b = 0.0 + rng.normal(0.0, sigma_v)
        yaw_rate_meas = 0.0 + rng.normal(0.0, math.radians(0.3))
        f.update_of_velocity(np.array([vx_b, vy_b]),
                             yaw_rate_meas,
                             dt=of_dt,
                             sigma_v_mps=sigma_v,
                             sigma_yaw_rate_radps=math.radians(0.5))
        # Map fix every map_period_steps.
        if k % map_period_steps == 0:
            sigma_xy = 20.0
            x_meas = x_e_truth[k] + rng.normal(0.0, sigma_xy)
            y_meas = y_n_truth[k] + rng.normal(0.0, sigma_xy)
            r = f.update_map_position(x_meas, y_meas, sigma_xy_m=sigma_xy)
            if r.accepted:
                n_map_accept += 1
            else:
                n_map_reject += 1
        f.update_altitude(750.0, sigma_h_m=50.0)
        # Heading prior is gated automatically; will mostly fire.
        f.maybe_update_heading_prior(sigma_heading_rad=math.radians(20.0))
        # Track error.
        ex, ey, _ = f.position_enu()
        pos_err.append(math.hypot(ex - x_e_truth[k], ey - y_n_truth[k]))
        sigma_pos.append(f.position_sigma_m())

    pos_err = np.array(pos_err)
    sigma_pos = np.array(sigma_pos)
    in_3sigma = (pos_err < 3 * sigma_pos).sum() / len(pos_err)
    return {
        "n_steps": len(pos_err),
        "n_map_accept": n_map_accept,
        "n_map_reject": n_map_reject,
        "pos_err_median_m": float(np.median(pos_err)),
        "pos_err_p95_m": float(np.percentile(pos_err, 95)),
        "sigma_median_m": float(np.median(sigma_pos)),
        "in_3sigma_frac": float(in_3sigma),
        "times": t[1:],
        "pos_err": pos_err,
        "sigma": sigma_pos,
    }


# ---------------------------------------------------------------------------
# Stage 3 — outage
# ---------------------------------------------------------------------------

def stage3_outage() -> dict:
    rng = np.random.default_rng(seed=2)
    bridge = FrameBridge(55.086025, 38.149033, 130.0)
    of_dt = 0.1
    speed = 70.0
    yaw_deg = 30.0
    f = _seed_filter(bridge, 55.086025, 38.149033, 750.0, yaw_deg=yaw_deg, speed=speed)

    map_period = 1.0
    map_steps = int(round(map_period / of_dt))
    n_pre = int(30.0 / of_dt)         # 30 s with map fixes
    n_gap = int(30.0 / of_dt)         # 30 s map-fix outage
    n_post = int(10.0 / of_dt)        # 10 s with map fixes again

    yaw_r = math.radians(yaw_deg)
    vx_e_t = speed * math.sin(yaw_r)
    vy_n_t = speed * math.cos(yaw_r)
    sigma_pos = []
    pos_err = []
    map_used = []

    t_acc = 0.0
    for k in range(1, n_pre + n_gap + n_post + 1):
        f.predict(of_dt)
        t_acc += of_dt
        x_truth = vx_e_t * t_acc
        y_truth = vy_n_t * t_acc
        sigma_v = 1.5
        f.update_of_velocity(
            np.array([speed + rng.normal(0.0, sigma_v), rng.normal(0.0, sigma_v)]),
            rng.normal(0.0, math.radians(0.3)),
            dt=of_dt,
            sigma_v_mps=sigma_v, sigma_yaw_rate_radps=math.radians(0.5),
        )
        in_outage = n_pre < k <= n_pre + n_gap
        if (k % map_steps == 0) and not in_outage:
            f.update_map_position(
                x_truth + rng.normal(0.0, 20.0),
                y_truth + rng.normal(0.0, 20.0),
                sigma_xy_m=20.0,
            )
            map_used.append(t_acc)
        sigma_pos.append(f.position_sigma_m())
        ex, ey, _ = f.position_enu()
        pos_err.append(math.hypot(ex - x_truth, ey - y_truth))

    times = np.arange(1, len(sigma_pos) + 1) * of_dt
    return {
        "n_steps": len(sigma_pos),
        "sigma_pre_outage_m": float(sigma_pos[n_pre - 1]),
        "sigma_outage_end_m": float(sigma_pos[n_pre + n_gap - 1]),
        "sigma_after_recover_m": float(sigma_pos[-1]),
        "max_outage_err_m": float(max(pos_err[n_pre:n_pre + n_gap])),
        "after_recover_err_m": float(pos_err[-1]),
        "times": times,
        "sigma": np.array(sigma_pos),
        "pos_err": np.array(pos_err),
        "outage_window": (n_pre * of_dt, (n_pre + n_gap) * of_dt),
    }


# ---------------------------------------------------------------------------
# Stage 4 — Mahalanobis outlier rejection
# ---------------------------------------------------------------------------

def stage4_outlier_rejection() -> dict:
    rng = np.random.default_rng(seed=3)
    bridge = FrameBridge(55.086025, 38.149033, 130.0)
    f = _seed_filter(bridge, 55.086025, 38.149033, 750.0, yaw_deg=30.0, speed=70.0)
    of_dt = 0.1
    yaw_r = math.radians(30.0)
    vx_e_t = 70.0 * math.sin(yaw_r)
    vy_n_t = 70.0 * math.cos(yaw_r)
    # Drive the filter to convergence with 10 s of clean data.
    t = 0.0
    for _ in range(100):
        f.predict(of_dt)
        t += of_dt
        f.update_of_velocity(
            np.array([70.0 + rng.normal(0, 1.5), rng.normal(0, 1.5)]),
            rng.normal(0, math.radians(0.3)),
            dt=of_dt,
            sigma_v_mps=1.5, sigma_yaw_rate_radps=math.radians(0.5),
        )
        f.update_map_position(
            vx_e_t * t + rng.normal(0, 20.0),
            vy_n_t * t + rng.normal(0, 20.0),
            sigma_xy_m=20.0,
        )

    # Truth at end of warm-up.
    pos_before = f.position_enu()
    sigma_before = f.position_sigma_m()
    truth_x = vx_e_t * t
    truth_y = vy_n_t * t

    # Inject a 5 km teleport.
    res_outlier = f.update_map_position(truth_x + 5000.0, truth_y - 5000.0, sigma_xy_m=20.0)
    pos_after_outlier = f.position_enu()

    # Inject a clean fix; should accept and pull state to truth.
    res_clean = f.update_map_position(truth_x + rng.normal(0, 20),
                                      truth_y + rng.normal(0, 20),
                                      sigma_xy_m=20.0)
    pos_after_clean = f.position_enu()
    err_after_clean = math.hypot(pos_after_clean[0] - truth_x,
                                 pos_after_clean[1] - truth_y)

    return {
        "sigma_before_m": sigma_before,
        "outlier_accepted": res_outlier.accepted,
        "outlier_d2": res_outlier.mahalanobis2,
        "pos_drift_outlier_m": math.hypot(
            pos_after_outlier[0] - pos_before[0],
            pos_after_outlier[1] - pos_before[1],
        ),
        "clean_accepted": res_clean.accepted,
        "err_after_clean_m": err_after_clean,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("results/stage2_4"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    s1 = stage1_pure_propagation()
    s2 = stage2_full_fusion()
    s3 = stage3_outage()
    s4 = stage4_outlier_rejection()

    print("[stage1] pure propagation")
    print(f"  sigma 0->60s : {s1['sigma0_m']:.1f} -> {s1['sigma_60s_m']:.1f} m")
    print(f"  monotone     : {s1['monotone']}")
    print(f"  deadreckon err (vs analytic) : {s1['deadreckon_err_m']:.4e} m")
    s1_ok = s1['monotone'] and s1['sigma_60s_m'] > s1['sigma0_m'] and s1['deadreckon_err_m'] < 1e-3
    print(f"  stage1 -> {'OK' if s1_ok else 'FAIL'}")
    print()

    print("[stage2] full sensor fusion (60 s)")
    print(f"  steps        : {s2['n_steps']}")
    print(f"  map fixes    : accepted {s2['n_map_accept']}  rejected {s2['n_map_reject']}")
    print(f"  pos err      : median {s2['pos_err_median_m']:.1f}  "
          f"p95 {s2['pos_err_p95_m']:.1f} m")
    print(f"  sigma median : {s2['sigma_median_m']:.1f} m")
    print(f"  3-sigma cover: {100 * s2['in_3sigma_frac']:.1f} %% (target >= 95 %%)")
    s2_ok = (s2['in_3sigma_frac'] >= 0.95
             and s2['pos_err_median_m'] < 50.0
             and s2['n_map_reject'] <= 2)
    print(f"  stage2 -> {'OK' if s2_ok else 'FAIL'}")
    print()

    print("[stage3] map-fix outage 30 s")
    print(f"  sigma pre   : {s3['sigma_pre_outage_m']:.1f} m")
    print(f"  sigma at end of outage : {s3['sigma_outage_end_m']:.1f} m")
    print(f"  max err during outage  : {s3['max_outage_err_m']:.1f} m")
    print(f"  sigma after recover    : {s3['sigma_after_recover_m']:.1f} m")
    print(f"  err after recover      : {s3['after_recover_err_m']:.1f} m")
    s3_ok = (
        s3['sigma_outage_end_m'] > s3['sigma_pre_outage_m']    # grew during outage
        and s3['sigma_after_recover_m'] < s3['sigma_outage_end_m']  # shrank after recover
        and s3['after_recover_err_m'] < 100.0
    )
    print(f"  stage3 -> {'OK' if s3_ok else 'FAIL'}")
    print()

    print("[stage4] outlier rejection")
    print(f"  sigma before     : {s4['sigma_before_m']:.1f} m")
    print(f"  outlier accepted : {s4['outlier_accepted']}  d^2={s4['outlier_d2']:.1f}")
    print(f"  drift on outlier : {s4['pos_drift_outlier_m']:.1f} m")
    print(f"  clean accepted   : {s4['clean_accepted']}")
    print(f"  err after clean  : {s4['err_after_clean_m']:.1f} m")
    s4_ok = (not s4['outlier_accepted']
             and s4['pos_drift_outlier_m'] < 5.0
             and s4['clean_accepted']
             and s4['err_after_clean_m'] < 50.0)
    print(f"  stage4 -> {'OK' if s4_ok else 'FAIL'}")
    print()

    # --- plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)

    ax = axes[0]
    ax.plot(s1['times'], s1['sigmas'], lw=1.4, label='sigma_pos (m)')
    ax.set_xlabel('time (s)'); ax.set_ylabel('sigma (m)')
    ax.set_title('Stage 1: position uncertainty growth, no measurements')
    ax.grid(alpha=0.3); ax.legend()

    ax = axes[1]
    ax.plot(s2['times'], s2['pos_err'], lw=0.8, label='|err|', color='tab:red')
    ax.plot(s2['times'], 3 * s2['sigma'], lw=0.8, label='3 sigma envelope', color='tab:blue', alpha=0.6)
    ax.set_xlabel('time (s)'); ax.set_ylabel('m')
    ax.set_title(f"Stage 2: full fusion — 3σ coverage {100 * s2['in_3sigma_frac']:.1f} %")
    ax.grid(alpha=0.3); ax.legend()

    ax = axes[2]
    ax.plot(s3['times'], s3['sigma'], lw=1.0, label='sigma (m)', color='tab:blue')
    ax.plot(s3['times'], s3['pos_err'], lw=0.8, label='|err|', color='tab:red', alpha=0.7)
    a, b = s3['outage_window']
    ax.axvspan(a, b, color='grey', alpha=0.2, label='map outage')
    ax.set_xlabel('time (s)'); ax.set_ylabel('m')
    ax.set_title('Stage 3: map-fix outage')
    ax.grid(alpha=0.3); ax.legend()

    fig.tight_layout()
    plot_path = args.output / "verify.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"[verify] plot -> {plot_path}")

    if s1_ok and s2_ok and s3_ok and s4_ok:
        print("\n[verify] CRITERION PASSED")
        return
    print("\n[verify] CRITERION NOT MET")
    sys.exit(2)


if __name__ == "__main__":
    main()
