"""Stage 2.5 verification: prove ``velocity_scale_bias`` converges
on noisy data and reduces post-outage drift.

The §2.5 criterion: "after ~5 minutes the velocity_scale_bias
converges to a stable value within ±2 %, and VO drift on subsequent
no-map sections decreases".

We test both halves:

  Stage A — convergence. Truth body speed 70 m/s, but the VO module
  *systematically under-reports* by a factor 1/1.5 (i.e. the true
  scale_bias is 1.5). Filter starts at 1.0 ± 0.3. Run 5 minutes with
  noisy OF + map fixes; check the final estimate is within 2 % of
  truth (1.47 ≤ scale_bias ≤ 1.53).

  Stage B — drift comparison. Run two parallel filters on the same
  synthetic measurement stream:
    - ``corrected``: lets scale_bias adapt (Stage 2.5 enabled).
    - ``baseline``: scale_bias clamped to 1.0 forever.
  After 5 min of fixes, drop map updates for 60 s on both. Compare
  drift at end of outage. The corrected filter must have *smaller*
  drift, since it learned the true scale.

Outputs:
    results/stage2_5/scale_convergence.png — bias trajectory + 1σ band
    results/stage2_5/drift_compare.png      — corrected vs baseline
    results/stage2_5/summary.txt
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


def _make_filter(bridge, yaw_deg, init_speed):
    """Realistic bootstrap: position seeded from a first map fix, but
    velocity left uncertain (σ=30 m/s) so the filter is honestly open
    to whatever the OF says — including a biased reading. Mirrors the
    production bootstrap where we don't know cruise speed in advance.
    """
    f = StateFilter(bridge)
    f.initialize_from_wgs84(bridge.lat0, bridge.lon0, bridge.alt0_msl + 620.0,
                            yaw=math.radians(yaw_deg),
                            sigma_pos_m=20.0, sigma_yaw_rad=math.radians(10.0))
    yr = math.radians(yaw_deg)
    f.x[StateFilter.IDX_VX_E] = init_speed * math.sin(yr)
    f.x[StateFilter.IDX_VY_N] = init_speed * math.cos(yr)
    f.P[StateFilter.IDX_VX_E, StateFilter.IDX_VX_E] = 30.0 ** 2
    f.P[StateFilter.IDX_VY_N, StateFilter.IDX_VY_N] = 30.0 ** 2
    return f


# ---------------------------------------------------------------------------
# Stage A — convergence
# ---------------------------------------------------------------------------

def stage_a_convergence(
    truth_scale: float = 1.5,
    duration_s: float = 300.0,
    of_dt: float = 0.1,
    map_period_s: float = 1.0,
    speed: float = 70.0,
    yaw_deg: float = 30.0,
    sigma_v: float = 1.5,
    sigma_xy: float = 15.0,
    target_tol: float = 0.02,
):
    rng = np.random.default_rng(seed=11)
    bridge = FrameBridge(55.086025, 38.149033, 130.0)
    f = _make_filter(bridge, yaw_deg=yaw_deg, init_speed=speed)

    # The VO outputs body velocity scaled by 1/truth_scale.
    vo_raw_forward = speed / truth_scale

    n_steps = int(round(duration_s / of_dt))
    map_steps = int(round(map_period_s / of_dt))
    yaw_r = math.radians(yaw_deg)
    vx_e_t = speed * math.sin(yaw_r)
    vy_n_t = speed * math.cos(yaw_r)

    times = []
    bias_hist = []
    sigma_hist = []
    err_hist = []
    n_scale_updates = 0

    t_acc = 0.0
    for k in range(1, n_steps + 1):
        f.predict(of_dt)
        t_acc += of_dt
        # Truth aircraft position.
        x_truth = vx_e_t * t_acc
        y_truth = vy_n_t * t_acc
        # OF measurement (raw, biased).
        v_raw = np.array([
            vo_raw_forward + rng.normal(0.0, sigma_v),
            rng.normal(0.0, sigma_v),
        ])
        f.update_of_velocity(
            v_raw,
            rng.normal(0.0, math.radians(0.3)),
            dt=of_dt,
            sigma_v_mps=sigma_v,
            sigma_yaw_rate_radps=math.radians(0.5),
        )
        if k % map_steps == 0:
            x_meas = x_truth + rng.normal(0.0, sigma_xy)
            y_meas = y_truth + rng.normal(0.0, sigma_xy)
            f.update_map_position(x_meas, y_meas, sigma_xy_m=sigma_xy)
        f.update_altitude(bridge.alt0_msl + 620.0, sigma_h_m=50.0)
        f.maybe_update_heading_prior(sigma_heading_rad=math.radians(20.0))

        if k % int(round(1.0 / of_dt)) == 0:  # log every 1 s
            times.append(t_acc)
            bias_hist.append(f.scale_bias())
            sigma_hist.append(f.scale_bias_sigma())
            ex, ey, _ = f.position_enu()
            err_hist.append(math.hypot(ex - x_truth, ey - y_truth))

    times = np.array(times)
    bias_hist = np.array(bias_hist)
    sigma_hist = np.array(sigma_hist)
    err_hist = np.array(err_hist)
    final = bias_hist[-1]
    final_sigma = sigma_hist[-1]
    final_rel_err = abs(final - truth_scale) / truth_scale
    return {
        "truth_scale": truth_scale,
        "final_scale": float(final),
        "final_sigma": float(final_sigma),
        "final_rel_err": float(final_rel_err),
        "passed": final_rel_err <= target_tol,
        "n_scale_updates": int(f.n_scale_updates()),
        "times": times,
        "bias_hist": bias_hist,
        "sigma_hist": sigma_hist,
        "err_hist": err_hist,
        "median_pos_err_m": float(np.median(err_hist)),
        "p95_pos_err_m": float(np.percentile(err_hist, 95)),
    }


# ---------------------------------------------------------------------------
# Stage B — drift comparison after outage
# ---------------------------------------------------------------------------

def stage_b_drift_compare(
    truth_scale: float = 1.5,
    warmup_s: float = 300.0,
    outage_s: float = 60.0,
    of_dt: float = 0.1,
    map_period_s: float = 1.0,
    speed: float = 70.0,
    yaw_deg: float = 30.0,
    sigma_v: float = 1.5,
    sigma_xy: float = 20.0,
):
    rng = np.random.default_rng(seed=22)
    bridge = FrameBridge(55.086025, 38.149033, 130.0)

    # Two parallel filters: corrected (scale adapts) vs baseline (clamped).
    f_corr = _make_filter(bridge, yaw_deg, speed)
    f_base = _make_filter(bridge, yaw_deg, speed)
    # Pin the baseline's scale_bias variance to ~zero so it never adapts.
    f_base.x[StateFilter.IDX_SCALE_BIAS] = 1.0
    f_base.P[StateFilter.IDX_SCALE_BIAS, StateFilter.IDX_SCALE_BIAS] = 1e-12
    f_base.q_scale = 0.0

    vo_raw_forward = speed / truth_scale
    yaw_r = math.radians(yaw_deg)
    vx_e_t = speed * math.sin(yaw_r)
    vy_n_t = speed * math.cos(yaw_r)
    map_steps = int(round(map_period_s / of_dt))

    n_warm = int(round(warmup_s / of_dt))
    n_outage = int(round(outage_s / of_dt))

    drift_corr_hist: list[float] = []
    drift_base_hist: list[float] = []
    times_outage: list[float] = []

    # Same noise stream for both filters so the comparison is paired.
    t_acc = 0.0
    for k in range(1, n_warm + n_outage + 1):
        f_corr.predict(of_dt)
        f_base.predict(of_dt)
        t_acc += of_dt
        x_truth = vx_e_t * t_acc
        y_truth = vy_n_t * t_acc
        nv1 = rng.normal(0.0, sigma_v)
        nv2 = rng.normal(0.0, sigma_v)
        nyr = rng.normal(0.0, math.radians(0.3))
        v_raw = np.array([vo_raw_forward + nv1, nv2])
        f_corr.update_of_velocity(v_raw, nyr, dt=of_dt,
                                  sigma_v_mps=sigma_v,
                                  sigma_yaw_rate_radps=math.radians(0.5))
        f_base.update_of_velocity(v_raw, nyr, dt=of_dt,
                                  sigma_v_mps=sigma_v,
                                  sigma_yaw_rate_radps=math.radians(0.5))
        if k <= n_warm and k % map_steps == 0:
            nx = rng.normal(0.0, sigma_xy)
            ny = rng.normal(0.0, sigma_xy)
            f_corr.update_map_position(x_truth + nx, y_truth + ny, sigma_xy_m=sigma_xy)
            f_base.update_map_position(x_truth + nx, y_truth + ny, sigma_xy_m=sigma_xy)
        f_corr.update_altitude(bridge.alt0_msl + 620.0, sigma_h_m=50.0)
        f_base.update_altitude(bridge.alt0_msl + 620.0, sigma_h_m=50.0)

        if k > n_warm:
            ex, ey, _ = f_corr.position_enu()
            drift_corr_hist.append(math.hypot(ex - x_truth, ey - y_truth))
            ex, ey, _ = f_base.position_enu()
            drift_base_hist.append(math.hypot(ex - x_truth, ey - y_truth))
            times_outage.append(t_acc - warmup_s)

    return {
        "truth_scale": truth_scale,
        "final_scale_corr": float(f_corr.scale_bias()),
        "final_scale_base": float(f_base.scale_bias()),
        "drift_corr_end_m": float(drift_corr_hist[-1]),
        "drift_base_end_m": float(drift_base_hist[-1]),
        "drift_ratio": float(drift_corr_hist[-1] / max(drift_base_hist[-1], 1e-9)),
        "times_outage": np.array(times_outage),
        "drift_corr_hist": np.array(drift_corr_hist),
        "drift_base_hist": np.array(drift_base_hist),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("results/stage2_5"))
    p.add_argument("--truth-scale", type=float, default=1.5)
    p.add_argument("--target-tol", type=float, default=0.02,
                   help="2 %% per §2.5 criterion")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    sa = stage_a_convergence(truth_scale=args.truth_scale, target_tol=args.target_tol)
    sb = stage_b_drift_compare(truth_scale=args.truth_scale)

    print("[stageA] scale_bias convergence on 5 min flight")
    print(f"  truth scale       : {sa['truth_scale']}")
    print(f"  final estimate    : {sa['final_scale']:.4f} ± {sa['final_sigma']:.4f}")
    print(f"  final rel error   : {100 * sa['final_rel_err']:.3f} %% "
          f"(target <= {100 * args.target_tol}%%)")
    print(f"  scale updates     : {sa['n_scale_updates']}")
    print(f"  pos err median/p95: {sa['median_pos_err_m']:.1f} / {sa['p95_pos_err_m']:.1f} m")
    sa_ok = sa["passed"]
    print(f"  stageA -> {'OK' if sa_ok else 'FAIL'}")
    print()

    print("[stageB] drift comparison: corrected vs baseline (60 s outage)")
    print(f"  scale corrected   : {sb['final_scale_corr']:.4f}")
    print(f"  scale baseline    : {sb['final_scale_base']:.4f}")
    print(f"  drift end (corr)  : {sb['drift_corr_end_m']:.1f} m")
    print(f"  drift end (base)  : {sb['drift_base_end_m']:.1f} m")
    print(f"  ratio             : {sb['drift_ratio']:.3f}")
    sb_ok = sb["drift_corr_end_m"] < 0.5 * sb["drift_base_end_m"]
    print(f"  stageB -> {'OK' if sb_ok else 'FAIL'} "
          f"(corrected drift must be < 50%% of baseline)")
    print()

    # --- plots ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sa["times"], sa["bias_hist"], lw=1.4, label="scale_bias estimate")
    ax.fill_between(sa["times"],
                    sa["bias_hist"] - sa["sigma_hist"],
                    sa["bias_hist"] + sa["sigma_hist"],
                    alpha=0.2, label="±1σ")
    ax.axhline(sa["truth_scale"], color="black", ls="--", lw=0.8, label=f"truth = {sa['truth_scale']}")
    band_lo = sa["truth_scale"] * (1 - args.target_tol)
    band_hi = sa["truth_scale"] * (1 + args.target_tol)
    ax.axhspan(band_lo, band_hi, color="green", alpha=0.1, label=f"±{100*args.target_tol:.0f}% target")
    ax.set_xlabel("time (s)"); ax.set_ylabel("scale_bias")
    ax.set_title(f"Stage A: scale_bias convergence  (final {sa['final_scale']:.3f})")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output / "scale_convergence.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sb["times_outage"], sb["drift_corr_hist"], lw=1.4, color="tab:green",
            label=f"corrected (final {sb['drift_corr_end_m']:.1f} m)")
    ax.plot(sb["times_outage"], sb["drift_base_hist"], lw=1.4, color="tab:red",
            label=f"baseline    (final {sb['drift_base_end_m']:.1f} m)")
    ax.set_xlabel("time since map outage start (s)"); ax.set_ylabel("|position error| (m)")
    ax.set_title("Stage B: drift after 60 s map outage")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output / "drift_compare.png", dpi=120)
    plt.close(fig)

    summary = "\n".join([
        f"truth_scale     = {args.truth_scale}",
        f"final_estimate  = {sa['final_scale']:.4f} ± {sa['final_sigma']:.4f}",
        f"rel_err         = {100*sa['final_rel_err']:.3f} %  (target <= {100*args.target_tol}%)",
        f"scale_updates   = {sa['n_scale_updates']}",
        f"drift_corr_end  = {sb['drift_corr_end_m']:.1f} m",
        f"drift_base_end  = {sb['drift_base_end_m']:.1f} m",
        f"drift_ratio     = {sb['drift_ratio']:.3f}",
        f"stageA          = {'OK' if sa_ok else 'FAIL'}",
        f"stageB          = {'OK' if sb_ok else 'FAIL'}",
    ])
    (args.output / "summary.txt").write_text(summary + "\n", encoding="utf-8")

    if sa_ok and sb_ok:
        print("[verify] CRITERION PASSED")
        return
    print("[verify] CRITERION NOT MET")
    sys.exit(2)


if __name__ == "__main__":
    main()
