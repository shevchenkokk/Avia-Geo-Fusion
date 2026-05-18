"""Проверка этапа 2.1: убедиться, что DemLookup возвращает корректные высоты
вдоль траектории, а билинейно интерполированная кривая остаётся гладкой на
известных элементах рельефа.

Критерий из PROJECT_PLAN.md §2.1: «на тестовой траектории через известные
перепады высот AGL меняется ожидаемо и без скачков».

Стратегия:
  1. Собрать синтетический DEM с аналитическим рельефом: холм в центре bbox
     плюс линейный уклон. Форма точная, поэтому ``DemLookup.elevation(lat, lon)``
     можно сравнивать с ``expected_elevation`` в любой точке.
  2. Поточечная корректность: 100 случайных точек внутри bbox; билинейно
     интерполированный ответ должен совпадать с аналитическим полем в пределах
     ~1 м — это ошибка дискретизации 90-метровой сетки на гладком гауссиане.
  3. Гладкость траектории: пройти диагональным маршрутом через холм и построить
     ground elevation и AGL. Сбой вроде ступеньки, NaN или резкого выброса будет
     виден; дополнительно численно проверяется ограниченность |dAGL/ds|.
  4. Запросы вне bbox возвращают None, чтобы вызывающий код мог это обработать.
  5. Краевые случаи: запрос прямо на границе bbox, в центре пикселя и на вершине
     синтетического холма.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dem_lookup import DemLookup
# Переиспользуем аналитическое описание рельефа, чтобы проверка оставалась
# синхронной с генератором: если форма меняется, двигаются обе стороны.
from scripts.make_test_dem import (  # noqa: E402
    expected_elevation,
    HILL_LAT,
    HILL_LON,
    HILL_HEIGHT,
    BASE_MSL,
)


def _check_pointwise(dem: DemLookup, lat0: float, lon0: float,
                     n: int = 100, tol_m: float = 1.5) -> tuple[int, float, float]:
    """Случайные точки внутри bbox; проверяем lookup ≈ analytic."""
    west, south, east, north = dem.bounds
    rng = np.random.default_rng(seed=0)
    # Берём точки на 1% внутрь bbox, чтобы не попасть ровно на край:
    # он проверяется отдельно.
    lats = rng.uniform(south + 0.01 * (north - south), north - 0.01 * (north - south), n)
    lons = rng.uniform(west + 0.01 * (east - west), east - 0.01 * (east - west), n)

    errs = []
    for la, lo in zip(lats, lons):
        got = dem.elevation(float(la), float(lo))
        want = expected_elevation(float(la), float(lo), lat0, lon0)
        assert got is not None, f"got None inside bbox at ({la}, {lo})"
        errs.append(abs(got - want))
    errs = np.array(errs)
    bad = int((errs > tol_m).sum())
    return bad, float(errs.max()), float(errs.mean())


def _check_smoothness(dem: DemLookup, lat0: float, lon0: float,
                      n: int = 2000) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Диагональная траектория по bbox с пересечением холма.

    Возвращает (max |dAGL/ds|, mean |dAGL/ds|, distance_m, agl_m) для графика.
    Нестабильный lookup дал бы отдельные ступеньки, где |dAGL/ds| скачет на
    порядки.
    """
    west, south, east, north = dem.bounds
    # Стартуем немного внутри bbox, чтобы не задеть границу; идём на северо-восток,
    # чтобы траектория прошла рядом с (HILL_LAT, HILL_LON).
    lat_a = south + 0.05 * (north - south)
    lon_a = west + 0.05 * (east - west)
    lat_b = north - 0.05 * (north - south)
    lon_b = east - 0.05 * (east - west)

    t = np.linspace(0.0, 1.0, n)
    lats = lat_a + t * (lat_b - lat_a)
    lons = lon_a + t * (lon_b - lon_a)

    altitude_msl = 750.0
    elev = np.full(n, np.nan)
    agl = np.full(n, np.nan)
    for i in range(n):
        e = dem.elevation(float(lats[i]), float(lons[i]))
        if e is not None:
            elev[i] = e
            agl[i] = altitude_msl - e

    # Приближённая дистанция вдоль траектории в метрах.
    cos_lat = np.cos(np.deg2rad(0.5 * (lat_a + lat_b)))
    dist = np.sqrt(((lats - lat_a) * 111111.0) ** 2
                   + ((lons - lon_a) * 111111.0 * cos_lat) ** 2)

    valid = np.isfinite(agl)
    d_agl = np.diff(agl[valid])
    d_dist = np.diff(dist[valid])
    rate = np.abs(d_agl / np.maximum(d_dist, 1e-9))  # |dAGL/ds|, m/m
    return float(rate.max()), float(rate.mean()), dist, agl


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dem", type=Path, default=Path("data/dem/test_synthetic.tif"))
    p.add_argument("--output", type=Path, default=Path("results/stage2_1"))
    p.add_argument("--altitude-msl", type=float, default=750.0)
    p.add_argument("--tolerance-m", type=float, default=1.5,
                   help="max acceptable error vs analytic field")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if not args.dem.exists():
        raise SystemExit(
            f"DEM not found: {args.dem}\n"
            f"Generate one with: python scripts/make_test_dem.py"
        )

    dem = DemLookup(args.dem)
    west, south, east, north = dem.bounds
    print(f"[verify] DEM    : {args.dem}")
    print(f"[verify] bounds : [{west:.4f}, {south:.4f}, {east:.4f}, {north:.4f}]")
    print(f"[verify] shape  : {dem.shape}")

    # 1. Pointwise correctness against analytic field.
    bad, max_err, mean_err = _check_pointwise(dem, lat0=south, lon0=west,
                                              tol_m=args.tolerance_m)
    print(f"[verify] pointwise: max_err={max_err:.3f} m  mean_err={mean_err:.3f} m  "
          f"out_of_tol={bad}/100")
    pointwise_ok = bad == 0

    # 2. Smoothness along trajectory.
    max_rate, mean_rate, dist_m, agl_m = _check_smoothness(dem, lat0=south, lon0=west)
    # Our synthetic terrain has slope ~0.1 m/m at the hill flank (steepest
    # spot of an 80 m gaussian with sigma~6.6 km). 0.5 m/m would already
    # be unrealistic (Russia is not the Alps); 5+ m/m would mean a glitch.
    print(f"[verify] |dAGL/ds|: max={max_rate:.4f} m/m  mean={mean_rate:.4f} m/m")
    smoothness_ok = max_rate < 0.5  # generous; real value should be ~0.05

    # 3. Out-of-bbox returns None.
    out_of_bbox = dem.elevation(south - 1.0, west - 1.0)
    print(f"[verify] out-of-bbox query: {out_of_bbox} (expect None)")
    bbox_ok = out_of_bbox is None

    # 4. Edge cases.
    edge_se = dem.elevation(south, east)
    edge_nw = dem.elevation(north, west)
    hill = dem.elevation(HILL_LAT, HILL_LON)
    hill_expected = expected_elevation(HILL_LAT, HILL_LON, south, west)
    print(f"[verify] edge SE corner (south, east): {edge_se}")
    print(f"[verify] edge NW corner (north, west): {edge_nw}")
    print(f"[verify] hill peak (~{HILL_LAT}, {HILL_LON}): "
          f"got {hill:.2f}  expected {hill_expected:.2f}")
    edge_ok = (edge_se is not None and edge_nw is not None
               and hill is not None
               and abs(hill - hill_expected) < args.tolerance_m)

    # 5. AGL contract.
    agl_at_hill = dem.height_agl(HILL_LAT, HILL_LON, args.altitude_msl)
    print(f"[verify] AGL at hill peak (msl={args.altitude_msl}): {agl_at_hill:.2f} m")

    # --- plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    valid = np.isfinite(agl_m)
    axes[0].plot(dist_m[valid], args.altitude_msl - agl_m[valid], lw=1.2,
                 label="terrain elevation (m MSL)")
    axes[0].axhline(args.altitude_msl, color="grey", ls=":", label=f"aircraft alt {args.altitude_msl} m")
    axes[0].set_ylabel("elevation (m MSL)")
    axes[0].set_title("Stage 2.1 — DEM lookup along diagonal flight track")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].plot(dist_m[valid], agl_m[valid], lw=1.2, color="tab:green")
    axes[1].set_ylabel("AGL (m)")
    axes[1].set_xlabel("distance along track (m)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    plot_path = args.output / "agl_track.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"[verify] plot   : {plot_path}")

    summary_lines = [
        f"DEM: {args.dem}",
        f"bounds:    [{west:.4f}, {south:.4f}, {east:.4f}, {north:.4f}]",
        f"shape:     {dem.shape}",
        f"pointwise: max_err={max_err:.3f} m mean_err={mean_err:.3f} m  -> "
        f"{'OK' if pointwise_ok else 'FAIL'}",
        f"smoothness:|dAGL/ds| max={max_rate:.4f} m/m mean={mean_rate:.4f} -> "
        f"{'OK' if smoothness_ok else 'FAIL'}",
        f"out-of-bbox returns None -> {'OK' if bbox_ok else 'FAIL'}",
        f"edge / hill peak match analytic -> {'OK' if edge_ok else 'FAIL'}",
        f"AGL @ hill peak = {agl_at_hill:.2f} m  (alt_msl={args.altitude_msl})",
    ]
    summary_txt = "\n".join(summary_lines)
    (args.output / "summary.txt").write_text(summary_txt + "\n", encoding="utf-8")
    print()
    print(summary_txt)

    all_ok = pointwise_ok and smoothness_ok and bbox_ok and edge_ok
    if all_ok:
        print("\n[verify] CRITERION PASSED")
        return
    print("\n[verify] CRITERION NOT MET")
    sys.exit(2)


if __name__ == "__main__":
    main()
