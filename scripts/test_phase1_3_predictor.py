"""Изолированный тест этапа 1.3: инициализировать Geolocator известной скоростью,
дать ему N кадров счисления пути и проверить, что:

  (a) ``predict_only()`` сдвигает позицию в ожидаемом направлении;
  (b) MapManager, получая предсказанные позиции, действительно сдвигает окно,
      то есть ``tile_id`` со временем меняется — критерий ворот из §1.3.

Матчер здесь полностью обходится: проверяется не качество сопоставления, а
контур счисления пути. Без этого критерий §1.3 нельзя проверить на
``GP010269.MP4``, потому что матчер не получает первый захват для bootstrap
фильтра Калмана: телеметрии нет, тайл неверный.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.geolocator import Geolocator
from src.map_manager import MapManager


def _tile_id(mm: MapManager) -> str:
    return f"{mm.zoom}/{mm.center_tx}/{mm.center_ty}"


def main() -> None:
    # Крейсерская позиция где-то над центральной Россией; точное значение
    # не важно, нужна только согласованность прогноза и сдвига тайла.
    start_lat, start_lon = 55.5, 38.5

    print("[test] initializing MapManager + Geolocator at "
          f"({start_lat}, {start_lon})")
    mm = MapManager(zoom=17, window_radius=3, closer_threshold=1)
    map_cv2, bbox = mm.initialize(start_lat, start_lon)
    geo = Geolocator(bbox=bbox, map_shape=map_cv2.shape)

    # Инициализируем Калман: крейсер на северо-восток около 70 м/с ≈ 0.0006°
    # в секунду и по lat, и по lon. Состояние Калмана — [lat, lon, v_lat, v_lon],
    # где скорости заданы *на шаг*. Geolocator работает на частоте диагностики,
    # поэтому задаём скорость за шаг напрямую. 0.0006°/шаг специально завышено,
    # чтобы сдвиг тайлов проявился быстро.
    v_lat, v_lon = 0.0006, 0.0006
    geo.kalman.statePre = np.array([[start_lat], [start_lon], [v_lat], [v_lon]], np.float32)
    geo.kalman.statePost = np.array([[start_lat], [start_lon], [v_lat], [v_lon]], np.float32)
    geo.is_kalman_inited = True

    initial_tile = _tile_id(mm)
    print(f"[test] initial tile     = {initial_tile}")
    print(f"[test] initial bbox     = {bbox}")

    seen_tiles = {initial_tile}
    last_pred = (start_lat, start_lon)
    for step in range(1, 41):
        pred = geo.predict_only()
        assert pred is not None, "predict_only returned None despite initialized Kalman"
        pred_lat, pred_lon = pred
        # Sanity-check: прогноз должен дрейфовать на северо-восток (lat+, lon+).
        assert pred_lat > last_pred[0] - 1e-9, f"lat went backwards at step {step}"
        assert pred_lon > last_pred[1] - 1e-9, f"lon went backwards at step {step}"
        last_pred = pred

        new_map, new_bbox = mm.update(pred_lat, pred_lon)
        if new_bbox != bbox:
            bbox = new_bbox
            geo.update_bbox(new_bbox, new_map.shape)
        tid = _tile_id(mm)
        if tid not in seen_tiles:
            seen_tiles.add(tid)
            d_lat = pred_lat - start_lat
            d_lon = pred_lon - start_lon
            print(f"[test] step {step:>3}  pred=({pred_lat:.5f}, {pred_lon:.5f})  "
                  f"d=(+{d_lat*111111:.0f}m, +{d_lon*111111:.0f}m)  "
                  f"tile={tid}  (NEW)")

    print()
    print(f"[test] distinct tiles visited: {len(seen_tiles)}")
    for t in sorted(seen_tiles):
        print(f"   {t}")
    if len(seen_tiles) < 2:
        raise SystemExit("CRITERION FAILED: tile_id never changed across 40 predicted steps")
    print("[test] CRITERION PASSED: tile_id changed across the dead-reckoning trajectory")


if __name__ == "__main__":
    main()
