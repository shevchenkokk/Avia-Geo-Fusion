"""Stage 1.3 isolated test: bootstrap a Geolocator with a known velocity,
let it dead-reckon for N frames, and verify that:

  (a) ``predict_only()`` advances position in the expected direction,
  (b) MapManager fed the predicted positions actually shifts its window
      (i.e. ``tile_id`` changes over time, the §1.3 gate criterion).

This bypasses the matcher entirely — we're not testing match quality
here, only the dead-reckoning loop. Without this, the §1.3 criterion
can't be verified on ``GP010269.MP4`` because the matcher never lands
a first lock to bootstrap the Kalman filter (no telemetry, wrong tile).
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
    # Cruise position somewhere over central Russia; exact value doesn't
    # matter — we only care that prediction + tile shift are consistent.
    start_lat, start_lon = 55.5, 38.5

    print("[test] initializing MapManager + Geolocator at "
          f"({start_lat}, {start_lon})")
    mm = MapManager(zoom=17, window_radius=3, closer_threshold=1)
    map_cv2, bbox = mm.initialize(start_lat, start_lon)
    geo = Geolocator(bbox=bbox, map_shape=map_cv2.shape)

    # Seed Kalman: cruise NE at ~70 m/s ≈ 0.0006 deg per second along
    # both lat and lon. The Kalman state is [lat, lon, v_lat, v_lon]
    # where velocities are *per timestep*. The geolocator runs at the
    # diagnostic sampling rate, so we set per-step velocity directly.
    # 0.0006 deg/step is exaggerated to make tile shifts happen quickly.
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
        # Sanity: prediction should drift NE (lat+, lon+).
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
