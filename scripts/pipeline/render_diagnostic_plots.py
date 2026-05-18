"""Построить диагностическую сводку и графики state-timeline из готового
``frames_<run>.csv``.

Полезно, если длинный диагностический прогон был прерван на середине, но
частичный CSV ещё пригоден, или если нужно перерисовать графики с другим
оформлением без повторного прохода матчера.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diagnostic_logger import DiagnosticLogger, _FIELDS  # type: ignore[attr-defined]


def _coerce_row(row: dict[str, str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for k in _FIELDS:
        v = row.get(k, "")
        if k in {"frame_idx", "num_keypoints_frame", "num_keypoints_tile",
                 "num_matches", "num_inliers"}:
            out[k] = int(v) if v not in {"", "-1"} else (int(v) if v == "-1" else -1)
        elif k in {"tile_changed", "is_keyframe"}:
            out[k] = (str(v).lower() == "true")
        elif k in {"backend", "state", "reject_reason", "tile_id"}:
            out[k] = v
        else:
            try:
                out[k] = float(v) if v != "" else float("nan")
            except ValueError:
                out[k] = float("nan")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path)
    p.add_argument("--run-name", type=str, default=None)
    args = p.parse_args()

    csv_path = args.csv
    output_dir = csv_path.parent
    run_name = args.run_name or csv_path.stem.replace("frames_", "")

    rows: list[dict[str, object]] = []
    with csv_path.open(encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for r in reader:
            rows.append(_coerce_row(r))

    logger = DiagnosticLogger(output_dir=output_dir, run_name=run_name + "_replot")
    # Подкладываем строки напрямую без перезаписи CSV: нужны только графики.
    logger._rows = rows  # noqa: SLF001
    logger._fp.close()
    summary = logger._summary()
    logger._render_plot(summary)
    logger._render_state_timeline()
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nrendered plots in {output_dir}")


if __name__ == "__main__":
    main()
