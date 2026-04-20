"""Stage 3.5 step 2: prove the retriever picks plausible tiles for
real cruise frames from GP010269.MP4.

Without telemetry on this footage we can't enforce a tight error
bound; the §3.5 verification is qualitative on three dimensions:

  1. **Signal vs noise.** The top-1 cosine similarity must
     significantly exceed the median over the whole DB. If they're
     comparable, the retriever is essentially returning random tiles.

  2. **Temporal consistency.** Cruise frames 1 s apart should retrieve
     either the same tile or geographic neighbours — the aircraft
     moves only ~70 m/s, well below the 1.4-km tile spacing at z=14,
     so consecutive top-1's must be within ≤2 tiles of each other.

  3. **Visual plausibility.** A contact sheet of (query frame, top-1
     retrieved tile) is written for human inspection.

If all three pass, the retriever is good enough to seed MapManager
for the full pipeline. Tuning to AnyLoc-grade recall (VLAD over
patches) is a follow-up if needed.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# macOS Python SSL cert workaround for torch.hub fetches.
try:
    import certifi  # type: ignore
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except Exception:
    pass

from src.retriever import Retriever
from src.video_processor import VideoProcessor


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return float(2 * R * math.asin(math.sqrt(a)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=Path("data/retrieval/db_z14"))
    p.add_argument("--video", type=Path, default=Path("data/videos/GP010269.MP4"))
    p.add_argument("--output", type=Path, default=Path("results/stage3_5"))
    p.add_argument("--cruise-start-s", type=float, default=45.0)
    p.add_argument("--cruise-end-s", type=float, default=110.0)
    p.add_argument("--sample-period-s", type=float, default=1.0,
                   help="for temporal-consistency check")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if not args.video.exists():
        raise SystemExit(f"video not found: {args.video}")

    retr = Retriever(device=args.device)
    retr.load_database(args.db)
    db = retr.database
    print(f"[verify] retriever device: {retr.device}")
    print(f"[verify] DB: {args.db}.npz  (N={len(db)} z={db.zoom})")

    proc = VideoProcessor(str(args.video), default_geo=(55.086025, 38.149033, 750.0))
    src_fps = proc.info.fps

    # Sample frames in clean cruise window.
    times = []
    cur = args.cruise_start_s
    while cur <= args.cruise_end_s:
        times.append(cur)
        cur += args.sample_period_s
    print(f"[verify] sampling {len(times)} frames at {args.sample_period_s} s spacing")

    # Stage A — score-vs-noise plus per-frame results.
    rows = []
    for t in times:
        fi = int(round(t * src_fps))
        frame = proc.extract_frame(fi)
        if frame is None:
            continue
        d = retr.encode(frame)
        # Full distribution for noise floor.
        scores_all = db.descriptors @ d
        med = float(np.median(scores_all))
        mx = float(scores_all.max())
        top = retr.query(d, top_k=args.top_k)
        rows.append({
            "t": t, "frame_idx": fi,
            "top1_score": top[0][1], "top1_lat": db.centers_latlon[top[0][0], 0],
            "top1_lon": db.centers_latlon[top[0][0], 1],
            "top1_tile": db.tile_ids[top[0][0]],
            "topK": top,
            "median_score": med, "max_score": mx,
            "frame": frame,
        })

    # ---- Stage A: signal vs noise --------------------------------------
    top1_scores = np.array([r["top1_score"] for r in rows])
    median_scores = np.array([r["median_score"] for r in rows])
    margin = top1_scores - median_scores
    print()
    print("[stageA] signal vs noise")
    print(f"  top1 median        : {float(np.median(top1_scores)):.4f}")
    print(f"  DB-median  median  : {float(np.median(median_scores)):.4f}")
    print(f"  top1 - DBmed median: {float(np.median(margin)):.4f}")
    sa_ok = float(np.median(margin)) > 0.05
    print(f"  stageA -> {'OK' if sa_ok else 'FAIL'} (criterion: top1 - DBmed median > 0.05)")

    # ---- Stage B: temporal consistency ---------------------------------
    print()
    print("[stageB] temporal consistency (consecutive samples)")
    jumps_m = []
    consistent_count = 0
    for r0, r1 in zip(rows[:-1], rows[1:]):
        d_m = _haversine_m(r0["top1_lat"], r0["top1_lon"], r1["top1_lat"], r1["top1_lon"])
        jumps_m.append(d_m)
        # Two adjacent z=14 tiles (1.4 km), allow 2 hops = 2.8 km.
        if d_m <= 2800.0:
            consistent_count += 1
    n_pairs = max(1, len(rows) - 1)
    consistent_frac = consistent_count / n_pairs
    print(f"  n_pairs            : {n_pairs}")
    print(f"  median hop (m)     : {float(np.median(jumps_m)):.0f}")
    print(f"  p95 hop (m)        : {float(np.percentile(jumps_m, 95)):.0f}")
    print(f"  within 2 tiles (%) : {100*consistent_frac:.1f}")
    sb_ok = consistent_frac >= 0.7
    print(f"  stageB -> {'OK' if sb_ok else 'FAIL'} (criterion: ≥70 % consistent hops)")

    # ---- Stage C: contact sheet ----------------------------------------
    # Pull the matching tile image to render side-by-side. We re-fetch
    # via MapDownloader to avoid re-indexing the cache.
    from src.map_loader import MapDownloader
    loader = MapDownloader(zoom=db.zoom)
    contact_rows = []
    sample_indices = list(range(0, len(rows), max(1, len(rows) // 8)))
    for ri in sample_indices[:8]:
        r = rows[ri]
        tile_z, tile_x, tile_y = (int(s) for s in r["top1_tile"].split("/"))
        pil = loader.download_tile(tile_x, tile_y, tile_z)
        if pil is None:
            continue
        tile_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        # Centre-crop the query frame to a square and resize to tile width.
        f = r["frame"]
        h, w = f.shape[:2]
        s = min(h, w)
        y0 = (h - s) // 2; x0 = (w - s) // 2
        q = f[y0:y0+s, x0:x0+s]
        side = tile_bgr.shape[0]
        q = cv2.resize(q, (side, side))
        sheet = np.hstack([q, tile_bgr])

        def _label(img, txt):
            o = img.copy()
            cv2.putText(o, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 4, cv2.LINE_AA)
            cv2.putText(o, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
            return o
        sheet = _label(
            sheet,
            f"f={r['frame_idx']} t={r['t']:.0f}s  top1_score={r['top1_score']:.3f}  "
            f"tile={r['top1_tile']} ({r['top1_lat']:.4f},{r['top1_lon']:.4f})",
        )
        contact_rows.append(sheet)
    if contact_rows:
        contact = np.vstack(contact_rows)
        cv2.imwrite(str(args.output / "contact.png"), contact)
        print(f"\n[verify] contact -> {args.output / 'contact.png'}")

    # ---- Summary -------------------------------------------------------
    summary_lines = [
        f"DB: {args.db}.npz   N={len(db)}",
        f"sampled frames: {len(rows)}",
        f"top1 score median:        {float(np.median(top1_scores)):.4f}",
        f"DB-median  score median:  {float(np.median(median_scores)):.4f}",
        f"top1 - DBmed margin:      {float(np.median(margin)):.4f}",
        f"temporal hops median (m): {float(np.median(jumps_m)):.0f}",
        f"temporal hops p95 (m):    {float(np.percentile(jumps_m, 95)):.0f}",
        f"within-2-tiles fraction:  {consistent_frac:.2f}",
        f"stageA = {'OK' if sa_ok else 'FAIL'}",
        f"stageB = {'OK' if sb_ok else 'FAIL'}",
    ]
    (args.output / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print()
    print("\n".join(summary_lines))

    if sa_ok and sb_ok:
        print("\n[verify] CRITERION PASSED")
        return
    print("\n[verify] CRITERION NOT MET — inspect contact.png")
    sys.exit(2)


if __name__ == "__main__":
    main()
