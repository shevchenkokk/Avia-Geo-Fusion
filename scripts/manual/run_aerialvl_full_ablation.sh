#!/bin/bash
# Полный ablation: 5 AerialVL sequences x 3 configs.
set -e

SEQS=(
  "short_trajtr 2023-03-11-11-48-35"
  "short_trajtr 2023-03-16-16-58-43"
  "short_trajtr 2023-03-18-16-43-16"
  "long_trajtr 2023-03-18-14-38-32"
  "long_trajtr 2023-03-18-15-01-14"
)

CONFIGS=(
  "baseline:"
  "sem_old:--semantic-mask --seg-config results/segformer_overture_b0_focus_resume_wclean_v3/focus_refine/segformer_overture_focus_cfg.py --seg-checkpoint results/segformer_overture_b0_focus_resume_wclean_v3/focus_refine/best_mIoU_iter_220.pth"
  "sem_new:--semantic-mask --seg-config results/segformer_overture_b0_phase_c_osm_manualcw/segformer_overture_quick_cfg.py --seg-checkpoint results/segformer_overture_b0_phase_c_osm_manualcw/best_mIoU_iter_1165.pth"
)

OUT_ROOT="results/aerialvl_full_ablation"
mkdir -p "$OUT_ROOT"

for SEQ_LINE in "${SEQS[@]}"; do
  SPLIT="${SEQ_LINE%% *}"
  SEQ="${SEQ_LINE#* }"
  SEQ_TAG="${SPLIT}_${SEQ}"
  for CFG_LINE in "${CONFIGS[@]}"; do
    CFG_NAME="${CFG_LINE%%:*}"
    CFG_ARGS="${CFG_LINE#*:}"
    OUT_DIR="$OUT_ROOT/${SEQ_TAG}__${CFG_NAME}"
    if [[ -f "$OUT_DIR/summary.json" ]]; then
      echo "SKIP $OUT_DIR (already done)"
      continue
    fi
    echo "=== $SPLIT/$SEQ / $CFG_NAME ==="
    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python3 scripts/run_aerialvl_pipeline.py \
      --dataset-root data/external/aerialvl \
      --split "$SPLIT" --sequence "$SEQ" \
      --backend xfeat --frame-stride 1 \
      --output "$OUT_DIR" \
      $CFG_ARGS 2>&1 | tail -3
  done
done

echo
echo "=== ALL 15 RUNS DONE ==="
echo
.venv/bin/python3 -c "
import json, glob, os
print(f\"{'sequence':<35}{'config':<12}{'med':>9}{'mean':>9}{'p95':>9}{'final':>9}\")
for p in sorted(glob.glob('$OUT_ROOT/*/summary.json')):
    s = json.load(open(p))
    name = os.path.basename(os.path.dirname(p))
    seq, cfg = name.rsplit('__', 1)
    seq = seq.replace('short_trajtr_', 'short/').replace('long_trajtr_', 'long/')
    print(f\"{seq:<35}{cfg:<12}{s['median_error_m']:>8.1f}m{s['mean_error_m']:>8.1f}m{s['p95_error_m']:>8.1f}m{s['final_error_m']:>8.1f}m\")
"
