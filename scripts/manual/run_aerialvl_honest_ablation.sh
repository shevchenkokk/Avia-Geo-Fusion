#!/bin/bash
# Полный честный ablation: 4 configs × 5 AerialVL sequences = 20 runs
# Configs:
#   baseline  — XFeat + EKF, без сегментации
#   sem_mask  — + фильтр sat-side по стабильному классу
#   sem_struct — + независимый mask-to-mask NCC канал
#   sem_both  — фильтр + канал
set -e

SEQS=(
  "short_trajtr 2023-03-11-11-48-35"
  "short_trajtr 2023-03-16-16-58-43"
  "short_trajtr 2023-03-18-16-43-16"
  "long_trajtr 2023-03-18-14-38-32"
  "long_trajtr 2023-03-18-15-01-14"
)

SEG_CFG="results/segformer_overture_b0_phase_c_osm_manualcw/segformer_overture_quick_cfg.py"
SEG_CKPT="results/segformer_overture_b0_phase_c_osm_manualcw/best_mIoU_iter_1165.pth"

CONFIGS=(
  "baseline:"
  "sem_mask:--semantic-mask --seg-config $SEG_CFG --seg-checkpoint $SEG_CKPT"
  "sem_struct:--semantic-structural-match --seg-config $SEG_CFG --seg-checkpoint $SEG_CKPT"
  "sem_both:--semantic-mask --semantic-structural-match --seg-config $SEG_CFG --seg-checkpoint $SEG_CKPT"
)

OUT_ROOT="results/aerialvl_honest_ablation"
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
echo "=== 20 RUNS DONE ==="
