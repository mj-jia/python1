#!/usr/bin/env bash
# 重跑 C/D 两组（LPIPS 权重已从 1.0 调整为 0.1）
set -u
cd "$(dirname "$0")/.." || exit 1

LOG=experiments/cd.log
: > "$LOG"

for EXP in C D; do
  echo "===== exp $EXP start $(date '+%F %T') =====" | tee -a "$LOG"
  python src/train/train_sr.py \
    --exp "$EXP" \
    --iters 6000 \
    --batch 16 \
    --patch 256 \
    --workers 6 \
    --val-every 1000 \
    --val-images 12 \
    --lambda-lpips 0.1 \
    --seed 42 \
    2>&1 | tee -a "$LOG"
  echo "===== exp $EXP done $(date '+%F %T') =====" | tee -a "$LOG"
done
echo "CD DONE $(date '+%F %T')" | tee -a "$LOG"
