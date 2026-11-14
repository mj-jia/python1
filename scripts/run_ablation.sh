#!/usr/bin/env bash
# 四组消融串行训练。同 seed、同迭代、同数据量。
# 用法: bash scripts/run_ablation.sh [iters]
set -u
cd "$(dirname "$0")/.." || exit 1

ITERS=${1:-6000}
LOG=experiments/ablation.log
mkdir -p experiments
: > "$LOG"

for EXP in A B C D; do
  echo "===== exp $EXP start $(date '+%F %T') =====" | tee -a "$LOG"
  python src/train/train_sr.py \
    --exp "$EXP" \
    --iters "$ITERS" \
    --batch 16 \
    --patch 256 \
    --workers 6 \
    --val-every 1000 \
    --val-images 12 \
    --save-every 2000 \
    --seed 42 \
    2>&1 | tee -a "$LOG"
  echo "===== exp $EXP done $(date '+%F %T') =====" | tee -a "$LOG"
done

echo "ALL DONE $(date '+%F %T')" | tee -a "$LOG"
