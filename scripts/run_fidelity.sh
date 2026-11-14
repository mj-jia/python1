#!/usr/bin/env bash
# 保真优先训练（目标：尽量接近原图，PSNR/SSIM 主导）
#
# 与 exp_D 的区别只有超参：
#   lambda_lpips 0.1  -> 0.05   （感知项减半：既要数值贴近原图，也要"看着像"，
#                                 压到 0.03 会让观感退回 A 组那种"糊但数值好"）
#   lambda_block 0.05 -> 0.03
#   iters        6000 -> 20000  （6000 只是跑通流程的量）
set -u
cd "$(dirname "$0")/.." || exit 1

LOG=experiments/fidelity.log
: > "$LOG"

python src/train/train_sr.py \
  --exp D \
  --out experiments/exp_F \
  --iters 20000 \
  --batch 16 \
  --patch 256 \
  --workers 6 \
  --lambda-lpips 0.05 \
  --lambda-block 0.03 \
  --val-every 2000 \
  --val-images 12 \
  --save-every 5000 \
  --seed 42 \
  2>&1 | tee -a "$LOG"

echo "FIDELITY DONE $(date '+%F %T')" | tee -a "$LOG"
