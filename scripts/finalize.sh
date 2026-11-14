#!/usr/bin/env bash
# 训练完成后的一键收尾：评测 -> 汇总 -> 出图
set -u
cd "$(dirname "$0")/.." || exit 1

for EXP in A B C D; do
  CK=experiments/exp_${EXP}/best.pth
  if [ ! -f "$CK" ]; then
    echo "[skip] $CK 不存在"
    continue
  fi
  echo "===== eval $EXP ====="
  python src/eval/benchmark.py --ckpt "$CK" --tag "$EXP" --out "experiments/exp_${EXP}"
done

# 退化图基线（不处理）
python src/eval/benchmark.py --tag degraded --out experiments/baseline_degraded

echo "===== summarize ====="
python tools/summarize.py --exps A B C D

echo "===== figures ====="
python tools/plot_results.py --exps A B C D

echo "ALL DONE"
