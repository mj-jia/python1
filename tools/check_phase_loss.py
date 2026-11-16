"""损失侧冒烟检查：多相位块效应损失能不能识破「错位网格」。

构造三张图，GT 相同：
  gt          干净图
  pred_align  在 (0,0) 对齐网格上加块边界台阶
  pred_shift  在偏移 4 像素的网格上加同样强度的台阶

期望：
  单相位损失  —— 对 pred_align 有惩罚，对 pred_shift 几乎无感（先验失配）
  多相位损失  —— 两者都能识别，且量级相近
  两者对 pred == gt 都必须 ≈ 0（不在干净图上误罚）

用法：python tools/check_phase_loss.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train.losses import BlockinessLoss  # noqa: E402


def smooth_image(size: int = 128, seed: int = 0) -> torch.Tensor:
    g = torch.rand(1, 1, size, size, generator=torch.Generator().manual_seed(seed))
    k = torch.ones(1, 1, 9, 9) / 81.0
    g = torch.nn.functional.conv2d(g, k, padding=4)
    return g.repeat(1, 3, 1, 1).clamp(0, 1)


def add_step(x: torch.Tensor, phase: int, amp: float = 0.04) -> torch.Tensor:
    """在 x % 8 == phase 的位置加一格台阶，模拟块边界不连续。"""
    y = x.clone()
    size = y.shape[-1]
    idx = torch.arange(size) % 8 == phase
    y[..., idx] += amp
    y[..., idx, :] += amp
    return y.clamp(0, 1)


def main() -> int:
    gt = smooth_image()
    pred_align = add_step(gt, 7)   # 网格原点 0，边界落在 7
    pred_shift = add_step(gt, 3)   # 平移 4 像素后的网格

    rows = []
    for name, phases in (("single", 1), ("multi", 8)):
        fn = BlockinessLoss(block=8, phases=phases)
        rows.append(
            (
                name,
                float(fn(gt, gt)),
                float(fn(pred_align, gt)),
                float(fn(pred_shift, gt)),
            )
        )

    print(f"{'mode':8s} {'pred==gt':>10s} {'对齐网格':>10s} {'错位网格':>10s}")
    for name, a, b, c in rows:
        print(f"{name:8s} {a:10.4f} {b:10.4f} {c:10.4f}")

    single = rows[0]
    multi = rows[1]
    print("\n判读：")
    print(f"  单相位对错位网格的检出率 = {single[3] / max(single[2], 1e-9):.2f}")
    print(f"  多相位对错位网格的检出率 = {multi[3] / max(multi[2], 1e-9):.2f}")
    ok = multi[3] > 3 * single[3] and multi[1] < 1e-3 and single[1] < 1e-3
    print(f"  结论：{'通过（多相位显著更能识破错位，且干净图不误罚）' if ok else '不通过，需复查'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
