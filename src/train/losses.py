"""损失函数：块效应感知损失（创新二）与空间自适应混合损失（创新三）。

两个损失都**不加网络分支**：
  - 块效应是输出的可直接度量属性，用解析度量即可，不需要判别器去"感觉"它；
  - 空间权重来自输入内容本身，不需要额外预测网络。

约定：所有损失输入张量为 (B,3,H,W)，值域 [0,1]。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 与 numpy 评测端保持一致的分母底噪（半个灰度级，换算到 [0,1]）
BLOCKINESS_EPS = 0.5 / 255.0


# --------------------------------------------------------------------------
# 创新二：块效应感知损失
# --------------------------------------------------------------------------
def blockiness_tensor(
    x: torch.Tensor,
    block: int = 8,
    eps: float = BLOCKINESS_EPS,
    phase: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    """逐样本块效应强度 = 块边界梯度均值 / (块内梯度均值 + eps)。

    x    : (B,C,H,W)，值域 [0,1]
    block: 退化图上 8x8 块的**等效像素间距**。训练用 2 倍下采样时该值为 16。
    phase: 8x8 网格原点相对图像左上角的偏移 (r,c)，用于度量非对齐网格上的块效应。
    return: (B,)
    """
    b, c, h, w = x.shape
    pr, pc = int(phase[0]), int(phase[1])
    if pr or pc:
        x = x[..., pr:, pc:]
        h, w = x.shape[-2], x.shape[-1]
    nb_h, nb_w = h // block, w // block
    if nb_h == 0 or nb_w == 0:
        raise ValueError(f"图像尺寸 {h}x{w} 小于一个块 {block}（phase={phase}）")

    x = x[..., : nb_h * block, : nb_w * block]

    # 块边界：相邻两块的末列/首列之差（末块没有"下一块"，需去掉多出的那一个）
    right = (
        x[..., :, block::block] - x[..., :, block - 1 :: block][..., :-1]
    ).abs().mean(dim=(1, 2, 3))
    bottom = (
        x[..., block::block, :] - x[..., block - 1 :: block, :][..., :-1, :]
    ).abs().mean(dim=(1, 2, 3))
    boundary = (right + bottom) / 2.0

    # 块内部：块内相邻像素差（先把块的两个空间维挪到最后，避免切片切错维）
    blk = x.reshape(b, c, nb_h, block, nb_w, block).permute(0, 1, 2, 4, 3, 5)
    inner_h = (blk[..., 1:] - blk[..., :-1]).abs().mean(dim=(1, 2, 3, 4, 5))
    inner_v = (blk[..., 1:, :] - blk[..., :-1, :]).abs().mean(dim=(1, 2, 3, 4, 5))
    inner = (inner_h + inner_v) / 2.0

    return boundary / (inner + eps)


class BlockinessLoss(nn.Module):
    """L_block = relu( blockiness(pred, phase) - blockiness(gt, phase) )。

    单向惩罚：只在"输出比 GT 更像块状"时罚，避免把正常边缘一起压掉。
    可选空间权重 w（创新三），让惩罚集中在平坦区。

    phases:
        1  —— 只在对齐网格（phase=(0,0)）上度量，即原实现；
        N>1 —— 在 N 个候选相位上各算一次，逐样本取**先 relu 再取最大**。
               注意顺序不能反过来：若先对 pred/gt 各自取最坏相位再相减，
               两张图可能取到不同相位，差值没有意义，还会在干净图上误罚。
    """

    def __init__(
        self,
        block: int = 8,
        eps: float = BLOCKINESS_EPS,
        detach_gt: bool = True,
        relative: bool = True,
        phases: int = 1,
    ):
        super().__init__()
        self.block = block
        self.eps = eps
        self.detach_gt = detach_gt
        self.relative = relative
        self.phases = int(phases)
        if self.phases <= 1:
            self.phase_list = [(0, 0)]
        else:
            # 用对角相位 (d,d)：列方向与行方向都覆盖到全部 d，代价只有 8 次度量。
            # 网格相位在真实链路里是 2D 联动的（同一代同时平移行和列），
            # 取对角比取 4x4 叉积更省且不丢覆盖。
            step = 1 if self.block <= 8 else max(1, self.block // 8)
            vals = list(range(0, self.block, step))
            self.phase_list = [(v, v) for v in vals]

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        g = gt.detach() if self.detach_gt else gt
        diffs = []
        for ph in self.phase_list:
            bp = blockiness_tensor(pred, self.block, self.eps, ph)  # (B,)
            bg = blockiness_tensor(g, self.block, self.eps, ph)     # (B,)
            d = F.relu(bp - bg)
            if self.relative:
                # 除以 GT 的块效应水平，做成无量纲相对量，便于与 L1 配权重
                d = d / bg.clamp(min=0.3)
            diffs.append(d)
        diff = diffs[0] if len(diffs) == 1 else torch.stack(diffs, 0).max(0).values

        if weight is None:
            return diff.mean()

        # weight: (B,1,H,W) 或 (B,1,1,1)，1=平坦区
        w = weight
        if w.shape[-1] != 1:
            w = F.adaptive_avg_pool2d(w, 1)
        w = w.reshape(-1)
        denom = w.sum().clamp(min=1e-6)
        return (diff * w).sum() / denom


# --------------------------------------------------------------------------
# 创新三：空间自适应混合损失
# --------------------------------------------------------------------------
def flatness_weight(
    x: torch.Tensor,
    kernel: int = 7,
    tau: float = 0.02,
    detach: bool = True,
) -> torch.Tensor:
    """由局部梯度能量生成"平坦度"权重图 w ∈ (0,1]，1 表示平坦。

    w = tau / (E + tau)，E 为局部平均梯度幅度。
    平坦区 E→0 => w→1（偏 L1 保数值）；纹理区 E 大 => w→0（偏 LPIPS 保观感）。
    """
    if detach:
        x = x.detach()
    g = x.mean(dim=1, keepdim=True)

    gx = torch.zeros_like(g)
    gy = torch.zeros_like(g)
    gx[..., :, 1:] = (g[..., :, 1:] - g[..., :, :-1]).abs()
    gy[..., 1:, :] = (g[..., 1:, :] - g[..., :-1, :]).abs()
    e = gx + gy

    pad = kernel // 2
    e = F.avg_pool2d(F.pad(e, (pad, pad, pad, pad), mode="reflect"), kernel, stride=1)
    return tau / (e + tau)


class LPIPSSpatial(nn.Module):
    """LPIPS 的空间版本。冻结参数，只在需要时加载。"""

    def __init__(self, net: str = "alex", device: str = "cuda"):
        super().__init__()
        try:
            import lpips as _lpips
        except ImportError as e:  # pragma: no cover
            raise ImportError("需要安装 lpips：pip install lpips") from e
        self.net = _lpips.LPIPS(net=net, spatial=True, verbose=False)
        for p in self.net.parameters():
            p.requires_grad = False
        self.net.eval()
        self.register_buffer("_dummy", torch.zeros(1))

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # LPIPS 期望 [-1,1]
        p = pred * 2.0 - 1.0
        g = gt * 2.0 - 1.0
        out = self.net(p, g)
        if out.dim() == 3:
            out = out.unsqueeze(1)
        return out  # (B,1,H,W)


class SpatialAdaptiveLoss(nn.Module):
    """L = mean(w ⊙ L1) / s1 + mean((1-w) ⊙ LPIPS_spatial) / s2

    实现要点（都踩过）：
      1. LPIPS 必须 spatial=True 才有空间图；
      2. 这是**空间**加权，不是训练中自适应调权重系数（后者是 uncertainty weighting）；
      3. L1 与 LPIPS 尺度差约一个量级，加权前必须归一化，否则某一项直接支配；
      4. 平坦区不纯用 L1（L1 最优解偏中值，会加重平滑感），由外部再叠加块效应项。
    """

    def __init__(
        self,
        kernel: int = 7,
        tau: float = 0.02,
        lambda_l1: float = 1.0,
        lambda_lpips: float = 1.0,
        use_lpips: bool = True,
        lpips_net: str = "alex",
        normalize: bool = True,
    ):
        super().__init__()
        self.kernel = kernel
        self.tau = tau
        self.lambda_l1 = lambda_l1
        self.lambda_lpips = lambda_lpips
        self.use_lpips = use_lpips
        self.normalize = normalize
        self.lpips = LPIPSSpatial(net=lpips_net) if use_lpips else None

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> dict:
        w = flatness_weight(gt, self.kernel, self.tau)  # (B,1,H,W)

        l1_map = (pred - gt).abs().mean(dim=1, keepdim=True)  # (B,1,H,W)
        if self.use_lpips and self.lpips is not None:
            lp_map = self.lpips(pred, gt)
            lp_map = F.interpolate(
                lp_map, size=l1_map.shape[-2:], mode="bilinear", align_corners=False
            )
        else:
            # 无 LPIPS 时的退路：用梯度差近似"结构观感"项
            lp_map = (
                (pred - gt).abs().mean(dim=1, keepdim=True)
                + (torch.abs(_grad(pred) - _grad(gt))).mean(dim=1, keepdim=True)
            )

        l1_w = (w * l1_map).mean()
        lp_w = ((1.0 - w) * lp_map).mean()

        if self.normalize:
            s1 = l1_map.detach().mean().clamp(min=1e-6)
            s2 = lp_map.detach().mean().clamp(min=1e-6)
            l1_n, lp_n = l1_w / s1, lp_w / s2
        else:
            l1_n, lp_n = l1_w, lp_w

        total = self.lambda_l1 * l1_n + self.lambda_lpips * lp_n
        return {
            "total": total,
            "l1": l1_n.detach(),
            "lpips": lp_n.detach(),
            "flat_ratio": w.mean().detach(),
        }


def _grad(x: torch.Tensor) -> torch.Tensor:
    """简单前向差分梯度幅值图，保持通道数。"""
    gx = torch.zeros_like(x)
    gy = torch.zeros_like(x)
    gx[..., :, 1:] = x[..., :, 1:] - x[..., :, :-1]
    gy[..., 1:, :] = x[..., 1:, :] - x[..., :-1, :]
    return gx.abs() + gy.abs()


class CharbonnierLoss(nn.Module):
    """可选：比 L1 更平滑的保真项，作为 A 组基线的等价物。"""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((pred - gt) ** 2 + self.eps2).mean()


__all__ = [
    "blockiness_tensor",
    "BlockinessLoss",
    "flatness_weight",
    "SpatialAdaptiveLoss",
    "LPIPSSpatial",
    "CharbonnierLoss",
]
