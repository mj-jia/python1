"""对抗训练组件（实验用，不影响原有 D/G 组）。

判别器用 UNet 结构 + 谱归一化（Real-ESRGAN 的做法）：
  - 4 次下采样获取全局感受野，再逐级上采样回原分辨率
  - 输出逐像素 logits（patch 判别），比单标量判别提供更密的梯度
  - 谱归一化代替 BN，训练更稳
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetDiscriminatorSN(nn.Module):
    """UNet 判别器，输出与输入同分辨率的 logits 图。"""

    def __init__(self, num_in_ch: int = 3, num_feat: int = 48, skip: bool = True):
        super().__init__()
        self.skip = skip
        sn = nn.utils.spectral_norm
        f = num_feat

        self.conv0 = sn(nn.Conv2d(num_in_ch, f, 3, 1, 1))
        self.conv1 = sn(nn.Conv2d(f, f * 2, 4, 2, 1, bias=False))
        self.conv2 = sn(nn.Conv2d(f * 2, f * 4, 4, 2, 1, bias=False))
        self.conv3 = sn(nn.Conv2d(f * 4, f * 8, 4, 2, 1, bias=False))
        self.conv4 = sn(nn.Conv2d(f * 8, f * 8, 3, 1, 1, bias=False))

        self.up3 = sn(nn.Conv2d(f * 8, f * 4, 3, 1, 1, bias=False))
        self.up2 = sn(nn.Conv2d(f * 8 if skip else f * 4, f * 2, 3, 1, 1, bias=False))
        self.up1 = sn(nn.Conv2d(f * 4 if skip else f * 2, f, 3, 1, 1, bias=False))
        self.conv_out = sn(nn.Conv2d(f * 2 if skip else f, 1, 3, 1, 1))

        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.act(self.conv0(x))                     # 1/1
        x1 = self.act(self.conv1(x0))                    # 1/2
        x2 = self.act(self.conv2(x1))                    # 1/4
        x3 = self.act(self.conv3(x2))                    # 1/8
        x4 = self.act(self.conv4(x3))                    # 1/8

        u3 = self.act(self.up3(x4))
        u3 = F.interpolate(u3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        u3 = torch.cat([u3, x2], 1) if self.skip else u3
        u2 = self.act(self.up2(u3))
        u2 = F.interpolate(u2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        u2 = torch.cat([u2, x1], 1) if self.skip else u2
        u1 = self.act(self.up1(u2))
        u1 = F.interpolate(u1, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        u1 = torch.cat([u1, x0], 1) if self.skip else u1
        return self.conv_out(u1)


class GANLoss(nn.Module):
    """非饱和对抗损失（生成器用），比饱和版梯度更稳。"""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward_d(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        return 0.5 * (
            self.bce(real, torch.ones_like(real)) + self.bce(fake, torch.zeros_like(fake))
        )

    def forward_g(self, fake: torch.Tensor) -> torch.Tensor:
        return self.bce(fake, torch.ones_like(fake))


__all__ = ["UNetDiscriminatorSN", "GANLoss"]
