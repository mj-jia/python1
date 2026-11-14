"""Real-ESRGAN 轻量骨干 SRVGGNetCompact（结构不改，scale=1）。

参数量与官方 realesr-general-x4v3 一致（≈1.15M），差别只在最后一层的
像素重排倍数：本项目是同分辨率复原（scale=1），不做超分。
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _act(kind: str, num_feat: int) -> nn.Module:
    if kind == "prelu":
        return nn.PReLU(num_parameters=num_feat)
    if kind == "relu":
        return nn.ReLU(inplace=True)
    if kind == "leakyrelu":
        return nn.LeakyReLU(0.1, inplace=True)
    raise ValueError(f"未知激活 {kind}")


class SRVGGNetCompact(nn.Module):
    """ECCV 2022 Real-ESRGAN 的 compact 生成器。

    scale=1 时不接 PixelShuffle，直接输出同尺寸残差。
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 32,
        upscale: int = 1,
        act_type: str = "prelu",
    ):
        super().__init__()
        self.upscale = upscale
        body: list[nn.Module] = [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)]
        body.append(_act(act_type, num_feat))
        for _ in range(num_conv - 1):
            body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            body.append(_act(act_type, num_feat))
        body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.body = nn.Sequential(*body)
        self.upsampler = nn.PixelShuffle(upscale) if upscale > 1 else nn.Identity()

        self._init()

    def _init(self) -> None:
        convs = [m for m in self.modules() if isinstance(m, nn.Conv2d)]
        for m in convs:
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # 最后一层置零：让初始输出严格等于输入（恒等映射），
        # 训练起点就是"什么都不做"，不会一上来就把图弄坏
        if convs:
            nn.init.zeros_(convs[-1].weight)
            if convs[-1].bias is not None:
                nn.init.zeros_(convs[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.body(x)
        out = self.upsampler(out)
        return out + x  # 残差学习：模型只学"要补回什么"

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(scale: int = 1, num_feat: int = 64, num_conv: int = 32) -> SRVGGNetCompact:
    return SRVGGNetCompact(upscale=scale, num_feat=num_feat, num_conv=num_conv)


__all__ = ["SRVGGNetCompact", "build_model"]
