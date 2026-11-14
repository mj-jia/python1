"""成对数据集：HR 原图 + 在线生成的传输退化图。

退化在 DataLoader worker 里用 numpy 在线生成，每轮参数都不同，
避免把退化"背下来"（固定退化会让模型学到特定几张图的逆映射）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.degrade.dct_degrade import DegradeConfig, degrade_image  # noqa: E402

cv2.setNumThreads(0)  # worker 内不要开多线程，否则线程数爆炸


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """(H,W,3) uint8 -> (3,H,W) float32 [0,1]"""
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float() / 255.0


class PairedDegradeDataset(Dataset):
    """随机裁 patch + 在线退化。

    hr_dir    : HR 图目录
    patch     : 输出 patch 边长，需为 16*max(scale_choices) 的倍数
    length    : 每个 epoch 的样本数（HR 图只有 800 张，靠重复采样）
    fixed_seed: 给定则退化参数可复现（验证/测试用）
    """

    def __init__(
        self,
        hr_dir: str | Path,
        patch: int = 512,
        length: int = 1000,
        cfg: DegradeConfig | None = None,
        fixed_seed: int | None = None,
        training: bool = True,
    ):
        super().__init__()
        self.files = sorted(Path(hr_dir).glob("*.png"))
        if not self.files:
            raise FileNotFoundError(f"{hr_dir} 下没有 png")
        self.patch = int(patch)
        self.length = int(length)
        self.cfg = cfg or DegradeConfig()
        self.fixed_seed = fixed_seed
        self.training = training

        align = 16 * max(self.cfg.scale_choices)
        if self.patch % align != 0:
            raise ValueError(f"patch={self.patch} 必须是 {align} 的倍数")
        self._align = align

    def __len__(self) -> int:
        return self.length

    def _rng(self, idx: int) -> np.random.Generator:
        if self.fixed_seed is not None:
            return np.random.default_rng(self.fixed_seed * 1_000_003 + idx)
        return np.random.default_rng()

    def __getitem__(self, idx: int):
        rng = self._rng(idx)
        path = self.files[idx % len(self.files)]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        p = self.patch
        if h < p or w < p:
            img = cv2.resize(img, (max(w, p), max(h, p)), interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]

        # 随机裁，起点对齐到 4 像素，避免缩放时产生额外的亚像素相位差
        y = int(rng.integers(0, (h - p) // 4 + 1)) * 4
        x = int(rng.integers(0, (w - p) // 4 + 1)) * 4
        gt = img[y : y + p, x : x + p]

        degraded = degrade_image(gt, self.cfg, rng)
        return to_tensor(degraded), to_tensor(gt)


class PaiRedEvalDataset(Dataset):
    """评测用：整图 + 固定种子的退化，保证四组消融看到完全相同的输入。"""

    def __init__(
        self,
        hr_dir: str | Path,
        cfg: DegradeConfig | None = None,
        seed: int = 1234,
        crop: int | None = 512,
        max_images: int | None = None,
    ):
        super().__init__()
        self.files = sorted(Path(hr_dir).glob("*.png"))
        if max_images:
            self.files = self.files[:max_images]
        if not self.files:
            raise FileNotFoundError(f"{hr_dir} 下没有 png")
        self.cfg = cfg or DegradeConfig()
        self.seed = seed
        self.crop = crop

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        align = 16 * max(self.cfg.scale_choices)
        if self.crop:
            h, w = img.shape[:2]
            size = min(self.crop, h, w)
            cy, cx = h // 2, w // 2
            y0 = (cy - size // 2) // align * align
            x0 = (cx - size // 2) // align * align
            img = img[y0 : y0 + size, x0 : x0 + size]
        img = img[: (img.shape[0] // align) * align, : (img.shape[1] // align) * align]

        rng = np.random.default_rng(self.seed * 7919 + idx)
        deg = degrade_image(img, self.cfg, rng)
        return to_tensor(deg), to_tensor(img), str(path)

    @staticmethod
    def collate(batch):
        deps = torch.stack([b[0] for b in batch])
        gts = torch.stack([b[1] for b in batch])
        names = [b[2] for b in batch]
        return deps, gts, names


__all__ = ["PairedDegradeDataset", "PaiRedEvalDataset", "to_tensor"]
