"""图像质量度量（numpy 实现，用于评测与数据核验）。"""

from __future__ import annotations

import numpy as np

BLOCK = 8
# 块效应度量的分母底噪（单位：灰度级）。平坦区内部梯度趋近 0 时的保护项。
BLOCKINESS_EPS = 0.5


def blockiness(img: np.ndarray, block: int = BLOCK) -> float:
    """块效应强度 = 块边界梯度均值 / (块内部梯度均值 + EPS)。

    img : (H,W) 或 (H,W,3)，0~255

    分母加 EPS=0.5（半个灰度级）作底噪保护：平坦区内部梯度趋近 0，
    不设下限会让比值爆炸成无意义的百万量级。加上后该量在整幅自然
    图像上约 0.5，强块效应可到 3 以上。
    """
    x = np.asarray(img, dtype=np.float64)
    if x.ndim == 3:
        x = x.mean(axis=2)
    h, w = x.shape
    h -= h % block
    w -= w % block
    x = x[:h, :w]

    # 竖直方向：跨块边界的列差 vs 块内部列差
    dh = np.abs(np.diff(x, axis=1))  # (h, w-1)，第 j 列 = x[:,j+1]-x[:,j]
    v_inner = np.delete(dh, np.arange(block - 1, w - 1, block), axis=1)
    v_bound = dh[:, block - 1 :: block]

    dv = np.abs(np.diff(x, axis=0))
    h_inner = np.delete(dv, np.arange(block - 1, h - 1, block), axis=0)
    h_bound = dv[block - 1 :: block, :]

    inner = (v_inner.mean() + h_inner.mean()) / 2.0
    bound = (v_bound.mean() + h_bound.mean()) / 2.0
    return float(bound / (inner + BLOCKINESS_EPS))


def blockiness_map(img: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """逐块块效应强度图 (nh,nw)，便于看空间分布（块边界响应 / 块内响应）。"""
    x = np.asarray(img, dtype=np.float64)
    if x.ndim == 3:
        x = x.mean(axis=2)
    h, w = x.shape
    h -= h % block
    w -= w % block
    x = x[:h, :w]
    nh, nw = h // block, w // block

    # 块内响应：每块内部相邻像素差的均值
    blocks = x.reshape(nh, block, nw, block).transpose(0, 2, 1, 3)  # (nh,nw,b,b)
    gx = np.abs(np.diff(blocks, axis=3)).mean(axis=(2, 3))
    gy = np.abs(np.diff(blocks, axis=2)).mean(axis=(2, 3))
    inner = (gx + gy) / 2.0

    # 边界响应：本块末列/末行 与 下一块首列/首行 的差，按块内行/列求均值
    def _to_grid(diff_map: np.ndarray, axis: int) -> np.ndarray:
        # diff_map: (h, nw-1) 或 (nh-1, w) -> 各自网格
        r, c = diff_map.shape
        if axis == 1:
            return diff_map.reshape(r // block, block, c).mean(axis=1)
        return diff_map.reshape(r, c // block, block).mean(axis=2)

    right = np.zeros((nh, nw))
    dr = np.abs(x[:, block - 1 :: block][:, :-1] - x[:, block::block])
    right[:, : nw - 1] = _to_grid(dr, 1)

    bottom = np.zeros((nh, nw))
    dbm = np.abs(x[block - 1 :: block, :][:-1, :] - x[block::block, :])
    bottom[: nh - 1, :] = _to_grid(dbm, 0)

    boundary = (right + bottom) / 2.0
    return boundary / (inner + BLOCKINESS_EPS)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean((a - b) ** 2))


def psnr(a: np.ndarray, b: np.ndarray, peak: float = 255.0) -> float:
    m = mse(a, b)
    if m <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(peak * peak / m))


def ssim(a: np.ndarray, b: np.ndarray, peak: float = 255.0) -> float:
    """简化的全局 SSIM（11x11 高斯窗，skimage 不可用时的替代实现）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim == 3:
        vals = [ssim(a[..., c], b[..., c], peak) for c in range(a.shape[2])]
        return float(np.mean(vals))

    c1 = (0.01 * peak) ** 2
    c2 = (0.03 * peak) ** 2
    win = 11
    k = np.exp(-((np.arange(win) - win // 2) ** 2) / (2 * 1.5**2))
    k = k / k.sum()
    ker = np.outer(k, k)

    def filt(im):
        from numpy.lib.stride_tricks import sliding_window_view

        pad = win // 2
        p = np.pad(im, pad, mode="reflect")
        windows = sliding_window_view(p, (win, win))
        return np.einsum("ijkl,kl->ij", windows, ker)

    mu_a, mu_b = filt(a), filt(b)
    saa = filt(a * a) - mu_a**2
    sbb = filt(b * b) - mu_b**2
    sab = filt(a * b) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (saa + sbb + c2)
    )
    return float(s.mean())


def gradient_energy(img: np.ndarray) -> float:
    """全图平均梯度能量，用于衡量纹理保留程度。"""
    x = np.asarray(img, dtype=np.float64)
    if x.ndim == 3:
        x = x.mean(axis=2)
    gx = np.diff(x, axis=1)
    gy = np.diff(x, axis=0)
    return float((np.mean(gx**2) + np.mean(gy**2)) / 2.0)


__all__ = [
    "blockiness",
    "blockiness_map",
    "mse",
    "psnr",
    "ssim",
    "gradient_energy",
]
