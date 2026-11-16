"""传输链路退化流水线（创新一：内容依赖的非均匀退化建模）。

不使用 libjpeg：libjpeg 的 quality 是图像级参数，DQT 表全图唯一，无法做到
"平坦块压得狠、纹理块压得轻"。这里用纯 numpy 走一遍 8x8 DCT，逐块分配量化步长。

链路顺序（对应真实传图的四个环节）：
    ① 编码压缩（非均匀 DCT + 4:2:0）
    ② 降分辨率（长边缩到 1080 档）
    ③ 显示端插值放大
    ④ 二次转码（均匀压缩）

方向约定（写反了整个创新失效）：
    E 大(纹理) -> quality 高 -> 量化步长 Q 小 -> 保细节
    E 小(平坦) -> quality 低 -> 量化步长 Q 大 -> 省比特，块效应外露

术语：
    q  : JPEG 质量因子，0~100，越大越好
    Q  : 量化步长矩阵 8x8，越大压得越狠
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import cv2
import numpy as np

# --------------------------------------------------------------------------
# 8x8 DCT-II 变换矩阵（正交归一化）
# --------------------------------------------------------------------------
BLOCK = 8


def _dct_matrix(n: int = BLOCK) -> np.ndarray:
    k = np.arange(n)[:, None]
    x = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * x + 1) * k / (2 * n))
    m[0, :] *= np.sqrt(1.0 / n)
    m[1:, :] *= np.sqrt(2.0 / n)
    return m.astype(np.float32)


D8 = _dct_matrix()
D8T = D8.T.copy()

# JPEG 标准量化表（Q50 基准）
Q50_LUMA = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float32,
)

Q50_CHROMA = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=np.float32,
)


# --------------------------------------------------------------------------
# 量化表
# --------------------------------------------------------------------------
def quality_to_scale(q: np.ndarray | float) -> np.ndarray:
    """JPEG 质量 -> 量化表的缩放系数 S（经典分段公式）。"""
    q = np.clip(np.asarray(q, dtype=np.float32), 1.0, 100.0)
    return np.where(q < 50.0, 5000.0 / q, 200.0 - 2.0 * q)


def quant_table(q: np.ndarray | float, base: np.ndarray = Q50_LUMA) -> np.ndarray:
    """由逐块 quality 生成逐块量化步长矩阵。

    q : 标量或 (N,) 数组
    return: (8,8) 或 (N,8,8) 的量化步长，值域 [1, 255]
    """
    q = np.asarray(q, dtype=np.float32)
    s = quality_to_scale(q)  # 标量或 (N,)
    if s.ndim == 0:
        t = np.floor((base * float(s) + 50.0) / 100.0)
    else:
        t = np.floor((base[None] * s[:, None, None] + 50.0) / 100.0)
    return np.clip(t, 1.0, 255.0)


# --------------------------------------------------------------------------
# 分块工具
# --------------------------------------------------------------------------
def _to_blocks(img: np.ndarray, b: int = BLOCK) -> np.ndarray:
    """(H,W) -> (N,b,b)，行优先分块。要求 H、W 是 b 的整数倍。"""
    h, w = img.shape
    assert h % b == 0 and w % b == 0, f"尺寸 {img.shape} 非 {b} 的整数倍"
    x = img.reshape(h // b, b, w // b, b)
    return x.transpose(0, 2, 1, 3).reshape(-1, b, b)


def _from_blocks(blocks: np.ndarray, h: int, w: int, b: int = BLOCK) -> np.ndarray:
    """(N,b,b) -> (H,W)"""
    nh, nw = h // b, w // b
    x = blocks.reshape(nh, nw, b, b).transpose(0, 2, 1, 3)
    return x.reshape(h, w)


def _pad_to(img: np.ndarray, m: int) -> tuple[np.ndarray, tuple[int, int]]:
    """把 H,W 反射填充到 m 的整数倍，返回填充后的图与其 (H,W) 尺寸。"""
    h, w = img.shape[:2]
    ph = (-h) % m
    pw = (-w) % m
    if ph == 0 and pw == 0:
        return img, (h, w)
    pad = ((0, ph), (0, pw)) if img.ndim == 2 else ((0, ph), (0, pw), (0, 0))
    out = np.pad(img, pad, mode="reflect")
    return out, out.shape[:2]


# --------------------------------------------------------------------------
# DCT / IDCT
# --------------------------------------------------------------------------
def dct2_blocks(blocks: np.ndarray) -> np.ndarray:
    """(N,8,8) 空域 -> (N,8,8) DCT 域"""
    return D8 @ blocks @ D8T


def idct2_blocks(coef: np.ndarray) -> np.ndarray:
    """(N,8,8) DCT 域 -> (N,8,8) 空域"""
    return D8T @ coef @ D8


# --------------------------------------------------------------------------
# 纹理能量与逐块 quality 分配
# --------------------------------------------------------------------------
def block_energy(channel: np.ndarray, b: int = BLOCK) -> np.ndarray:
    """逐块局部纹理能量 E（块内梯度平方均值），返回 (N,)。

    用像素域梯度而不是 DCT 交流系数：编码器的 RDO 也是在像素域估误差，
    且这样能量与是否压缩无关，可作为"内容属性"单独统计。
    """
    img, _ = _pad_to(channel.astype(np.float32), b)
    dx = np.zeros_like(img)
    dy = np.zeros_like(img)
    dx[:, 1:] = img[:, 1:] - img[:, :-1]
    dy[1:, :] = img[1:, :] - img[:-1, :]
    g = dx * dx + dy * dy
    blocks = _to_blocks(g, b)
    return blocks.mean(axis=(1, 2))


def energy_to_quality(
    energy: np.ndarray,
    q_lo: float = 20.0,
    q_hi: float = 88.0,
    pct: tuple[float, float] = (3.0, 97.0),
    gamma: float = 1.0,
) -> np.ndarray:
    """局部能量 -> 逐块 quality。E 大 -> q 大（量化步长小）。

    用对数域 + 百分位归一化，使分配比例与图像整体反差无关，便于跨图稳定。
    """
    e = np.log1p(energy)
    lo, hi = np.percentile(e, pct[0]), np.percentile(e, pct[1])
    t = np.clip((e - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    t = t ** float(gamma)
    return q_lo + (q_hi - q_lo) * t


# --------------------------------------------------------------------------
# 色彩空间
# --------------------------------------------------------------------------
def rgb_to_ycbcr(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
    return y, cb, cr


def ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    r = y + 1.402 * (cr - 128.0)
    g = y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0)
    b = y + 1.772 * (cb - 128.0)
    return np.stack([r, g, b], axis=-1)


# --------------------------------------------------------------------------
# 单通道非均匀量化
# --------------------------------------------------------------------------
def _quantize_channel(
    channel: np.ndarray,
    block_q: np.ndarray,
    base_table: np.ndarray,
    return_stats: bool = False,
):
    """按逐块 quality 对单通道做 DCT 量化-反量化。

    return_stats=True 时额外返回码率代理指标：
        bpp           —— 量化系数熵上界（bit/pixel），近似实际比特开销
        nonzero_ratio —— 非零量化系数占比
    """
    h, w = channel.shape
    blocks = _to_blocks(channel.astype(np.float32), BLOCK)
    coef = dct2_blocks(blocks)
    qmat = quant_table(block_q, base_table)  # (N,8,8)
    coef_q = np.round(coef / qmat)
    rec = idct2_blocks(coef_q * qmat)
    out = _from_blocks(rec, h, w, BLOCK)
    if not return_stats:
        return out
    stats = {
        "bpp": float(np.log2(1.0 + np.abs(coef_q)).sum() / (h * w)),
        "nonzero_ratio": float((coef_q != 0).mean()),
    }
    return out, stats


def _uniform_quality(n_blocks: int, q: float, jitter: float = 0.0, rng=None) -> np.ndarray:
    q_arr = np.full(n_blocks, float(q), dtype=np.float32)
    if jitter > 0 and rng is not None:
        q_arr += rng.normal(0.0, jitter, size=n_blocks).astype(np.float32)
        q_arr = np.clip(q_arr, 1.0, 100.0)
    return q_arr


# --------------------------------------------------------------------------
# 退化配置与主入口
# --------------------------------------------------------------------------
@dataclass
class DegradeConfig:
    """传输链路退化参数。所有环节都可单独开关，用于消融。

    链路顺序（与真实传图一致：先缩到 1080 档，再编码上传，最后显示端放大）：
        缩放 -> 编码压缩 -> 二次转码 -> 插值放大
    注意：缩放必须是**整数倍**，否则放大后 8x8 块间距不再是整数，
    块效应无法用固定网格度量，训练时的 BlockinessLoss 也会失准。
    """

    # ① 编码压缩
    q_lo: float = 42.0            # 平坦块质量下限
    q_hi: float = 78.0            # 纹理块质量上限
    energy_pct: tuple = (3.0, 97.0)
    energy_gamma: float = 1.0
    quality_jitter: float = 3.0   # 模拟 RDO 的块级抖动
    chroma_420: bool = True
    chroma_quality_ratio: float = 0.72  # 色度相对亮度的质量比例

    # 是否用均匀分配（False=创新一的做法，True=Real-ESRGAN 式全图统一）
    uniform_quality: bool = False
    uniform_q_range: tuple = (22.0, 60.0)

    # ② 降分辨率：整数倍
    scale_choices: tuple = (2,)   # 训练时可 (1, 2, 3)，评测固定

    # ③ 插值放大
    up_mode: str = "random"        # cubic / bilinear / random
    do_up_blur: bool = True        # 放大后轻微高斯模糊（显示端 resample）

    # ④ 二次转码
    do_second: bool = True
    second_q_range: tuple = (58.0, 72.0)

    # ------------------------------------------------------------------
    # 链路式多代退化（创新一的空间非均匀版）
    # n_gen_range=None 时走上面的原有两段式，保证老实验可复现；
    # 设成 (1,5) 这类区间则启用"每代独立参数 + 网格相位漂移"。
    # ------------------------------------------------------------------
    n_gen_range: tuple | None = None   # 每样本随机抽取代次
    phase_drift: bool = False          # 每代网格原点随机偏移 (r,c)∈[0,8)
    resize_gen_prob: float = 0.0       # 每代以该概率"降采样→回放大"，模拟平台改尺寸
    resize_gen_range: tuple = (0.60, 0.95)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _encode_once(
    x: np.ndarray,
    block_q: np.ndarray,
    chroma_ratio: float,
    chroma_420: bool,
    rng,
    quality_jitter: float = 0.0,
) -> np.ndarray:
    """对一张 float32 RGB 图（尺寸必须是 16 的倍数）做一次 DCT 编解码。"""
    hp, wp = x.shape[:2]
    y, cb, cr = rgb_to_ycbcr(x)
    y_rec, st = _quantize_channel(y, block_q, Q50_LUMA, return_stats=True)

    q_c = float(np.mean(block_q)) * chroma_ratio
    if chroma_420:
        cb_s = cv2.resize(cb, (wp // 2, hp // 2), interpolation=cv2.INTER_AREA)
        cr_s = cv2.resize(cr, (wp // 2, hp // 2), interpolation=cv2.INTER_AREA)
        n_c = (cb_s.shape[0] // BLOCK) * (cb_s.shape[1] // BLOCK)
        qc = _uniform_quality(n_c, q_c, quality_jitter, rng)
        cb_r = _quantize_channel(cb_s, qc, Q50_CHROMA)
        cr_r = _quantize_channel(cr_s, qc, Q50_CHROMA)
        cb_r = cv2.resize(cb_r, (wp, hp), interpolation=cv2.INTER_LINEAR)
        cr_r = cv2.resize(cr_r, (wp, hp), interpolation=cv2.INTER_LINEAR)
    else:
        n_c = (hp // BLOCK) * (wp // BLOCK)
        qc = _uniform_quality(n_c, q_c, quality_jitter, rng)
        cb_r = _quantize_channel(cb, qc, Q50_CHROMA)
        cr_r = _quantize_channel(cr, qc, Q50_CHROMA)

    out = np.clip(ycbcr_to_rgb(y_rec, cb_r, cr_r), 0, 255)
    return out, st


def _resize_roundtrip(x: np.ndarray, factor: float) -> np.ndarray:
    """降采样再放大回原尺寸：模拟平台改分辨率，尺寸不变但引入重采样模糊。"""
    h, w = x.shape[:2]
    nh = max(16, int(round(h * factor)) // 16 * 16)
    nw = max(16, int(round(w * factor)) // 16 * 16)
    y = cv2.resize(x, (nw, nh), interpolation=cv2.INTER_AREA)
    return cv2.resize(y, (w, h), interpolation=cv2.INTER_CUBIC)


def _gen_block_q(x: np.ndarray, cfg: "DegradeConfig", rng):
    """按当前代次的图像内容生成本代的逐块 quality；(N,8,8) 与标量均值。"""
    h, w = x.shape[:2]
    n_blk = (h // BLOCK) * (w // BLOCK)
    if cfg.uniform_quality:
        q = float(rng.uniform(*cfg.uniform_q_range))
        return _uniform_quality(n_blk, q, cfg.quality_jitter, rng), q
    e = block_energy(rgb_to_ycbcr(x)[0], BLOCK)
    bq = energy_to_quality(e, cfg.q_lo, cfg.q_hi, cfg.energy_pct, cfg.energy_gamma)
    if cfg.quality_jitter > 0:
        bq = np.clip(
            bq + rng.normal(0.0, cfg.quality_jitter, size=bq.shape), 1.0, 100.0
        ).astype(np.float32)
    return bq, float(np.mean(bq))


def _encode_gen(x: np.ndarray, cfg: "DegradeConfig", rng, phase: tuple[int, int]):
    """带网格相位偏移的一代编码。

    phase=(r,c)：先在左上补 r 行 c 列，编码后再裁回，使本代 8x8 网格原点
    相对图像偏移 (8-r, 8-c) mod 8。多代累积后图里会同时存在多套相位。
    """
    r, c = int(phase[0]), int(phase[1])
    h, w = x.shape[:2]
    xp = np.pad(x, ((r, 0), (c, 0), (0, 0)), mode="symmetric") if (r or c) else x
    ph, pw = (-xp.shape[0]) % 16, (-xp.shape[1]) % 16
    if ph or pw:
        xp = np.pad(xp, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    block_q, q_mean = _gen_block_q(xp, cfg, rng)
    out, st = _encode_once(
        xp, block_q, cfg.chroma_quality_ratio, cfg.chroma_420, rng, cfg.quality_jitter
    )
    return np.ascontiguousarray(out[r : r + h, c : c + w]), st, q_mean


def degrade_image(
    img: np.ndarray,
    cfg: DegradeConfig | None = None,
    rng: np.random.Generator | None = None,
    return_info: bool = False,
):
    """对一张 uint8 RGB 图执行完整传输链路退化。

    img : (H,W,3) uint8
    return: 退化图 (H,W,3) uint8；return_info=True 时额外返回 info dict
        info["scale"]      整数倍下采样倍数
        info["block_size"] 退化图上 8x8 块的等效像素间距（= 8*scale）
        info["quality_map"] 编码阶段的逐块 quality，网格 (h/scale/8, w/scale/8)
    """
    cfg = cfg or DegradeConfig()
    rng = rng or np.random.default_rng()
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"期望 (H,W,3) RGB，得到 {img.shape}")

    h0, w0 = img.shape[:2]
    x = img.astype(np.float32)

    # ---------- 第 0 步：降分辨率（整数倍） ----------
    scale = int(rng.choice(list(cfg.scale_choices)))
    m = 16 * scale  # 保证缩放后仍是 16 的倍数（8x8 块 + 4:2:0）
    xp, (hp, wp) = _pad_to(x, m)
    if scale > 1:
        small = cv2.resize(
            xp, (wp // scale, hp // scale), interpolation=cv2.INTER_AREA
        )
    else:
        small = xp

    hs, ws = small.shape[:2]
    n_blk = (hs // BLOCK) * (ws // BLOCK)
    info: dict[str, Any] = {
        "config": cfg.to_dict(),
        "scale": scale,
        "block_size": BLOCK * scale,
    }

    if cfg.n_gen_range is None:
        # ================= 原有两段式（保证老实验可复现） =================
        # ---------- 第 1 步：编码压缩（非均匀 DCT） ----------
        if cfg.uniform_quality:
            q_img = float(rng.uniform(*cfg.uniform_q_range))
            block_q = _uniform_quality(n_blk, q_img, cfg.quality_jitter, rng)
        else:
            e = block_energy(rgb_to_ycbcr(small)[0], BLOCK)
            block_q = energy_to_quality(
                e, cfg.q_lo, cfg.q_hi, cfg.energy_pct, cfg.energy_gamma
            )
            if cfg.quality_jitter > 0:
                block_q = np.clip(
                    block_q + rng.normal(0.0, cfg.quality_jitter, size=block_q.shape),
                    1.0,
                    100.0,
                ).astype(np.float32)

        small, st1 = _encode_once(
            small, block_q, cfg.chroma_quality_ratio, cfg.chroma_420, rng, cfg.quality_jitter
        )

        info.update(
            {
                "quality_map": block_q.reshape(hs // BLOCK, ws // BLOCK),
                "quality_lo": float(np.percentile(block_q, 5)),
                "quality_hi": float(np.percentile(block_q, 95)),
                "quality_mean": float(np.mean(block_q)),
                "bpp_pass1": st1["bpp"],
                "nonzero_ratio": st1["nonzero_ratio"],
                "bpp_pass2": None,
                "second_q": None,
            }
        )

        # ---------- 第 2 步：二次转码（均匀质量） ----------
        if cfg.do_second:
            q2 = float(rng.uniform(*cfg.second_q_range))
            info["second_q"] = q2
            q_arr2 = _uniform_quality(n_blk, q2, 0.0, rng)
            small, st2 = _encode_once(
                small, q_arr2, cfg.chroma_quality_ratio, cfg.chroma_420, rng
            )
            info["bpp_pass2"] = st2["bpp"]
    else:
        # ============ 链路式：N 代非同质重编码 + 网格相位漂移 ============
        lo, hi = int(cfg.n_gen_range[0]), int(cfg.n_gen_range[1])
        n_gen = int(rng.integers(lo, hi + 1))
        phases: list[list[int]] = []
        gen_q: list[float] = []
        gen_bpp: list[float] = []
        for _ in range(n_gen):
            # 平台改尺寸：每代都有一定概率经历一次"降采样→回放大"
            if cfg.resize_gen_prob > 0.0 and rng.random() < cfg.resize_gen_prob:
                small = _resize_roundtrip(small, float(rng.uniform(*cfg.resize_gen_range)))
            phase = (
                (int(rng.integers(0, BLOCK)), int(rng.integers(0, BLOCK)))
                if cfg.phase_drift
                else (0, 0)
            )
            small, st, qm = _encode_gen(small, cfg, rng, phase)
            phases.append([int(phase[0]), int(phase[1])])
            gen_q.append(qm)
            gen_bpp.append(float(st["bpp"]))

        info.update(
            {
                "n_gen": n_gen,
                "phases": phases,
                "gen_quality": gen_q,
                "gen_bpp": gen_bpp,
                "quality_map": None,
                "quality_lo": float(min(gen_q)),
                "quality_hi": float(max(gen_q)),
                "quality_mean": float(np.mean(gen_q)),
                "bpp_pass1": gen_bpp[0] if gen_bpp else None,
                "nonzero_ratio": None,
                "bpp_pass2": gen_bpp[1] if len(gen_bpp) > 1 else None,
                "second_q": None,
            }
        )

    # ---------- 第 3 步：显示端插值放大 ----------
    if scale > 1:
        mode = cfg.up_mode
        if mode == "random":
            mode = "cubic" if rng.random() < 0.7 else "bilinear"
        interp = cv2.INTER_CUBIC if mode == "cubic" else cv2.INTER_LINEAR
        small = cv2.resize(small, (wp, hp), interpolation=interp)
        if cfg.do_up_blur:
            small = cv2.GaussianBlur(small, (0, 0), sigmaX=rng.uniform(0.2, 0.6))

    out = np.clip(np.round(small), 0, 255).astype(np.uint8)[:h0, :w0]
    if return_info:
        return out, info
    return out


def degrade_batch(
    imgs: list[np.ndarray],
    cfg: DegradeConfig | None = None,
    seed: int | None = None,
) -> list[np.ndarray]:
    """批量退化，每张图独立随机参数。"""
    rng = np.random.default_rng(seed)
    return [degrade_image(im, cfg, rng) for im in imgs]


__all__ = [
    "BLOCK",
    "DegradeConfig",
    "degrade_image",
    "degrade_batch",
    "block_energy",
    "energy_to_quality",
    "quant_table",
    "quality_to_scale",
    "dct2_blocks",
    "idct2_blocks",
]
