"""退化侧冒烟检查：确认链路式退化真的产生了「多代 + 网格相位漂移」。

不训练、不需要 GPU，纯 numpy/cv2，几秒钟出结果。

检查三件事：
  1. link 配置下每张图真的抽到 1~5 代，且记录了每代相位
  2. 相位确实不是全 (0,0)
  3. 用「块边界梯度谱」把图像里的网格相位测出来——如果同时存在 2 套以上
     相位，说明网格错位是真实发生的，不是我们的想象

用法：
    python tools/check_link_degrade.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.degrade.dct_degrade import BLOCK, DegradeConfig, degrade_image  # noqa: E402


def load_sample(size: int = 256) -> np.ndarray:
    """优先用系统壁纸；找不到就用合成图，保证脚本到处能跑。"""
    for p in Path("C:/Windows/Web/Wallpaper").rglob("*.jpg"):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        y, x = max(0, h // 2 - size // 2), max(0, w // 2 - size // 2)
        crop = img[y : y + size, x : x + size]
        if crop.shape[0] == size and crop.shape[1] == size:
            return crop
    rng = np.random.default_rng(0)
    base = rng.integers(40, 210, (size, size, 3), dtype=np.uint8)
    return cv2.GaussianBlur(base, (0, 0), 6)


def phase_contrast(gray: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """每个相位的「边界能量 / 全图平均边界能量」。

    注意不能跨样本取平均再比较——不同图的强相位不同，平均会把信号抹平。
    必须逐样本算，再看强相位分布。
    """
    g = gray.astype(np.float32)
    dx = np.abs(np.diff(g, axis=1)).mean(axis=0)  # (W-1,)
    dy = np.abs(np.diff(g, axis=0)).mean(axis=1)  # (H-1,)
    out = np.empty(block, dtype=np.float32)
    for p in range(block):
        col = dx[p::block].mean() if dx[p::block].size else 0.0
        row = dy[p::block].mean() if dy[p::block].size else 0.0
        out[p] = 0.5 * (col + row)
    return out / max(float(out.mean()), 1e-6)


def summarize(name: str, contrasts: np.ndarray) -> None:
    """contrasts: (n_sample, 8)。打印强相位分布与多相位现象占比。"""
    top = contrasts.argmax(axis=1)
    hist = np.bincount(top, minlength=BLOCK)
    # 有 2 个及以上相位的边界能量都明显高于平均 => 图里同时存在多套网格
    multi = (contrasts > 1.05).sum(axis=1)
    print(f"\n[{name}] 逐样本最强相位分布 {dict(enumerate(hist.tolist()))}")
    print(f"[{name}] 强相位个数的均值 {multi.mean():.2f}（≥2 说明存在多套网格）")
    print(f"[{name}] 满足多网格的样本占比 {float((multi >= 2).mean()):.2f}")
    print(f"[{name}] 相位对比度极差均值 {(contrasts.max(1) - contrasts.min(1)).mean():.3f}")
    print(
        f"[{name}] 平均相位曲线 " + " ".join(f"{v:.2f}" for v in contrasts.mean(axis=0))
    )


def main() -> int:
    img = load_sample()
    print(f"样本 {img.shape}")

    link = DegradeConfig(
        scale_choices=(1,),
        uniform_quality=False,
        n_gen_range=(1, 5),
        phase_drift=True,
        resize_gen_prob=0.5,
    )
    official = DegradeConfig(
        scale_choices=(1,),
        uniform_quality=True,
        uniform_q_range=(30.0, 90.0),
        n_gen_range=(2, 2),
        phase_drift=False,
        resize_gen_prob=0.0,
    )

    rng = np.random.default_rng(7)
    gens, all_phases, c_link = [], [], []
    for _ in range(16):
        deg, info = degrade_image(img, link, rng, return_info=True)
        gens.append(info["n_gen"])
        all_phases.extend(info["phases"])
        c_link.append(phase_contrast(cv2.cvtColor(deg, cv2.COLOR_RGB2GRAY)))

    print(f"\n[link] 代次分布 {sorted(set(gens))} 均值 {np.mean(gens):.2f}")
    ph = np.asarray(all_phases)
    print(f"[link] 相位非零占比 {float((ph.sum(1) > 0).mean()):.2f}")
    print(f"[link] 相位覆盖 r={sorted(set(ph[:,0].tolist()))} c={sorted(set(ph[:,1].tolist()))}")
    summarize("link", np.vstack(c_link))

    rng = np.random.default_rng(7)
    c_off = []
    for _ in range(16):
        deg, _info = degrade_image(img, official, rng, return_info=True)
        c_off.append(phase_contrast(cv2.cvtColor(deg, cv2.COLOR_RGB2GRAY)))
    summarize("official", np.vstack(c_off))

    print("\n判读：official 每代网格都对齐在左上角，强相位应集中在同一个位置；")
    print("link 每代原点随机，强相位应分散，且同时满足多网格的样本占比更高。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
