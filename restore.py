"""把一张传输退化的糊图复原成清晰图。

既可以直接当命令行用，也可以被 app.py 调用。

命令行：
    python restore.py 我的图.jpg                  # 输出到 outputs/
    python restore.py a.jpg b.png -o results/     # 批量
    python restore.py 我的图.jpg --tile 512       # 手动指定分块大小

Python：
    from restore import Restorer
    r = Restorer()
    out = r.restore(图像numpy数组)   # uint8 RGB 进，uint8 RGB 出，尺寸不变
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.models.srvgg import build_model  # noqa: E402

# 权重优先用仓库里的 checkpoints/best.pth；本地训练产物在 experiments/exp_*/ 下时也能兜住
CKPT_CANDIDATES = [
    ROOT / "checkpoints" / "best.pth",
    ROOT / "experiments" / "exp_G" / "best.pth",
]
DEFAULT_CKPT = next((p for p in CKPT_CANDIDATES if p.exists()), CKPT_CANDIDATES[0])
DEFAULT_TILE = 512
DEFAULT_OVERLAP = 48


class Restorer:
    """同分辨率复原器：输入糊图，输出同尺寸清晰图。

    大图自动分块（tiling），避免显存爆掉；块之间带重叠并用羽化权重融合，
    拼接处不会出现接缝。
    """

    def __init__(
        self,
        ckpt: str | Path = DEFAULT_CKPT,
        device: str | None = None,
        tile: int = DEFAULT_TILE,
        overlap: int = DEFAULT_OVERLAP,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = build_model(scale=1).to(self.device)
        if not Path(ckpt).is_file():
            raise FileNotFoundError(
                f"找不到模型权重: {ckpt}\n"
                "仓库自带权重在 checkpoints/best.pth，用默认参数即可；"
                "自己训练的可用 --ckpt 指到 experiments/exp_X/best.pth"
            )
        state = torch.load(str(ckpt), map_location="cpu")
        self.model.load_state_dict(state.get("params", state), strict=False)
        self.model.eval()
        self.tile = int(tile)
        self.overlap = int(min(overlap, tile // 4))

    # ---------------- 内部 ----------------
    def _forward(self, patch: np.ndarray) -> np.ndarray:
        """patch: float32 [0,1] (h,w,3) -> 同尺寸输出"""
        x = torch.from_numpy(patch.transpose(2, 0, 1)).float().unsqueeze(0)
        x = x.to(self.device)
        with torch.no_grad():
            y = self.model(x).clamp(0, 1)
        return y[0].permute(1, 2, 0).float().cpu().numpy()

    @staticmethod
    def _feather(n: int, ov: int) -> np.ndarray:
        """一维羽化权重：两端线性降到 0.01，中间为 1。"""
        w = np.ones(n, dtype=np.float32)
        if ov > 0:
            ramp = np.linspace(0.01, 1.0, ov, dtype=np.float32)
            w[:ov] = np.minimum(w[:ov], ramp)
            w[-ov:] = np.minimum(w[-ov:], ramp[::-1])
        return w

    def _starts(self, total: int) -> list[int]:
        t, ov = self.tile, self.overlap
        if total <= t:
            return [0]
        step = t - ov
        pos = list(range(0, total - t + 1, step))
        if pos[-1] != total - t:
            pos.append(total - t)
        return pos

    # ---------------- 对外 ----------------
    def restore(self, img_u8: np.ndarray, progress=None) -> np.ndarray:
        """img_u8: (H,W,3) uint8 RGB -> (H,W,3) uint8 RGB"""
        if img_u8.ndim == 2:
            img_u8 = np.stack([img_u8] * 3, axis=-1)
        if img_u8.shape[2] == 4:
            img_u8 = img_u8[:, :, :3]
        img = img_u8.astype(np.float32) / 255.0
        h, w = img.shape[:2]

        if h <= self.tile and w <= self.tile:
            out = self._forward(img)
        else:
            out = np.zeros((h, w, 3), np.float32)
            wsum = np.zeros((h, w, 1), np.float32)
            ys, xs = self._starts(h), self._starts(w)
            total = len(ys) * len(xs)
            done = 0
            fy = self._feather(self.tile, self.overlap)[:, None, None]
            fx = self._feather(self.tile, self.overlap)[None, :, None]
            for y in ys:
                for x in xs:
                    res = self._forward(img[y : y + self.tile, x : x + self.tile])
                    wt = fy * fx
                    out[y : y + self.tile, x : x + self.tile] += res * wt
                    wsum[y : y + self.tile, x : x + self.tile] += wt
                    done += 1
                    if progress is not None:
                        progress(done / total, f"分块 {done}/{total}")
            out = out / np.maximum(wsum, 1e-6)

        return (np.clip(out, 0, 1) * 255).round().astype(np.uint8)


def load_image(path: str | Path) -> np.ndarray:
    """读图，处理中文路径与 EXIF 旋转。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"读不出这张图: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(path: str | Path, img_rgb: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ext = path.suffix or ".png"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), 95] if ext.lower() in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, bgr, params)
    if not ok:
        raise RuntimeError(f"写不出去: {path}")
    buf.tofile(str(path))


def main() -> int:
    ap = argparse.ArgumentParser(description="传输退化图像复原")
    ap.add_argument("inputs", nargs="+", help="输入图片，可多张")
    ap.add_argument("-o", "--out", default="outputs", help="输出目录或文件")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--tile", type=int, default=DEFAULT_TILE, help="分块边长，显存小就调小")
    ap.add_argument("--max-side", type=int, default=4000, help="超长边先等比缩小")
    args = ap.parse_args()

    r = Restorer(args.ckpt, tile=args.tile)
    print(f"[model] {args.ckpt}  设备 {r.device}  分块 {r.tile}")

    out_arg = Path(args.out)
    single = len(args.inputs) == 1 and out_arg.suffix

    for src in args.inputs:
        img = load_image(src)
        h0, w0 = img.shape[:2]
        if max(h0, w0) > args.max_side:
            s = args.max_side / max(h0, w0)
            img = cv2.resize(img, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_AREA)
            print(f"[warn] 原图 {w0}x{h0} 过大，缩到 {img.shape[1]}x{img.shape[0]}")

        out = r.restore(img)
        dst = out_arg if single else out_arg / f"{Path(src).stem}_restored.png"
        save_image(dst, out)
        print(f"[ok] {src} -> {dst}  ({out.shape[1]}x{out.shape[0]})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
