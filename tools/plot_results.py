"""出图：退化/各组复原的并排对比、局部放大、训练曲线。

用法：
    python tools/plot_results.py                       # 用 experiments/exp_*/best.pth
    python tools/plot_results.py --img path/to.png --exps A B C D
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.degrade.dct_degrade import DegradeConfig, degrade_image  # noqa: E402
from src.models.srvgg import build_model  # noqa: E402

DIV2K = ROOT / "SR_Project" / "dataset"


def read(path: Path) -> np.ndarray:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)[..., ::-1]


def label(img: np.ndarray, text: str, scale: float = 0.7) -> np.ndarray:
    out = img.copy()
    h = int(22 * scale / 0.7)
    cv2.rectangle(out, (0, 0), (out.shape[1], h), (0, 0, 0), -1)
    cv2.putText(out, text, (6, int(h * 0.75)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5 * scale / 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def infer(model, deg_u8: np.ndarray) -> np.ndarray:
    dev = next(model.parameters()).device
    x = torch.from_numpy(deg_u8.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        y = model(x.to(dev)).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()
    return (y * 255).round().clip(0, 255).astype(np.uint8)


def plot_compare(img_path: Path, exps: list[str], out_dir: Path, crop: int = 512) -> str:
    gt = read(img_path)
    align = 32
    gt = gt[:(gt.shape[0] // align) * align, :(gt.shape[1] // align) * align]

    h, w = gt.shape[:2]
    size = min(crop, h, w)
    y0 = ((h - size) // 2) // align * align
    x0 = ((w - size) // 2) // align * align
    gt = gt[y0 : y0 + size, x0 : x0 + size]

    rng = np.random.default_rng(2024)
    deg = degrade_image(gt, DegradeConfig(scale_choices=(2,)), rng)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    panels = [("GT", gt), ("传输退化", deg)]
    for exp in exps:
        ck = ROOT / "experiments" / f"exp_{exp}" / "best.pth"
        if not ck.exists():
            print(f"[skip] {ck} 不存在")
            continue
        model = build_model(scale=1).to(device)
        sd = torch.load(ck, map_location="cpu")
        model.load_state_dict(sd.get("params", sd))
        model.eval()
        panels.append((f"{exp} (it {sd.get('iter', '?')})", infer(model, deg)))

    gap = np.full((size, 8, 3), 255, dtype=np.uint8)
    row = []
    for i, (name, im) in enumerate(panels):
        if i:
            row.append(gap)
        row.append(label(im, name))
    canvas = np.hstack(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    p1 = out_dir / "compare_full.png"
    cv2.imwrite(str(p1), canvas[..., ::-1])

    # 局部放大：取中心 1/4 区域，2 倍放大看块效应
    z = size // 2
    zy, zx = (size - z) // 2 // 8 * 8, (size - z) // 2 // 8 * 8
    zooms = []
    for i, (name, im) in enumerate(panels):
        if i:
            zooms.append(np.full((z * 2, 8, 3), 255, dtype=np.uint8))
        zooms.append(label(cv2.resize(im[zy : zy + z, zx : zx + z], None, fx=2, fy=2,
                                      interpolation=cv2.INTER_NEAREST), name))
    p2 = out_dir / "compare_zoom.png"
    cv2.imwrite(str(p2), np.hstack(zooms)[..., ::-1])
    print(f"[ok] {p1}\n[ok] {p2}")
    return str(p1)


def plot_curves(root: Path, exps: list[str], out_dir: Path) -> str | None:
    """训练曲线：验证 PSNR 与块效应随 iter 的变化。"""
    series = {}
    for exp in exps:
        log = root / f"exp_{exp}" / "train_log.jsonl"
        if not log.exists():
            continue
        pts = []
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "psnr" in r:
                pts.append((r["iter"], r["psnr"], r.get("blockiness_pred")))
        if pts:
            series[exp] = pts
    if not series:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for exp, pts in series.items():
        it = [p[0] for p in pts]
        axes[0].plot(it, [p[1] for p in pts], marker="o", ms=3, label=exp)
        axes[1].plot(it, [p[2] for p in pts], marker="o", ms=3, label=exp)
    axes[0].set_title("val PSNR (dB)"); axes[0].set_xlabel("iter"); axes[0].grid(alpha=.3)
    axes[1].set_title("val blockiness (lower better)"); axes[1].set_xlabel("iter"); axes[1].grid(alpha=.3)
    for a in axes:
        a.legend()
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "train_curves.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"[ok] {p}")
    return str(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=str, default=None,
                    help="测试图，默认自动挑选纹理最丰富的一张")
    ap.add_argument("--exps", nargs="*", default=["A", "B", "C", "D"])
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--out", type=str, default=str(ROOT / "experiments" / "figures"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    img = Path(args.img) if args.img else None
    if img is None:
        # 挑一张细节多的验证图
        best, best_score = None, -1.0
        for f in sorted((DIV2K / "DIV2K_valid_HR").glob("*.png"))[:20]:
            im = cv2.imread(str(f), cv2.IMREAD_REDUCED_COLOR_4)
            score = float(np.abs(np.diff(im.astype(np.float32), axis=1)).mean())
            if score > best_score:
                best, best_score = f, score
        img = best
    print(f"[info] 测试图 {img}")
    plot_compare(img, args.exps, out_dir, args.crop)
    plot_curves(ROOT / "experiments", args.exps, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
