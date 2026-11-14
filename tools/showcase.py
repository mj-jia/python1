"""生成一张给人看的对比图：原图 / 传输退化 / 各组复原。

用法：
    python tools/showcase.py                       # 自动挑场景
    python tools/showcase.py --img xx.png --exps A C D
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.degrade.dct_degrade import DegradeConfig, degrade_image  # noqa: E402
from src.models.srvgg import build_model  # noqa: E402

DIV2K = ROOT / "SR_Project" / "dataset" / "DIV2K_valid_HR"


def read(p: Path) -> np.ndarray:
    return cv2.imread(str(p), cv2.IMREAD_COLOR)[..., ::-1]


def put_label(img: np.ndarray, text: str, color=(255, 255, 255)) -> np.ndarray:
    out = img.copy()
    band = 26
    cv2.rectangle(out, (0, 0), (out.shape[1], band), (20, 20, 20), -1)
    cv2.putText(out, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return out


def infer(model, u8: np.ndarray, device) -> np.ndarray:
    x = torch.from_numpy(u8.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        y = model(x).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()
    return (y * 255).round().clip(0, 255).astype(np.uint8)


def pick_scenes(n: int = 2) -> list[Path]:
    """挑一张纹理多的 + 一张大片平坦区的（避开夜景：平均亮度过低的不选）。"""
    files = sorted(DIV2K.glob("*.png"))[:40]
    scored = []
    for f in files:
        im = cv2.imread(str(f), cv2.IMREAD_REDUCED_COLOR_4)
        if im is None:
            continue
        g = im.mean(axis=2)
        if g.mean() < 90 or g.mean() > 200:  # 夜景 / 过曝
            continue
        grad = np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=0)).mean()
        flat_ratio = float((np.abs(np.diff(g, axis=1)) < 2).mean())
        scored.append((f, grad, flat_ratio))
    if not scored:
        return sorted(DIV2K.glob("*.png"))[:n]
    scored.sort(key=lambda z: -z[1])
    tex = scored[0][0]
    flat = max(scored, key=lambda z: z[2])[0]
    out = [tex] if tex == flat else [tex, flat]
    return out[:n]


def load_model(exp: str, device):
    ck = ROOT / "experiments" / f"exp_{exp}" / "best.pth"
    if not ck.exists():
        return None
    m = build_model(scale=1).to(device)
    m.load_state_dict(torch.load(ck, map_location="cpu")["params"])
    m.eval()
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", nargs="*", default=None)
    ap.add_argument("--exps", nargs="*", default=["A", "C", "D"])
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--zoom", type=int, default=160)
    ap.add_argument("--out", type=str, default=str(ROOT / "experiments" / "figures" / "showcase.png"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for e in args.exps:
        m = load_model(e, device)
        if m is None:
            print(f"[skip] exp_{e} 无权重")
        else:
            models[e] = m

    imgs = [Path(p) for p in args.img] if args.img else pick_scenes(2)
    full_rows, zoom_rows = [], []
    s = args.size
    z = args.zoom

    for path in imgs:
        gt = read(path)
        gt = gt[:(gt.shape[0] // 32) * 32, :(gt.shape[1] // 32) * 32]
        h, w = gt.shape[:2]
        gt = gt[(h - s) // 2 // 32 * 32 : (h - s) // 2 // 32 * 32 + s,
                (w - s) // 2 // 32 * 32 : (w - s) // 2 // 32 * 32 + s]
        deg = degrade_image(gt, DegradeConfig(scale_choices=(2,)), np.random.default_rng(2024))

        cells = [("GT", gt, (190, 255, 190)), ("Degraded", deg, (255, 205, 150))]
        for e, m in models.items():
            cells.append((e, infer(m, deg, device), (255, 255, 255)))

        zy = (s - z) // 2 // 8 * 8
        zx = (s - z) // 2 // 8 * 8
        strip, zstrip = [], []
        for i, (name, im, col) in enumerate(cells):
            if i:
                strip.append(np.full((s, 6, 3), 255, dtype=np.uint8))
                zstrip.append(np.full((z * 3, 6, 3), 255, dtype=np.uint8))
            strip.append(put_label(im, name, col))
            crop = cv2.resize(im[zy : zy + z, zx : zx + z], (z * 3, z * 3),
                              interpolation=cv2.INTER_NEAREST)
            zstrip.append(put_label(crop, name, col))

        full_rows.append(np.hstack(strip))
        full_rows.append(np.full((10, full_rows[-1].shape[1], 3), 255, dtype=np.uint8))
        zoom_rows.append(np.hstack(zstrip))
        zoom_rows.append(np.full((10, zoom_rows[-1].shape[1], 3), 255, dtype=np.uint8))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for rows, suffix in ((full_rows, "full"), (zoom_rows, "zoom")):
        canvas = np.vstack(rows)
        p = out.with_name(out.stem + "_" + suffix + out.suffix)
        cv2.imwrite(str(p), canvas[..., ::-1])
        print(f"[ok] {p}  {canvas.shape[1]}x{canvas.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
