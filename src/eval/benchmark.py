"""统一评测：全局指标 + 分区局部指标。

评测口径（写进报告里）：
  - 测试集：DIV2K_valid_HR 后 80 张（前 20 张留给训练期验证），单独划分
  - 裁剪：512x512 center crop
  - 退化：固定 seed，所有方法看到完全相同的输入
  - 指标：PSNR / SSIM / LPIPS（保真+观感）+ 分区局部指标（平坦区块效应、
          纹理区梯度能量）+ 全局块效应

用法：
    python src/eval/benchmark.py --ckpt experiments/exp_A/best.pth --tag A
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.paired_dataset import PaiRedEvalDataset  # noqa: E402
from src.degrade.dct_degrade import DegradeConfig, block_energy, rgb_to_ycbcr  # noqa: E402
from src.eval.metrics import blockiness, gradient_energy, psnr, ssim  # noqa: E402
from src.models.srvgg import build_model  # noqa: E402

DIV2K = ROOT / "SR_Project" / "dataset"
TEST_SPLIT_START = 20  # 前 20 张给验证


def eval_cfg(scale: int = 2) -> DegradeConfig:
    return DegradeConfig(scale_choices=(scale,))


def infer_tta(model, x: torch.Tensor) -> torch.Tensor:
    """8 路翻转/旋转测试增强平均。

    对"尽量接近原图"的目标是稳赚的：推理成本 x8，PSNR 通常 +0.1~0.2 dB，
    且不需要重训。
    """
    outs = []
    for k in range(4):
        for flip in (False, True):
            xi = torch.rot90(x, k, dims=[2, 3])
            if flip:
                xi = torch.flip(xi, dims=[3])
            yi = model(xi)
            if flip:
                yi = torch.flip(yi, dims=[3])
            yi = torch.rot90(yi, -k, dims=[2, 3])
            outs.append(yi)
    return torch.stack(outs, 0).mean(0)


def load_model(ckpt: str | None, device) -> torch.nn.Module:
    model = build_model(scale=1).to(device)
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(sd.get("params", sd), strict=True)
        model.eval()
    return model


def region_split(gt_u8: np.ndarray, block: int) -> tuple[np.ndarray, np.ndarray]:
    """按 block 网格的局部能量分成平坦区 / 纹理区（各占一半块数）。"""
    y, _, _ = rgb_to_ycbcr(gt_u8.astype(np.float32))
    h = (y.shape[0] // block) * block
    w = (y.shape[1] // block) * block
    e = block_energy(y[:h, :w], block).reshape(h // block, w // block)
    med = np.median(e)
    return e <= med, e >= med


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None, help="留空则评测退化图本身（baseline）")
    ap.add_argument("--tag", type=str, default="model")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--max-images", type=int, default=80)
    ap.add_argument("--seed", type=int, default=9999)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--tta", type=int, default=0, help="1=启用 8 路翻转平均")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = eval_cfg(args.scale)
    block = 8 * args.scale

    ds = PaiRedEvalDataset(
        DIV2K / "DIV2K_valid_HR",
        cfg=cfg,
        seed=args.seed,
        crop=args.crop,
        max_images=None,
    )
    # 测试集：跳过前 20 张
    ds.files = ds.files[TEST_SPLIT_START : TEST_SPLIT_START + args.max_images]
    loader = DataLoader(ds, batch_size=args.batch, num_workers=2, collate_fn=ds.collate)

    model = load_model(args.ckpt, device)

    try:
        import lpips as _lpips

        lp_net = _lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    except Exception:
        lp_net = None

    acc = {k: [] for k in [
        "psnr", "ssim", "lpips", "blockiness", "blockiness_flat", "blockiness_tex",
        "grad_tex", "grad_flat", "psnr_flat", "psnr_tex",
    ]}
    deg_acc = {k: [] for k in ["psnr", "ssim", "lpips", "blockiness"]}

    for deg, gt, names in loader:
        deg = deg.to(device)
        gt = gt.to(device)
        if args.ckpt:
            with torch.no_grad():
                pred = (infer_tta(model, deg) if args.tta else model(deg)).clamp(0, 1)
        else:
            pred = deg

        gt_u8 = (gt[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pr_u8 = (pred[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        dg_u8 = (deg[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        flat, tex = region_split(gt_u8, block)

        acc["psnr"].append(psnr(gt_u8, pr_u8))
        acc["ssim"].append(ssim(gt_u8, pr_u8))
        acc["blockiness"].append(blockiness(pr_u8, block))
        acc["grad_tex"].append(gradient_energy(pr_u8) / max(gradient_energy(gt_u8), 1e-6))

        # 分区指标：按块统计后取均值
        from src.eval.metrics import blockiness_map

        bm = blockiness_map(pr_u8, block)
        gm = blockiness_map(gt_u8, block)
        n = min(bm.shape[0], flat.shape[0]), min(bm.shape[1], flat.shape[1])
        bm, gm = bm[: n[0], : n[1]], gm[: n[0], : n[1]]
        f, t = flat[: n[0], : n[1]], tex[: n[0], : n[1]]
        acc["blockiness_flat"].append(float((bm[f] - gm[f]).mean()))
        acc["blockiness_tex"].append(float((bm[t] - gm[t]).mean()))

        # 分区 PSNR（逐块 MSE 再分区平均）
        g = gt_u8.astype(np.float32)
        p = pr_u8.astype(np.float32)
        err = ((g - p) ** 2).mean(axis=2)
        eb = err[: n[0] * block, : n[1] * block].reshape(n[0], block, n[1], block).mean(axis=(1, 3))
        acc["psnr_flat"].append(float(10 * np.log10(255**2 / max(eb[f].mean(), 1e-6))))
        acc["psnr_tex"].append(float(10 * np.log10(255**2 / max(eb[t].mean(), 1e-6))))
        acc["grad_flat"].append(0.0)

        # 退化基线
        deg_acc["psnr"].append(psnr(gt_u8, dg_u8))
        deg_acc["ssim"].append(ssim(gt_u8, dg_u8))
        deg_acc["blockiness"].append(blockiness(dg_u8, block))

        if lp_net is not None:
            with torch.no_grad():
                v = lp_net(pred * 2 - 1, gt * 2 - 1)
            acc["lpips"].append(float(v.mean()))
            with torch.no_grad():
                vd = lp_net(deg * 2 - 1, gt * 2 - 1)
            deg_acc["lpips"].append(float(vd.mean()))

    res = {k: float(np.mean(v)) if v else None for k, v in acc.items()}
    res["degraded"] = {k: float(np.mean(v)) if v else None for k, v in deg_acc.items()}
    res["num_images"] = len(ds.files)
    res["tag"] = args.tag
    res["ckpt"] = args.ckpt
    res["eval"] = {
        "crop": args.crop,
        "scale": args.scale,
        "block": block,
        "seed": args.seed,
        "test_start": TEST_SPLIT_START,
    }

    out_dir = Path(args.out) if args.out else (Path(args.ckpt).parent if args.ckpt else ROOT / "experiments" / args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n[{args.tag}] n={res['num_images']}")
    print(f"  退化基线 : PSNR {res['degraded']['psnr']:.3f}  SSIM {res['degraded']['ssim']:.4f}  "
          f"LPIPS {res['degraded']['lpips']:.4f}  block {res['degraded']['blockiness']:.3f}")
    print(f"  复原结果 : PSNR {res['psnr']:.3f}  SSIM {res['ssim']:.4f}  "
          f"LPIPS {res['lpips']:.4f}  block {res['blockiness']:.3f}")
    print(f"  分区     : 平坦区 Δblock {res['blockiness_flat']:+.3f} (PSNR {res['psnr_flat']:.2f}) | "
          f"纹理区 Δblock {res['blockiness_tex']:+.3f} (PSNR {res['psnr_tex']:.2f}) | "
          f"纹理梯度保留 {res['grad_tex']:.3f}")
    print(f"[ok] {out_dir/'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
