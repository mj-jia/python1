"""对抗微调实验（H 组）。

从已训练好的 G 组权重出发，加判别器再训一段，看"更像照片"能到什么程度。
**不改动原有 D/G 组的任何代码与结果**，输出单独存到 experiments/exp_H。

用法：
    python src/train/train_gan.py --iters 2000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.paired_dataset import PaiRedEvalDataset, PairedDegradeDataset  # noqa: E402
from src.degrade.dct_degrade import DegradeConfig  # noqa: E402
from src.eval.metrics import psnr as psnr_np  # noqa: E402
from src.models.srvgg import build_model  # noqa: E402
from src.train.gan import GANLoss, UNetDiscriminatorSN  # noqa: E402
from src.train.losses import (  # noqa: E402
    BlockinessLoss,
    SpatialAdaptiveLoss,
    blockiness_tensor,
    flatness_weight,
)

DIV2K = ROOT / "SR_Project" / "dataset"


def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=str, default=str(ROOT / "experiments" / "exp_G" / "best.pth"))
    ap.add_argument("--out", type=str, default=str(ROOT / "experiments" / "exp_H"))
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--lr-g", type=float, default=1e-4)
    ap.add_argument("--lr-d", type=float, default=1e-4)
    ap.add_argument("--lambda-gan", type=float, default=0.1)
    ap.add_argument("--d-steps", type=int, default=1, help="每轮判别器更新次数")
    ap.add_argument("--gan-warmup", type=int, default=0, help="对抗权重从 0 线性升到目标值的步数")
    ap.add_argument("--lambda-lpips", type=float, default=0.05)
    ap.add_argument("--lambda-block", type=float, default=0.03)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-images", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def validate(G, cfg, args, device, seed=1234) -> dict:
    ds = PaiRedEvalDataset(
        DIV2K / "DIV2K_valid_HR", cfg=cfg, seed=seed,
        crop=args.patch, max_images=args.val_images,
    )
    loader = DataLoader(ds, batch_size=1, num_workers=2)
    G.eval()
    ps, bs = [], []
    with torch.no_grad():
        for deg, gt, _ in loader:
            deg, gt = deg.to(device), gt.to(device)
            pred = G(deg).clamp(0, 1)
            ps.append(psnr_np(gt[0].cpu().numpy(), pred[0].cpu().numpy(), peak=1.0))
            bs.append(float(blockiness_tensor(pred, 8 * args.scale).mean()))
    G.train()
    return {"psnr": float(np.mean(ps)), "blockiness": float(np.mean(bs))}


def main() -> int:
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = DegradeConfig(scale_choices=(args.scale,))
    ds = PairedDegradeDataset(
        DIV2K / "DIV2K_train_HR", patch=args.patch,
        length=max(args.iters * args.batch, 800), cfg=cfg,
    )
    loader = DataLoader(
        ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=True, persistent_workers=args.workers > 0,
    )

    G = build_model(scale=1).to(device)
    sd = torch.load(args.init, map_location="cpu")
    G.load_state_dict(sd.get("params", sd))
    D = UNetDiscriminatorSN().to(device)
    print(f"[init] 生成器 <- {args.init} | 判别器参数 {sum(p.numel() for p in D.parameters())/1e6:.2f}M")

    pixel_loss = SpatialAdaptiveLoss(lambda_lpips=args.lambda_lpips).to(device)
    block_loss = BlockinessLoss(block=8 * args.scale).to(device)
    gan_loss = GANLoss()

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.9, 0.99))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.9, 0.99))
    scaler_g = torch.amp.GradScaler("cuda")
    scaler_d = torch.amp.GradScaler("cuda")

    log_f = (out_dir / "train_log.jsonl").open("a", encoding="utf-8")
    (out_dir / "meta.json").write_text(
        json.dumps({"args": vars(args), "type": "gan_finetune"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    it, best = 0, -1e9
    t0 = time.time()
    acc = {"g": 0.0, "d": 0.0, "gan": 0.0, "pix": 0.0, "n": 0}
    G.train()
    D.train()

    while it < args.iters:
        for deg, gt in loader:
            if it >= args.iters:
                break
            deg, gt = deg.to(device, non_blocking=True), gt.to(device, non_blocking=True)

            # ---- 判别器（可多步更新，先让它变强）----
            loss_d_val = 0.0
            for _ in range(args.d_steps):
                with torch.no_grad():
                    fake_d = G(deg).clamp(0, 1)
                with torch.amp.autocast("cuda"):
                    loss_d = gan_loss.forward_d(D(gt), D(fake_d.detach()))
                opt_d.zero_grad(set_to_none=True)
                scaler_d.scale(loss_d).backward()
                scaler_d.step(opt_d)
                scaler_d.update()
                loss_d_val += float(loss_d.detach()) / args.d_steps

            # ---- 生成器 ----
            # 对抗权重预热：前期让判别器先站稳，避免生成器被噪声梯度带偏
            lam = args.lambda_gan
            if args.gan_warmup > 0:
                lam = args.lambda_gan * min(1.0, it / args.gan_warmup)
            with torch.amp.autocast("cuda"):
                fake = G(deg)
                ld = pixel_loss(fake.clamp(0, 1), gt)
                lb = block_loss(fake.clamp(0, 1), gt, weight=flatness_weight(gt))
                l_gan = gan_loss.forward_g(D(fake.clamp(0, 1)))
                loss_g = ld["total"] + args.lambda_block * lb + lam * l_gan
            opt_g.zero_grad(set_to_none=True)
            scaler_g.scale(loss_g).backward()
            scaler_g.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            scaler_g.step(opt_g)
            scaler_g.update()

            acc["g"] += float(loss_g.detach())
            acc["d"] += loss_d_val
            acc["gan"] += float(l_gan.detach())
            acc["pix"] += float(ld["total"].detach())
            acc["n"] += 1
            it += 1

            if it % 100 == 0:
                n = max(1, acc["n"])
                rec = {
                    "iter": it, "loss_g": acc["g"] / n, "loss_d": acc["d"] / n,
                    "gan": acc["gan"] / n, "pix": acc["pix"] / n,
                    "sec_per_iter": (time.time() - t0) / it,
                }
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                if it % 500 == 0:
                    print(f"  it {it:5d} G {rec['loss_g']:.4f} D {rec['loss_d']:.4f} "
                          f"gan {rec['gan']:.4f} {rec['sec_per_iter']:.2f}s/it")
                acc = {"g": 0.0, "d": 0.0, "gan": 0.0, "pix": 0.0, "n": 0}

            if it % args.val_every == 0 or it == args.iters:
                vm = validate(G, cfg, args, device)
                log_f.write(json.dumps({"iter": it, **vm}) + "\n")
                log_f.flush()
                print(f"  [val] it {it} PSNR {vm['psnr']:.3f} 块效应 {vm['blockiness']:.3f}")
                torch.save({"params": G.state_dict(), "iter": it, **vm}, out_dir / "last.pth")
                # 用 PSNR 和块效应各占一半的综合分挑最优，避免只挑 PSNR 挑到最糊的
                score = vm["psnr"] - 2.0 * max(0.0, vm["blockiness"] - 0.9)
                if score > best:
                    best = score
                    torch.save({"params": G.state_dict(), "iter": it, **vm}, out_dir / "best.pth")

    log_f.close()
    print(f"[done] {(time.time()-t0)/60:.1f} min -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
