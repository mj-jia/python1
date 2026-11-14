"""训练脚本：四组消融（A/B/C/D），同 seed、同迭代、同数据顺序。

A 基线    非均匀退化 + 普通 L1
B         非均匀退化 + 普通 L1 + 块效应损失
C         非均匀退化 + 空间自适应混合损失
D 完整    非均匀退化 + 空间自适应混合损失 + 块效应损失

用法：
    python src/train/train_sr.py --exp A
    python src/train/train_sr.py --exp D --iters 20000
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
from src.train.losses import (  # noqa: E402
    BlockinessLoss,
    SpatialAdaptiveLoss,
    blockiness_tensor,
    flatness_weight,
)

DIV2K = ROOT / "SR_Project" / "dataset"

# 四组消融的开关（与文档第八节表格一一对应）
EXPERIMENTS = {
    "A": dict(uniform_degrade=False, block=False, spatial=False),
    "B": dict(uniform_degrade=False, block=True, spatial=False),
    "C": dict(uniform_degrade=False, block=False, spatial=True),
    "D": dict(uniform_degrade=False, block=True, spatial=True),
    # 附加对照组：均匀退化 + 全损失，用来证明"退化建模本身有贡献"
    "U": dict(uniform_degrade=True, block=True, spatial=True),
}


def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=str, default="A", choices=list(EXPERIMENTS))
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--scale", type=int, default=2, help="退化下采样整数倍")
    ap.add_argument("--lambda-block", type=float, default=0.05)
    # LPIPS 权重必须远小于 L1：两项都归一化到均值 1，1:1 会让感知项压垮保真，
    # 实测 PSNR 掉 1.3 dB。0.1 是"观感略好、保真不掉"的折中。
    ap.add_argument("--lambda-lpips", type=float, default=0.1)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-images", type=int, default=12)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--amp", type=int, default=1)
    # 允许手动覆盖实验开关
    ap.add_argument("--block", type=str, default=None, choices=["on", "off"])
    ap.add_argument("--spatial", type=str, default=None, choices=["on", "off"])
    return ap.parse_args()


def build_degrade_cfg(scale: int, uniform: bool) -> DegradeConfig:
    cfg = DegradeConfig(scale_choices=(scale,))
    if uniform:
        cfg.uniform_quality = True
    return cfg


def build_losses(args, flags, block_size: int, device):
    l1 = nn.L1Loss()
    spatial = None
    if flags["spatial"]:
        spatial = SpatialAdaptiveLoss(
            lambda_lpips=args.lambda_lpips, use_lpips=True
        ).to(device)
    block = None
    if flags["block"]:
        block = BlockinessLoss(block=block_size).to(device)
    return l1, spatial, block


def validate(model, cfg, args, device, fixed_seed=1234) -> dict:
    ds = PaiRedEvalDataset(
        DIV2K / "DIV2K_valid_HR",
        cfg=cfg,
        seed=fixed_seed,
        crop=args.patch,
        max_images=args.val_images,
    )
    loader = DataLoader(ds, batch_size=1, num_workers=2, shuffle=False)
    model.eval()
    ps, bs_pred, bs_gt = [], [], []
    with torch.no_grad():
        for deg, gt, _ in loader:
            deg, gt = deg.to(device), gt.to(device)
            pred = model(deg).clamp(0, 1)
            # 注意：张量在 [0,1]，peak 必须传 1.0
            ps.append(
                psnr_np(gt[0].cpu().numpy(), pred[0].cpu().numpy(), peak=1.0)
            )
            bs_pred.append(float(blockiness_tensor(pred, 8 * args.scale).mean()))
            bs_gt.append(float(blockiness_tensor(gt, 8 * args.scale).mean()))
    model.train()
    return {
        "psnr": float(np.mean(ps)),
        "blockiness_pred": float(np.mean(bs_pred)),
        "blockiness_gt": float(np.mean(bs_gt)),
        "blockiness_deg": 0.0,
    }


def main() -> int:
    args = get_args()
    flags = dict(EXPERIMENTS[args.exp])
    if args.block is not None:
        flags["block"] = args.block == "on"
    if args.spatial is not None:
        flags["spatial"] = args.spatial == "on"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out) if args.out else ROOT / "experiments" / f"exp_{args.exp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_degrade_cfg(args.scale, flags["uniform_degrade"])
    train_ds = PairedDegradeDataset(
        DIV2K / "DIV2K_train_HR",
        patch=args.patch,
        length=max(args.iters * args.batch, 800),
        cfg=cfg,
        training=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
    )

    model = build_model(scale=1).to(device)
    if args.resume:
        raw = torch.load(args.resume, map_location="cpu")
        if isinstance(raw, dict):
            for k in ("params_ema", "params"):
                if k in raw and isinstance(raw[k], dict):
                    raw = raw[k]
                    break
        mine = model.state_dict()
        # strict=False 只忽略多余的 key，形状不匹配仍会报错，必须手动过滤
        usable = {k: v for k, v in raw.items() if k in mine and mine[k].shape == v.shape}
        dropped = [k for k in mine if k not in usable]
        model.load_state_dict(usable, strict=False)
        print(
            f"[init] 载入 {args.resume}：{len(usable)}/{len(mine)} 参数命中，"
            f"跳过 {len(dropped)} 个（{dropped[:2]}）"
        )

    l1_loss, spatial_loss, block_loss = build_losses(args, flags, 8 * args.scale, device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp) and device.type == "cuda")

    def lr_at(it: int) -> float:
        if it < args.warmup:
            return args.lr * (it + 1) / args.warmup
        t = (it - args.warmup) / max(1, args.iters - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    log_path = out_dir / "train_log.jsonl"
    log_f = log_path.open("a", encoding="utf-8")
    meta = {
        "exp": args.exp,
        "flags": flags,
        "args": vars(args),
        "degrade": cfg.to_dict(),
        "params": model.num_params(),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[exp {args.exp}] {flags} | params {model.num_params()/1e6:.3f}M | out {out_dir}")

    it = 0
    best_psnr = 0.0
    t0 = time.time()
    running = {"loss": 0.0, "l1": 0.0, "lpips": 0.0, "block": 0.0, "n": 0}

    model.train()
    while it < args.iters:
        for deg, gt in train_loader:
            if it >= args.iters:
                break
            deg = deg.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            for g in opt.param_groups:
                g["lr"] = lr_at(it)

            with torch.amp.autocast("cuda", enabled=bool(args.amp) and device.type == "cuda"):
                pred = model(deg)
                if spatial_loss is not None:
                    ld = spatial_loss(pred.clamp(0, 1), gt)
                    loss = ld["total"]
                    l1_v = float(ld["l1"])
                    lp_v = float(ld["lpips"])
                else:
                    loss = l1_loss(pred, gt)
                    l1_v, lp_v = float(loss.detach()), 0.0
                blk_v = 0.0
                if block_loss is not None:
                    w = flatness_weight(gt) if flags["spatial"] else None
                    lb = block_loss(pred.clamp(0, 1), gt, weight=w)
                    loss = loss + args.lambda_block * lb
                    blk_v = float(lb.detach())

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            running["loss"] += float(loss.detach())
            running["l1"] += l1_v
            running["lpips"] += lp_v
            running["block"] += blk_v
            running["n"] += 1
            it += 1

            if it % 100 == 0:
                n = max(1, running["n"])
                rec = {
                    "iter": it,
                    "loss": running["loss"] / n,
                    "l1": running["l1"] / n,
                    "lpips": running["lpips"] / n,
                    "block": running["block"] / n,
                    "lr": opt.param_groups[0]["lr"],
                    "sec_per_iter": (time.time() - t0) / max(1, it),
                }
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                if it % 500 == 0:
                    print(
                        f"  it {it:6d} loss {rec['loss']:.4f} l1 {rec['l1']:.4f} "
                        f"lpips {rec['lpips']:.4f} blk {rec['block']:.4f} "
                        f"{rec['sec_per_iter']:.2f}s/it"
                    )
                running = {"loss": 0.0, "l1": 0.0, "lpips": 0.0, "block": 0.0, "n": 0}

            if it % args.val_every == 0 or it == args.iters:
                vm = validate(model, build_degrade_cfg(args.scale, flags["uniform_degrade"]), args, device)
                rec = {"iter": it, **vm}
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                print(f"  [val] it {it} PSNR {vm['psnr']:.3f} 块效应 {vm['blockiness_pred']:.3f}")
                torch.save({"params": model.state_dict(), "iter": it, "psnr": vm["psnr"]},
                           out_dir / "last.pth")
                if vm["psnr"] > best_psnr:
                    best_psnr = vm["psnr"]
                    torch.save({"params": model.state_dict(), "iter": it, "psnr": vm["psnr"]},
                               out_dir / "best.pth")

    log_f.close()
    print(f"[done] exp {args.exp} best PSNR {best_psnr:.3f} | {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
