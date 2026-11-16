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
    # ---------------- 空间非均匀改造的四组开关 ----------------
    ap.add_argument(
        "--degrade-mode",
        type=str,
        default="legacy",
        choices=["legacy", "official", "link"],
        help="legacy=原有两段式；official=Real-ESRGAN 式两代均匀；link=社交链路式多代+相位漂移",
    )
    ap.add_argument(
        "--block-mode",
        type=str,
        default="single",
        choices=["single", "multi"],
        help="single=只在单一对齐网格上算块效应损失；multi=多相位取最大",
    )
    ap.add_argument("--n-gen", type=str, default="1,5", help="链路式退化代次区间，如 1,5")
    ap.add_argument("--hr-train", type=str, default=str(DIV2K / "DIV2K_train_HR"))
    ap.add_argument("--hr-val", type=str, default=str(DIV2K / "DIV2K_valid_HR"))
    ap.add_argument("--tag", type=str, default=None, help="输出目录后缀，便于同机多组并行")
    return ap.parse_args()


def build_degrade_cfg(
    scale: int, uniform: bool, mode: str = "legacy", n_gen: tuple = (1, 5)
) -> DegradeConfig:
    """构造退化配置。

    official : 两代、每代整图统一质量、无相位漂移、不改尺寸
               （对应 Real-ESRGAN 的二阶均匀退化，作为"现有做法"的对照）
    link     : 1~5 代、逐块内容自适应质量、每代随机改尺寸、网格相位漂移
               （对应真实社交转发链路，作为本项目的做法）
    legacy   : 完全保持原有行为，老实验可复现
    """
    cfg = DegradeConfig(scale_choices=(scale,))
    if mode == "official":
        cfg.uniform_quality = True
        cfg.uniform_q_range = (30.0, 90.0)
        cfg.n_gen_range = (2, 2)
        cfg.phase_drift = False
        cfg.resize_gen_prob = 0.0
    elif mode == "link":
        cfg.uniform_quality = False
        cfg.n_gen_range = (int(n_gen[0]), int(n_gen[1]))
        cfg.phase_drift = True
        cfg.resize_gen_prob = 0.5
    else:
        cfg.uniform_quality = bool(uniform)
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
        block = BlockinessLoss(
            block=block_size, phases=16 if args.block_mode == "multi" else 1
        ).to(device)
    return l1, spatial, block


# 评测时的候选相位：同时给出"对齐网格"和"最坏相位"两个块效应读数
PHASES = [(r, c) for r in (0, 2, 4, 6) for c in (0, 2, 4, 6)]


def validate(model, cfg, args, device, tag: str = "", fixed_seed: int = 1234) -> dict:
    """固定种子退化下的 PSNR 与块效应。

    blk_aligned 只在 (0,0) 相位上量；blk_worst 取 16 个相位的最大，
    后者才是"网格错位后真实可见的块效应"，也是本项目要压下去的指标。
    """
    ds = PaiRedEvalDataset(
        args.hr_val,
        cfg=cfg,
        seed=fixed_seed,
        crop=args.patch,
        max_images=args.val_images,
    )
    loader = DataLoader(ds, batch_size=1, num_workers=2, shuffle=False)
    block = 8 * args.scale
    model.eval()
    ps, worst_p, worst_g, aligned_p = [], [], [], []
    with torch.no_grad():
        for deg, gt, _ in loader:
            deg, gt = deg.to(device), gt.to(device)
            pred = model(deg).clamp(0, 1)
            # 注意：张量在 [0,1]，peak 必须传 1.0
            ps.append(psnr_np(gt[0].cpu().numpy(), pred[0].cpu().numpy(), peak=1.0))
            vp = [float(blockiness_tensor(pred, block, phase=ph).mean()) for ph in PHASES]
            vg = [float(blockiness_tensor(gt, block, phase=ph).mean()) for ph in PHASES]
            worst_p.append(max(vp))
            worst_g.append(max(vg))
            aligned_p.append(vp[0])
    model.train()
    return {
        f"psnr{tag}": float(np.mean(ps)),
        f"blk_worst{tag}": float(np.mean(worst_p)),
        f"blk_worst_gt{tag}": float(np.mean(worst_g)),
        f"blk_aligned{tag}": float(np.mean(aligned_p)),
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
    if args.tag:
        out_dir = out_dir.parent / f"{out_dir.name}_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_gen = tuple(int(v) for v in args.n_gen.split(","))
    if len(n_gen) != 2:
        raise ValueError("--n-gen 需要形如 1,5")

    cfg = build_degrade_cfg(args.scale, flags["uniform_degrade"], args.degrade_mode, n_gen)
    cfg_link = build_degrade_cfg(args.scale, False, "link", n_gen)
    cfg_official = build_degrade_cfg(args.scale, True, "official", n_gen)
    train_ds = PairedDegradeDataset(
        args.hr_train,
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
                # 两套退化分布各评一次：link 是目标分布，official 是对照分布
                vm = {}
                vm.update(validate(model, cfg_link, args, device, tag="_link"))
                vm.update(validate(model, cfg_official, args, device, tag="_off"))
                vm["psnr"] = vm["psnr_link"]
                rec = {"iter": it, **vm}
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                print(
                    f"  [val] it {it} PSNR(link) {vm['psnr_link']:.3f} "
                    f"PSNR(off) {vm['psnr_off']:.3f} "
                    f"块效应(最坏相位) {vm['blk_worst_link']:.3f} "
                    f"GT {vm['blk_worst_gt_link']:.3f}"
                )
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
