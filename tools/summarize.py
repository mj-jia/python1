"""把各组 metrics.json 汇总成一张 markdown 对比表。

用法：
    python tools/summarize.py
    python tools/summarize.py --exps A B C D U
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BASE_ROW = (
    "| 退化基线 | - | - | - | {psnr:.3f} | {ssim:.4f} | {lpips:.4f} | {blk:.3f} | - | - |"
)


def load(exp: str, root: Path) -> dict | None:
    p = root / f"exp_{exp}" / "metrics.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text(encoding="utf-8"))
    meta_p = root / f"exp_{exp}" / "meta.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        m["flags"] = meta.get("flags", {})
        m["resume"] = (meta.get("args") or {}).get("resume")
        m["iters"] = (meta.get("args") or {}).get("iters")
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="*", default=["A", "B", "C", "D", "U"])
    ap.add_argument("--root", type=str, default=str(ROOT / "experiments"))
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    root = Path(args.root)
    rows, degraded = [], None
    for exp in args.exps:
        m = load(exp, root)
        if m is None:
            print(f"[skip] exp_{exp} 缺少 metrics.json")
            continue
        if degraded is None:
            degraded = m.get("degraded")
        rows.append((exp, m))

    if not rows:
        print("没有可汇总的结果")
        return 1

    lines = ["## 四组消融结果\n"]
    lines.append("评测口径：DIV2K_valid 后 80 张，512×512 center crop，固定退化 seed。\n")
    lines.append(
        "| 组 | 退化建模 | 块效应损失 | 空间加权 | PSNR↑ | SSIM↑ | LPIPS↓ | 块效应↓ | "
        "平坦区 Δ块效应↓ | 纹理梯度保留↑ |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    on_off = lambda v: "✓" if v else "✗"  # noqa: E731
    for exp, m in rows:
        f = m.get("flags", {})
        deg = "均匀" if f.get("uniform_degrade") else "**非均匀**"
        name = exp
        if m.get("resume"):
            name = f"{exp}（官方权重微调）"
        lines.append(
            f"| {name} | {deg} | {on_off(f.get('block'))} | {on_off(f.get('spatial'))} | "
            f"{m['psnr']:.3f} | {m['ssim']:.4f} | {m['lpips']:.4f} | {m['blockiness']:.3f} | "
            f"{m['blockiness_flat']:+.3f} | {m['grad_tex']:.3f} |"
        )

    if degraded:
        lines.append(
            f"| 退化图（不处理） | - | - | - | {degraded['psnr']:.3f} | {degraded['ssim']:.4f} | "
            f"{degraded.get('lpips') or 0:.4f} | {degraded['blockiness']:.3f} | - | 1.000 |"
        )

    lines.append("")

    # 相对基线的增益
    base = next((m for e, m in rows if e == "A"), None)
    if base and len(rows) > 1:
        lines.append("### 相对 A 组（只用退化建模）的增量\n")
        lines.append("| 组 | ΔPSNR | ΔSSIM | ΔLPIPS | Δ块效应 |")
        lines.append("|---|---|---|---|---|")
        for exp, m in rows:
            lines.append(
                f"| {exp} | {m['psnr'] - base['psnr']:+.3f} | {m['ssim'] - base['ssim']:+.4f} | "
                f"{m['lpips'] - base['lpips']:+.4f} | {m['blockiness'] - base['blockiness']:+.3f} |"
            )
        lines.append("")

    text = "\n".join(lines)
    print(text)
    out = Path(args.out) if args.out else root / "summary.md"
    out.write_text(text, encoding="utf-8")
    print(f"[ok] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
