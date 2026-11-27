#!/usr/bin/env python
"""Generate plain publication figures for Phase 0 (SVD intrinsic rank & control recall) and
Phase 1/2 (full systems + speculative planning progression and Pareto frontier).

Follows the repo's plain figure conventions: no bold headline titles, hairline axes, one small
caption per panel, and full sentences in footnotes.
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.transforms import blended_transform_factory

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, BLUE_DARK, BLUE_LIGHT, GRAY, ORANGE, TEAL = "#2a78d6", "#0d366b", "#86b6ef", "#a8a69e", "#d96b27", "#1f8a70"

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"], "font.size": 10.5, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "savefig.dpi": 220, "savefig.facecolor": "white",
})


def plot_probe_figure(probe_json, out_png):
    R = json.loads(Path(probe_json).read_text())
    P = R["cem_probe"]
    iters = P["probe_iters"]
    ranks = P["ranks"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.55))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.89, bottom=0.25, wspace=0.28)

    # Panel A: Cumulative variance explained vs rank R for early / mid / late CEM iterations
    show_iters = [iters[0], iters[len(iters) // 2], iters[-1]]
    colors_a = [BLUE_LIGHT, BLUE, BLUE_DARK]
    for it, col in zip(show_iters, colors_a):
        s = P["svd_by_iter"][str(it)]
        ys = [100.0 * s["var_explained"][str(r)] for r in ranks]
        ax1.plot(ranks, ys, marker="o", ms=4.5, lw=2.0, color=col,
                 label=f"iter {it} (sigma={s['sigma_mean']:.2f}, eff. rank {s['eff_rank_var']:.1f})")
    ax1.axvline(32, color=AXIS, ls="--", lw=1.0)
    ax1.text(34, 86.5, "R = 32\n(97-99% var)", fontsize=9, color=MUTED, va="center")
    ax1.set_xlabel("Subspace rank R")
    ax1.set_ylabel("Variance explained (%)")
    ax1.set_ylim(70, 101.5)
    ax1.set_xlim(0, max(ranks) + 4)
    ax1.grid(axis="y", color=GRID, lw=0.7)
    ax1.set_axisbelow(True)
    ax1.legend(frameon=False, fontsize=9, loc="lower right")
    ax1.set_title("Intrinsic rank of CEM rollout perturbations in DINOv2 latent space", fontsize=10, color=MUTED, loc="left", pad=8)

    # Panel B: recall@45 across CEM iterations
    m_key = "45"
    series = [
        ("Oracle rank-16 SVD", [P["oracle_rank_by_iter"][str(it)]["16"]["recall"][m_key] for it in iters], BLUE_DARK, "--"),
        ("Latent Jacobian (B=50)", [P["controls_by_iter"]["jacobian_lin"][str(it)]["recall"][m_key] for it in iters], BLUE, "-"),
        ("Online ridge (0 extra F calls)", [np.nan if P["controls_by_iter"]["online_ridge"][str(it)] is None else P["controls_by_iter"]["online_ridge"][str(it)]["recall"][m_key] for it in iters], TEAL, "-"),
        ("1st-order scalar grad (B=1)", [P["controls_by_iter"]["grad_lin"][str(it)]["recall"][m_key] for it in iters], ORANGE, "-"),
        ("Copy reference (chance)", [P["controls_by_iter"]["copy_ref"][str(it)]["recall"][m_key] for it in iters], GRAY, ":"),
    ]
    for label, ys, col, ls in series:
        ax2.plot(iters, ys, marker="o", ms=4.5, lw=2.0, color=col, ls=ls, label=label)
    ax2.set_xlabel("CEM iteration")
    ax2.set_ylabel("Elite recall@45 (true top-30 in draft top-45)")
    ax2.set_ylim(-0.03, 1.05)
    ax2.grid(axis="y", color=GRID, lw=0.7)
    ax2.set_axisbelow(True)
    ax2.legend(frameon=False, fontsize=8.8, loc="center right")
    ax2.set_title("Elite recall@45 of zero-training reference controls across CEM", fontsize=10, color=MUTED, loc="left", pad=8)

    fig.text(
        0.07, 0.03,
        "Measured on 8 PushT tasks (30 CEM iterations x 300 candidates, horizon 5, RTX 5090, fp16_reuse).\n"
        "Scalar gradient linearization collapses after iteration 0, while latent Gauss-Newton curvature (Jacobian and zero-cost online ridge) improves as CEM converges.",
        fontsize=8.8, color=MUTED, va="bottom", linespacing=1.35,
    )
    fig.savefig(out_png)
    plt.close(fig)


def plot_pareto_figure(eval32_v1, eval32_v2, out_png):
    d1 = json.loads(Path(eval32_v1).read_text())["summary"]
    d2 = json.loads(Path(eval32_v2).read_text())["summary"]

    # Unified 32-task held-out comparison on fp16_unmasked_reuse
    rows = [
        ("Full CEM (K=300)", d2["full"], GRAY, "o"),
        ("Few-150 (K=150)", d2["few150"], GRAY, "s"),
        ("Few-100 (K=100)", d1["few100"], GRAY, "^"),
        ("NoRef-G D&V (m=45)", d1["norefg:m45"], ORANGE, "D"),
        ("Ref-G D&V (m=45)", d2["refg:m45"], BLUE_LIGHT, "o"),
        ("Online Ridge (K=1000, m=60)", d2["online:k1000:m60"], TEAL, "s"),
        ("Hybrid (K=1000, m=60)", d2["hybrid:k1000:m60"], BLUE, "o"),
        ("Hybrid (K=1000, m=90-45)", d2["hybrid:k1000:m90-45"], BLUE_DARK, "*"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.7))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.89, bottom=0.25, wspace=0.28)

    # Panel A: State distance (lower is better) vs median plan latency
    for label, s, col, mk in rows:
        ms = 11 if mk == "*" else 6.5
        x, y = s["episode_median_s"], s["mean_final_state_dist"]
        ax1.scatter(x, y, c=col, marker=mk, s=ms**2, zorder=3, edgecolors="white", linewidths=0.7)
        ha, dx, dy = "left", 0.18, 0.0
        if "m=90-45" in label:
            dx, dy = 0.24, -1.3
        elif "m=60)" in label and "Hybrid" in label:
            dx, dy = 0.24, 0.9
        elif "Online" in label:
            dx, dy = 0.22, 0.6
        elif "Full" in label:
            ha, dx, dy = "right", 0.0, 1.5
        ax1.text(x + dx, y + dy, label, fontsize=8.5, color=INK2, va="center", ha=ha)
    ax1.axhline(d2["full"]["mean_final_state_dist"], color=AXIS, ls="--", lw=0.9)
    ax1.set_xlabel("Median plan latency (s), lower is better")
    ax1.set_ylabel("Mean final state distance to goal, lower is better")
    ax1.set_xlim(0.8, 10.0)
    ax1.set_ylim(23, 50)
    ax1.grid(axis="y", color=GRID, lw=0.7)
    ax1.set_axisbelow(True)
    ax1.set_title("Plan quality (state distance) against latency on 32 held-out tasks", fontsize=10, color=MUTED, loc="left", pad=8)

    # Panel B: Task success rate vs peak GPU memory (GiB)
    for label, s, col, mk in rows:
        ms = 11 if mk == "*" else 6.5
        x, y = s["peak_mem_alloc_mb"] / 1024.0, 100.0 * s["success_rate"]
        ax2.scatter(x, y, c=col, marker=mk, s=ms**2, zorder=3, edgecolors="white", linewidths=0.7)
        ha, dx, dy = "left", 0.20, 0.0
        if "Full" in label:
            ha, dx, dy = "right", 0.0, 2.0
        elif "Online" in label:
            ha, dx, dy = "right", 0.0, -2.0
        elif "m=90-45" in label:
            dx, dy = 0.22, 1.8
        elif "m=60)" in label and "Hybrid" in label:
            dx, dy = 0.22, -1.6
        ax2.text(x + dx, y + dy, label, fontsize=8.5, color=INK2, va="center", ha=ha)
    ax2.set_xlabel("Peak GPU memory allocated (GiB), lower is better")
    ax2.set_ylabel("Task success rate (%), higher is better")
    ax2.set_xlim(0.8, 10.1)
    ax2.set_ylim(58, 87)
    ax2.grid(axis="y", color=GRID, lw=0.7)
    ax2.set_axisbelow(True)
    ax2.set_title("Task success against peak GPU memory on 32 held-out tasks", fontsize=10, color=MUTED, loc="left", pad=8)

    fig.text(
        0.07, 0.03,
        "32 paired PushT tasks from held-out episodes (val+heldout, seed 99, 30 CEM iterations, horizon 5, RTX 5090, FlashAttention UnmaskedReusePredict).\n"
        "Hybrid (K=1000, m=90-45) matches full CEM success (81.2%) with lower state distance (27.1 vs 29.9) at 4.92x lower latency and 3.25x lower GPU memory.",
        fontsize=8.8, color=MUTED, va="bottom", linespacing=1.35,
    )
    fig.savefig(out_png)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-json", required=True)
    ap.add_argument("--eval-v1", required=True)
    ap.add_argument("--eval-v2", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_probe_figure(args.probe_json, out / "fig_probe_rank.png")
    plot_pareto_figure(args.eval_v1, args.eval_v2, out / "fig_specplan_pareto.png")
    print(f"Wrote figures to {out}")


if __name__ == "__main__":
    main()
