#!/usr/bin/env python
"""results.json -> markdown summary (stdout); next to it summary.png (the result table as a figure) and scaling.png (baseline profile, rollout cost against K).
Usage: summarize.py runs/NAME/results.json"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.transforms import blended_transform_factory

path = Path(sys.argv[1]); R = json.loads(path.read_text()); M, V = R["meta"], R["variants"]
rng = np.random.default_rng(0)
gpu = M["gpu"].replace("NVIDIA GeForce ", "")


# Figure style: one hue for our work and gray for the released code, values written at the marks, hairline axes.
INK, INK2, MUTED, GRID, AXIS, BLUE, GRAY = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#2a78d6", "#a8a69e"
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # light to dark: each later optimization step is darker
LABEL = {"official": "Released code", "cache": "+ start frame encoded once", "cache_sdpa": "+ fused attention (SDPA)",
         "cache_sdpa_tf32": "+ TF32 matmul", "cache_sdpa_fp16": "+ FP16 autocast, in place of TF32",
         "fp16_compile": "FP16 + torch.compile", "fp16_reuse": "FP16 + key/value reuse",
         "fp16_reuse_compile": "FP16 + key/value reuse + torch.compile"}
SHORT = {"cache_sdpa": "+ fused attention", "cache_sdpa_tf32": "+ TF32", "cache_sdpa_fp16": "+ FP16", "fp16_reuse": "+ key/value reuse"}
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 11, "text.color": INK,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
                     "ytick.labelcolor": INK2, "savefig.dpi": 220, "savefig.facecolor": "white"})


def hbars(ax, values, colors, fmt, xmax, h=0.44):
    """Thin horizontal bars from one baseline, square at the baseline and rounded at the data end, value at the tip."""
    ax.set_xlim(0, xmax); ax.set_ylim(len(values) - 0.5, -0.5); ax.axis("off")
    box = ax.get_window_extent(); px_x, px_y = box.width / xmax, box.height / len(values)
    r = 3 * ax.figure.dpi / 72 / px_x  # 3 pt corner radius, in x data units
    for y, (v, c) in enumerate(zip(values, colors)):
        if v > 2 * r:
            ax.add_patch(FancyBboxPatch((0, y - h / 2), v, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                        mutation_aspect=px_x / px_y, lw=0, fc=c))
        ax.add_patch(Rectangle((0, y - h / 2), max(min(v, r), 1.5 / px_x), h, lw=0, fc=c))  # a sliver stays a sliver
        ax.text(v + 6 / px_x, y, fmt(v), va="center", ha="left", color=INK)
    ax.plot([0, 0], [-0.5, len(values) - 0.5], color=AXIS, lw=0.8, clip_on=False)


def kind(op):
    o = op.lower()
    for key, label in (("fmha", "fused attention kernel"), ("gemm", "GEMM (linear layers, attention matmul)"), ("softmax", "softmax"),
                       ("memcpy", "memcpy"), ("elementwise", "elementwise"), ("layer_norm", "layer norm"), ("gelu", "GELU")):
        if key in o:
            return label
    return "other"


def paired_speedup(a, b):
    """Speedup of b over a: ratio of mean latencies, 95% CI by bootstrap over the paired tasks."""
    a, b = np.array(a), np.array(b); idx = rng.integers(0, len(a), (10000, len(a)))
    boot = a[idx].mean(1) / b[idx].mean(1)
    return a.mean() / b.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


print(f"Hardware: {M['gpu']} (driver {M['driver']}), {M['cpu']}; torch {M['torch']}, CUDA {M['cuda']}, cuDNN {M['cudnn']}, "
      f"{M['backend']}; dtype {M['dtype']}; TF32 matmul {M['tf32_matmul']}, TF32 cuDNN {M['tf32_cudnn']}.")
print(f"Model: {M['model']}; context {M['num_hist']} frames x {M['tokens_per_frame']} tokens; DINO-WM commit {M['dino_wm_commit'][:7]}.")
if M.get("merged"):  # written by merge_runs.py
    g = M["merged"]
    print(f"Two processes on the same card, same tasks and noise: {', '.join(g['extra_variants'])} come from run {g['extra_run']}, the rest "
          f"from {g['main_run']}. Shared variant, median plan in each: "
          + "; ".join(f"{k} {a:.3f} s and {b:.3f} s" for k, (a, b) in g["anchor_median_s"].items()) + ".")
first = {k: d["first_rollout_s"] for k, d in R["selfcheck"].items() if isinstance(d, dict) and "compile" in k and "first_rollout_s" in d}
if first:
    print("torch.compile cold start (first rollout call, not part of any plan latency): " + ", ".join(f"{k} {v:.1f} s" for k, v in first.items()) + ".")

if V:
    base = V.get("official") or next(iter(V.values()))
    print(f"Workload: CEM {M['cem_opt_steps']} iterations x {M['batch_candidates']} candidates (batch), horizon "
          f"{M['horizon']} (frameskip {M['frameskip']}), top-{M['cem_topk']}; {M['n_tasks']} paired tasks, seed {M['seed']}; "
          f"warm-up {M['warmup_plans']} plan(s) discarded; {M['repetitions']}; {M['timer']}.\n")
    print("| variant | median plan (s) | mean ± sd (s) | p95 (s) | ms / CEM iter | candidates/s | peak GPU mem (MiB) | "
          "speedup vs official [95% CI] | success | same outcome as official |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, v in V.items():
        s, lo, hi = paired_speedup(base["latency_s"], v["latency_s"])
        same = np.mean(np.array(v["success"]) == np.array(base["success"]))
        print(f"| {name} | {v['latency_median_s']:.2f} | {v['latency_mean_s']:.2f} ± {v['latency_std_s']:.2f} | "
              f"{v['latency_p95_s']:.2f} | {v['per_cem_iter_median_ms']:.0f} | {v['candidates_per_s']:.0f} | "
              f"{v['peak_mem_alloc_mb']:.0f} | {s:.2f}x [{lo:.2f}, {hi:.2f}] | {v['success_rate']:.2f} "
              f"({sum(v['success'])}/{len(v['success'])}) | {same:.2f} |")
    print("\nReference rows (no planning): " + ", ".join(f"{k} success {r['success_rate']:.2f}" for k, r in R["reference"].items()))
    print("\nPhase share of one plan (instrumented pass):")
    for name, v in V.items():
        print(f"- {name}: " + ", ".join(f"{k} {100 * f:.1f}%" for k, f in v["phase_fraction"].items())
              + f"  (total {sum(v['phase_seconds_per_plan'].values()):.2f} s)")
    print("\nPlanned-action agreement with official:")
    for name, v in V.items():
        d = np.abs(np.array(v["actions"]) - np.array(base["actions"])).max()
        print(f"- {name}: max |action - official| {d:.2e}; mean final CEM loss {np.mean([t[-1] for t in v['cem_loss_trace']]):.4f}; "
              f"mean state distance to goal {np.mean(v['state_dist']):.1f}")
    if R.get("profile_top_ops"):
        print("\nGPU kernel time by kind, PyTorch Profiler over one plan (share of the kernel rows among the profiler's top rows):")
        kern = {}
        for name, rows in R["profile_top_ops"].items():
            k = [r for r in rows if not r["op"].startswith(("aten::", "cuda", "cudaLaunch"))]  # kernels only: aten:: rows repeat their kernels' time
            tot = sum(r["self_gpu_ms"] for r in k) or 1; g = {}
            for r in k:
                g[kind(r["op"])] = g.get(kind(r["op"]), 0) + r["self_gpu_ms"] / tot
            kern[name] = g
            print(f"- {name}: " + ", ".join(f"{a} {100 * b:.0f}%" for a, b in sorted(g.items(), key=lambda x: -x[1])) + f"  ({tot / 1e3:.2f} s of kernels)")
    ours = str(M.get("gpu_index_visible"))  # compute-app lists are empty inside the sandbox's pid namespace: judge by the other card
    busy = sorted({n for n, v in V.items() for st in (v["gpu_state_start"], v["gpu_state_end"]) for row in st["gpus"]
                   for i, util, mem in [[x.strip() for x in row.split(",")]] if i != ours and (float(util) > 10 or float(mem) > 100)})
    print("\nGPU exclusivity: runs only start on a card with no compute process and are stopped if another appears on it. "
          + ("The other card was busy during: " + ", ".join(busy) + "." if busy else "The other card was idle at the start and end of every variant."))

    # summary.png: one row per variant, read as a table with the plan time drawn as a bar
    names = list(V); n = len(names); best = min(V.values(), key=lambda v: v["latency_median_s"])
    fig = plt.figure(figsize=(11.6, 1.2 + 0.47 * n)); H = fig.get_figheight(); top, bot = 1 - 0.42 / H, 0.75 / H
    ax = fig.add_axes([0.375, bot, 0.30, top - bot]); row = blended_transform_factory(fig.transFigure, ax.transData)
    hbars(ax, [V[k]["latency_median_s"] for k in names], [GRAY if V[k] is base else BLUE for k in names],
          lambda v: f"{v:.1f} s", 1.2 * max(V[k]["latency_median_s"] for k in names))
    cols = [(0.012, "left", "Planner variant"), (0.375, "left", "Median time per plan"), (0.775, "right", "Speedup"),
            (0.89, "right", "Peak GPU memory"), (0.988, "right", "Tasks solved")]
    for x, ha, text in cols:
        fig.text(x, top + 0.07 / H, text, ha=ha, va="bottom", fontsize=10, color=MUTED)
    fig.add_artist(plt.Line2D([0.012, 0.988], [top, top], color=GRID, lw=0.8))
    for y, k in enumerate(names):
        v = V[k]; bold = "bold" if v is best else "normal"
        ax.text(0.012, y - 0.13, LABEL.get(k, k), transform=row, va="center", fontweight=bold)
        ax.text(0.012, y + 0.25, k, transform=row, va="center", fontsize=8, color=MUTED, family="Menlo")
        cells = [f"{base['latency_mean_s'] / v['latency_mean_s']:.2f}\u00d7", f"{v['peak_mem_alloc_mb'] / 1024:.1f} GiB",
                 f"{sum(v['success'])}/{len(v['success'])}"]
        for (x, ha, _), text in zip(cols[2:], cells):
            ax.text(x, y, text, transform=row, ha=ha, va="center", fontweight=bold)
    fig.text(0.012, 0.45 / H, f"{M['n_tasks']} paired PushT tasks, {M['cem_opt_steps']} CEM iterations of {M['batch_candidates']} candidates, "
             f"seed {M['seed']}. {M['warmup_plans']} warm-up plan discarded, then {M['repetitions']}. GPU-synchronized timer.",
             fontsize=10, color=INK2, va="center")
    fig.text(0.012, 0.2 / H, f"{gpu}, PyTorch {M['torch'].split('+')[0]}, CUDA {M['cuda']}. fp32, FP16 autocast where named. "
             "Speedup: mean plan time relative to the first row.", fontsize=10, color=INK2, va="center")
    fig.savefig(path.with_name("summary.png")); plt.close(fig)

print("\nRollout vs the official rollout on identical inputs (one task, all candidates):\n")
print("| compared with official | max abs feature diff | mean abs diff | max objective diff (spread across candidates) | CEM elite overlap | same best candidate |")
print("|---|---|---|---|---|---|")
for k, d in R["selfcheck"].items():
    if isinstance(d, dict):
        print(f"| {k} | {d['max_abs_diff']:.1e} | {d['mean_abs_diff']:.1e} | {d['loss_max_abs_diff']:.1e} ({d['loss_spread']:.2f}) | {d['topk_overlap']:.2f} | {d['best_candidate_same']} |")

par = {k: d for k, d in R["selfcheck"].items() if isinstance(d, dict) and "parent" in d}
if par:
    print("\nAgainst the same computation without the extra step (max abs feature diff): "
          + "; ".join(f"{k} vs {d['parent']} {d['max_abs_diff_parent']:.1e}" for k, d in par.items()) + ".")

if R.get("scaling"):
    rows = [r for r in R["scaling"] if not r.get("oom")]; names = list(dict.fromkeys(r["variant"] for r in rows))
    Ks = sorted({r["K"] for r in rows}); get = {(r["variant"], r["K"]): r for r in rows}
    print(f"\nOne rollout call (one CEM iteration's world-model cost) vs candidates K; horizon {M['horizon']}, "
          f"warm-up {rows[0]['warmup']} calls discarded, median of {rows[0]['reps']} calls, ms (peak GPU MiB):\n")
    print("| K | " + " | ".join(names) + " |"); print("|---|" + "---|" * len(names))
    for K in Ks:
        print(f"| {K} | " + " | ".join(f"{get[n, K]['median_ms']:.1f} ({get[n, K]['peak_mem_alloc_mb']:.0f})" if (n, K) in get else "OOM" for n in names) + " |")
    # scaling.png: where the released code spends a plan (left), and what one rollout call costs against K (right)
    fig = plt.figure(figsize=(11.6, 4.2)); H = fig.get_figheight()
    ph = base["phase_seconds_per_plan"]; tot = sum(ph.values())
    groups = [(f"{LABEL.get(base['variant'], base['variant'])}: wall time of one plan, {tot:.1f} s", [("Predictor rollouts", ph.get("predict", 0) / tot),
               ("Encoding observations", ph.get("encode_obs", 0) / tot),
               ("Sampling, objective, other", 1 - (ph.get("predict", 0) + ph.get("encode_obs", 0)) / tot)])]
    g = (kern if R.get("profile_top_ops") else {}).get(base["variant"])
    if g:
        short = {"GEMM (linear layers, attention matmul)": "Matrix multiply (GEMM)", "memcpy": "Memory copy"}
        groups.append(("GPU kernel time, PyTorch Profiler", [(short.get(k, k[0].upper() + k[1:]), f) for k, f in
                                                             sorted(g.items(), key=lambda x: -x[1]) if f >= 0.01]))
    rows_total = sum(len(r) + 1.4 for _, r in groups); y1 = 1 - 0.1 / H; unit = (y1 - 0.3 / H) / rows_total
    for title, rws in groups:
        fig.text(0.012, y1 - 0.55 * unit, title, fontsize=10, color=MUTED, va="center")
        ax = fig.add_axes([0.215, y1 - (len(rws) + 1) * unit, 0.25, len(rws) * unit])
        hbars(ax, [f for _, f in rws], [GRAY] * len(rws), lambda f: f"{100 * f:.1f}%", 1.25)
        for y, (name, _) in enumerate(rws):
            ax.text(0.012, y, name, transform=blended_transform_factory(fig.transFigure, ax.transData), va="center")
        y1 -= (len(rws) + 1.4) * unit

    x0 = 0.53; ax = fig.add_axes([x0 + 0.045, 0.6 / H, 0.265, 1 - 1.15 / H])
    fig.text(x0, 1 - 0.1 / H - 0.55 * unit, f"Median time of one rollout call, horizon {M['horizon']}", fontsize=10, color=MUTED, va="center")
    others = [k for k in names if k != base["variant"]]
    color = {k: plt.matplotlib.colors.LinearSegmentedColormap.from_list("r", RAMP)(i / max(len(others) - 1, 1))
             for i, k in enumerate(others)} | {base["variant"]: GRAY}
    kmax = max(Ks); kref = [k for k in Ks if k >= 16] or Ks; low = min(get[k, kref[0]]["median_ms"] for k in names if (k, kref[0]) in get)
    ax.plot([kref[0], kmax], [0.4 * low, 0.4 * low * kmax / kref[0]], color=AXIS, lw=1)  # slope-1 guide under the curves
    ax.text(kmax * 1.12, 0.4 * low * kmax / kref[0], "linear in K", color=MUTED, fontsize=10, va="center")
    ends = []
    for k in names:
        xs = [K for K in Ks if (k, K) in get]; ys = [get[k, K]["median_ms"] for K in xs]
        ax.plot(xs, ys, color=color[k], lw=2, solid_capstyle="round", marker="o", ms=5, mec="white", mew=1.2, clip_on=False)
        t = f"{ys[-1] / 1000:.2f} s" if ys[-1] >= 1000 else f"{ys[-1]:.0f} ms"
        ends.append((ax.text(xs[-1] * 1.12, ys[-1], SHORT.get(k, LABEL.get(k, k)), fontsize=10, va="center"), t))
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlim(min(Ks) * 0.85, kmax * 1.05)
    ticks = [K for K in Ks if K & (K - 1) == 0][::2]; ax.set_xticks(ticks, [str(K) for K in ticks]); ax.minorticks_off()
    ax.set_yticks([10, 100, 1000], ["10 ms", "100 ms", "1 s"]); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.tick_params(length=0, pad=6); ax.set_xlabel("Candidates per rollout call, K")
    for label, t in ends:  # the value at the largest K follows its series name, in the quieter ink
        ax.annotate(t, (1, 0.5), xycoords=label, xytext=(7, 0), textcoords="offset points", fontsize=10, color=MUTED, va="center")
    fig.savefig(path.with_name("scaling.png")); plt.close(fig)
