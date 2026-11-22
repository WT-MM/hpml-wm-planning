#!/usr/bin/env python
"""results.json -> markdown summary (stdout), summary.png and scaling.png next to it.
Usage: summarize.py runs/NAME/results.json"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = Path(sys.argv[1]); R = json.loads(path.read_text()); M, V = R["meta"], R["variants"]
rng = np.random.default_rng(0)
gpu = M["gpu"].replace("NVIDIA GeForce ", "")


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

if V:
    base = V.get("official") or next(iter(V.values()))
    print(f"Workload: CEM {M['cem_opt_steps']} iterations x {M['batch_candidates']} candidates (batch), horizon "
          f"{M['horizon']} (frameskip {M['frameskip']}), top-{M['cem_topk']}; {M['n_tasks']} paired tasks, seed {M['seed']}; "
          f"warm-up {M['warmup_plans']} plan(s) discarded; {M['repetitions']}; {M['timer']}.\n")
    print("| variant | median plan (s) | mean ± sd (s) | p95 (s) | ms / CEM iter | candidates/s | peak GPU mem (MB) | "
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
        for name, rows in R["profile_top_ops"].items():
            k = [r for r in rows if not r["op"].startswith(("aten::", "cuda", "cudaLaunch"))]  # kernels only: aten:: rows repeat their kernels' time
            tot = sum(r["self_gpu_ms"] for r in k) or 1; g = {}
            for r in k:
                g[kind(r["op"])] = g.get(kind(r["op"]), 0) + r["self_gpu_ms"] / tot
            print(f"- {name}: " + ", ".join(f"{a} {100 * b:.0f}%" for a, b in sorted(g.items(), key=lambda x: -x[1])) + f"  ({tot / 1e3:.2f} s of kernels)")
    ours = str(M.get("gpu_index_visible"))  # compute-app lists are empty inside the sandbox's pid namespace: judge by the other card
    busy = sorted({n for n, v in V.items() for st in (v["gpu_state_start"], v["gpu_state_end"]) for row in st["gpus"]
                   for i, util, mem in [[x.strip() for x in row.split(",")]] if i != ours and (float(util) > 10 or float(mem) > 100)})
    print("\nGPU exclusivity: runs only start on a card with no compute process and are stopped if another appears on it. "
          + ("The other card was busy during: " + ", ".join(busy) + "." if busy else "The other card was idle at the start and end of every variant."))

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6)); names = list(V)
    ax[0].boxplot([V[n]["latency_s"] for n in names], tick_labels=names)
    ax[0].set_ylabel("planning latency per task (s)"); ax[0].set_ylim(bottom=0); ax[0].tick_params(axis="x", labelsize=8)
    ax[0].set_title(f"CEM plan latency, {M['n_tasks']} paired tasks, {gpu}", fontsize=10)
    bottom = np.zeros(len(names))
    for k in base["phase_seconds_per_plan"]:
        vals = np.array([V[n]["phase_seconds_per_plan"].get(k, 0) for n in names])
        ax[1].bar(names, vals, bottom=bottom, label=k.split("(")[0]); bottom += vals
    ax[1].set_ylabel("seconds per plan"); ax[1].legend(fontsize=8); ax[1].tick_params(axis="x", labelsize=8)
    ax[1].set_title("Where the time goes (instrumented pass)", fontsize=10)
    fig.tight_layout(); fig.savefig(path.with_name("summary.png"), dpi=160)

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
          f"warm-up {rows[0]['warmup']} calls discarded, median of {rows[0]['reps']} calls, ms (peak GPU MB):\n")
    print("| K | " + " | ".join(names) + " |"); print("|---|" + "---|" * len(names))
    for K in Ks:
        print(f"| {K} | " + " | ".join(f"{get[n, K]['median_ms']:.1f} ({get[n, K]['peak_mem_alloc_mb']:.0f})" if (n, K) in get else "OOM" for n in names) + " |")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for n in names:
        ax.plot([K for K in Ks if (n, K) in get], [get[n, K]["median_ms"] for K in Ks if (n, K) in get], marker="o", ms=3, label=n)
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("candidates per rollout call (K)")
    ax.set_ylabel("latency (ms)"); ax.legend(fontsize=8); ax.set_title(f"World-model rollout cost vs K, {gpu}", fontsize=10)
    fig.tight_layout(); fig.savefig(path.with_name("scaling.png"), dpi=160)
