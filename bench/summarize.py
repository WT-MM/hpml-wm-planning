#!/usr/bin/env python
"""results.json -> markdown summary (stdout) and summary.png next to it.  Usage: summarize.py runs/NAME/results.json"""
import json, sys
from pathlib import Path
import numpy as np

path = Path(sys.argv[1]); R = json.loads(path.read_text()); M, V = R["meta"], R["variants"]
base = V.get("official") or next(iter(V.values()))
rng = np.random.default_rng(0)


def paired_speedup(a, b):
    """Speedup of b over a: ratio of mean latencies, 95% CI by bootstrap over the paired tasks."""
    a, b = np.array(a), np.array(b); idx = rng.integers(0, len(a), (10000, len(a)))
    boot = a[idx].mean(1) / b[idx].mean(1)
    return a.mean() / b.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


print(f"Hardware: {M['gpu']} (driver {M['driver']}), {M['cpu']}; torch {M['torch']}, CUDA {M['cuda']}, {M['backend']}.")
print(f"Workload: {M['model']}; CEM {M['cem_opt_steps']} iterations x {M['batch_candidates']} candidates, horizon "
      f"{M['horizon']} (frameskip {M['frameskip']}), top-{M['cem_topk']}; context {M['num_hist']} frames x "
      f"{M['tokens_per_frame']} tokens; {M['n_tasks']} paired tasks, seed {M['seed']}; warm-up {M['warmup_plans']} "
      f"plan(s) discarded; {M['repetitions']}; {M['timer']}.\n")
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
print("\nRollout equivalence to the official rollout (max abs feature difference, eval mode): "
      + ", ".join(f"{k} {d['max_abs_diff']:.1e}" for k, d in R["selfcheck"].items())
      + f"; mean |feature| {next(iter(R['selfcheck'].values()))['ref_abs_mean']:.2f}")
for name, v in V.items():
    d = np.abs(np.array(v["actions"]) - np.array(base["actions"])).max()
    print(f"- {name}: max |planned action - official| {d:.2e}; final CEM loss mean {np.mean([t[-1] for t in v['cem_loss_trace']]):.4f}")
for name, rows in R.get("profile_top_ops", {}).items():
    print(f"\nTop GPU ops, {name} (one plan): " + "; ".join(f"{r['op']} {100 * r['self_gpu_frac']:.0f}%" for r in rows[:6]))
ours = str(M.get("gpu_index_visible"))  # compute-app lists are empty inside the sandbox's pid namespace, so judge by the other card
busy = sorted({n for n, v in V.items() for st in (v["gpu_state_start"], v["gpu_state_end"]) for row in st["gpus"]
               for i, util, mem in [[x.strip() for x in row.split(",")]] if i != ours and (float(util) > 10 or float(mem) > 100)})
print("\nGPU exclusivity: runs only start on a card with no compute process and are stopped if another appears on it. "
      + ("The other card was busy during: " + ", ".join(busy) + "." if busy else "The other card was idle at the start and end of every variant."))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
names = list(V); keys = list(base["phase_seconds_per_plan"])
ax[0].boxplot([V[n]["latency_s"] for n in names], tick_labels=names, showfliers=True)
ax[0].set_ylabel("planning latency per task (s)"); ax[0].set_ylim(bottom=0); ax[0].tick_params(axis="x", labelsize=8)
ax[0].set_title(f"CEM plan latency, {M['n_tasks']} paired tasks, {M['gpu'].replace('NVIDIA GeForce ', '')}", fontsize=10)
bottom = np.zeros(len(names))
for k in keys:
    vals = np.array([V[n]["phase_seconds_per_plan"].get(k, 0) for n in names])
    ax[1].bar(names, vals, bottom=bottom, label=k.split("(")[0]); bottom += vals
ax[1].set_ylabel("seconds per plan"); ax[1].legend(fontsize=8); ax[1].tick_params(axis="x", labelsize=8)
ax[1].set_title("Where the time goes (instrumented pass)", fontsize=10)
fig.tight_layout(); fig.savefig(path.with_name("summary.png"), dpi=160)
