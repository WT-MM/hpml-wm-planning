#!/usr/bin/env python
"""Upload a finished benchmark run to Weights & Biases: one W&B run per variant (grouped by the run folder's name)
plus a `summary` run holding the tables and figures.
Usage: wandb_upload.py results/NAME/results.json --entity ENTITY [--project hpml-wm-planning]"""
import argparse, json
from pathlib import Path
import numpy as np
import wandb

ap = argparse.ArgumentParser()
ap.add_argument("results"); ap.add_argument("--entity", required=True); ap.add_argument("--project", default="hpml-wm-planning")
args = ap.parse_args()
path = Path(args.results); R = json.loads(path.read_text()); M, V = R["meta"], R["variants"]
group = path.parent.name; base = V["official"]
# The numbers were measured on the GPU box; this script only uploads them, so keep the uploading machine out of the record.
quiet = wandb.Settings(disable_git=True, disable_code=True, x_disable_meta=True, x_disable_stats=True, host="rtx5090-workstation")


def start(name, job_type, extra=None):
    return wandb.init(entity=args.entity, project=args.project, group=group, name=f"{group}/{name}", job_type=job_type,
                      config={**M, **(extra or {})}, settings=quiet,
                      notes=f"Uploaded after the run from results/{group}/results.json. Hardware and timing conditions are in the config.")


for name, v in V.items():
    run = start(name, "benchmark", {"variant": name})
    run.define_metric("task"); run.define_metric("task/*", step_metric="task")
    run.define_metric("K"); run.define_metric("rollout/*", step_metric="K")
    for i, lat in enumerate(v["latency_s"]):
        run.log({"task": i, "task/plan_latency_s": lat, "task/success": int(v["success"][i]), "task/state_dist": v["state_dist"][i],
                 "task/final_cem_loss": v["cem_loss_trace"][i][-1]})
    for r in R.get("scaling", []):
        if r["variant"] == name and not r.get("oom"):
            run.log({"K": r["K"], "rollout/latency_ms": r["median_ms"], "rollout/ms_per_candidate": r["ms_per_candidate"],
                     "rollout/peak_mem_mb": r["peak_mem_alloc_mb"]})
    run.summary.update({
        "plan_latency_median_s": v["latency_median_s"], "plan_latency_mean_s": v["latency_mean_s"], "plan_latency_std_s": v["latency_std_s"],
        "plan_latency_p95_s": v["latency_p95_s"], "ms_per_cem_iter": v["per_cem_iter_median_ms"], "candidates_per_s": v["candidates_per_s"],
        "peak_gpu_mem_mb": v["peak_mem_alloc_mb"], "speedup_vs_official": base["latency_mean_s"] / v["latency_mean_s"],
        "success_rate": v["success_rate"], "mean_state_dist": float(np.mean(v["state_dist"])),
        "mean_final_cem_loss": float(np.mean([t[-1] for t in v["cem_loss_trace"]])),
        **{f"phase_fraction/{k.split('(')[0]}": f for k, f in v["phase_fraction"].items()}})
    img = path.parent / name / "rollouts.png"
    if img.exists():
        run.log({"rollouts": wandb.Image(str(img), caption=f"{name}: planned rollouts executed in the simulator")})
    run.finish()

run = start("summary", "summary")
cols = ["variant", "median plan (s)", "mean (s)", "sd (s)", "p95 (s)", "ms / CEM iter", "candidates/s", "peak GPU mem (MiB)", "speedup vs official", "success"]
run.log({"latency": wandb.Table(columns=cols, data=[
    [n, v["latency_median_s"], v["latency_mean_s"], v["latency_std_s"], v["latency_p95_s"], v["per_cem_iter_median_ms"], v["candidates_per_s"],
     v["peak_mem_alloc_mb"], base["latency_mean_s"] / v["latency_mean_s"], f"{sum(v['success'])}/{len(v['success'])}"] for n, v in V.items()])})
sc = [r for r in R.get("scaling", []) if not r.get("oom")]
if sc:
    run.log({"rollout_scaling": wandb.Table(columns=["variant", "K", "median ms", "ms per candidate", "peak GPU mem (MiB)"],
                                            data=[[r["variant"], r["K"], r["median_ms"], r["ms_per_candidate"], r["peak_mem_alloc_mb"]] for r in sc])})
chk = {k: d for k, d in R["selfcheck"].items() if isinstance(d, dict)}
run.log({"selfcheck_vs_official": wandb.Table(
    columns=["compared", "max abs feature diff", "mean abs diff", "max objective diff", "objective spread", "elite overlap", "same best candidate"],
    data=[[k, d["max_abs_diff"], d["mean_abs_diff"], d["loss_max_abs_diff"], d["loss_spread"], d["topk_overlap"], bool(d["best_candidate_same"])]
          for k, d in chk.items()])})
for fig in ("summary.png", "scaling.png"):
    if (path.parent / fig).exists():
        run.log({fig.split(".")[0]: wandb.Image(str(path.parent / fig))})
run.summary.update({f"reference/{k}_success_rate": r["success_rate"] for k, r in R["reference"].items()})
run.finish()
print(f"https://wandb.ai/{args.entity}/{args.project}")
