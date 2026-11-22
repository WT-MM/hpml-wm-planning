#!/usr/bin/env python
"""Add the variants of a second run (same tasks, seed, GPU and code path; a separate process) to a main run's results.
Usage: merge_runs.py main/results.json extra/results.json out/results.json
A variant present in both is the anchor: its two medians show how far the processes drifted apart."""
import json, sys
from pathlib import Path

main, extra, out = (Path(p) for p in sys.argv[1:4])
R, X = json.loads(main.read_text()), json.loads(extra.read_text())
assert all(R["meta"][k] == X["meta"][k] for k in ("gpu", "gpu_index_visible", "driver", "torch", "n_tasks", "seed", "batch_candidates",
                                                    "cem_opt_steps", "cem_topk", "horizon", "dino_wm_commit")), "runs are not comparable"
new = [n for n in X["variants"] if n not in R["variants"]]
R["meta"]["merged"] = {"main_run": main.parent.name, "extra_run": extra.parent.name, "extra_variants": new,
                       "anchor_median_s": {n: [R["variants"][n]["latency_median_s"], X["variants"][n]["latency_median_s"]]
                                           for n in X["variants"] if n in R["variants"]}}
R["meta"].setdefault("variant_options", {}).update({n: X["meta"].get("variant_options", {}).get(n, {}) for n in new})
for n in new:
    R["variants"][n] = {**X["variants"][n], "from_run": extra.parent.name}
    R["selfcheck"][n] = X["selfcheck"][n]
    if n in X.get("profile_top_ops", {}):
        R.setdefault("profile_top_ops", {})[n] = X["profile_top_ops"][n]
order = [n for n in pb_order if n in R["variants"]] if (pb_order := ["official", "cache", "cache_sdpa", "cache_sdpa_tf32", "cache_sdpa_fp16",
                                                                    "fp16_compile", "fp16_reuse", "fp16_reuse_compile"]) else []
order += [n for n in R["variants"] if n not in order]
R["variants"] = {n: R["variants"][n] for n in order}
if "profile_top_ops" in R:
    R["profile_top_ops"] = {n: R["profile_top_ops"][n] for n in order if n in R["profile_top_ops"]}
out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(R))
print("merged", new, "anchor", R["meta"]["merged"]["anchor_median_s"])
