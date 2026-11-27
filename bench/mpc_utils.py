"""Pure-Python/NumPy utilities for closed-loop MPC evaluation: leak-free episode splits,
deterministic task sampling, planner spec & shrinking-budget schedules, and multi-seed summaries.
"""
import random
import numpy as np

HELDOUT_TRAIN_N = 500


def episode_split(n_train, n_val, heldout_n=HELDOUT_TRAIN_N):
    """Deterministic episode-level split so held-out eval episodes are never used to train G.
    Returns dict mapping split name -> list of episode indices within that dataset."""
    if heldout_n < 0 or heldout_n >= n_train:
        raise ValueError(f"heldout_n={heldout_n} invalid for n_train={n_train}")
    cut = n_train - heldout_n
    return {
        "train_G": list(range(0, cut)),
        "heldout_train": list(range(cut, n_train)),
        "val": list(range(0, n_val)),
    }


def sample_dset_tasks(seq_lengths_by_split, task_source, n_tasks, traj_len, seed, heldout_n=HELDOUT_TRAIN_N):
    """Deterministically sample n_tasks of (split_key, episode_idx, offset) with length >= traj_len.
    task_source in {'val', 'heldout', 'val+heldout', 'train_G'}."""
    n_train = len(seq_lengths_by_split["train"])
    n_val = len(seq_lengths_by_split["valid"])
    sp = episode_split(n_train, n_val, heldout_n=heldout_n)
    if task_source == "val":
        pool = [("valid", i) for i in sp["val"]]
    elif task_source == "heldout":
        pool = [("train", i) for i in sp["heldout_train"]]
    elif task_source == "val+heldout":
        pool = [("valid", i) for i in sp["val"]] + [("train", i) for i in sp["heldout_train"]]
    elif task_source == "train_G":
        pool = [("train", i) for i in sp["train_G"]]
    else:
        raise ValueError(f"unknown task_source: {task_source}")

    eligible = [(k, i, seq_lengths_by_split[k][i]) for k, i in pool if seq_lengths_by_split[k][i] >= traj_len]
    if not eligible:
        raise ValueError(f"no episodes in {task_source} have length >= {traj_len}")

    rng = random.Random(seed)
    tasks = []
    for _ in range(n_tasks):
        k, i, T = eligible[rng.randrange(len(eligible))]
        off = rng.randint(0, T - traj_len)
        tasks.append({"split": k, "episode": int(i), "offset": int(off)})
    return tasks


def parse_planner_spec(spec):
    """Parse planner spec '<kind>:<variant>' into dict.
    Supported kinds:
      cem             : full num_samples CEM
      few<N>          : fixed N samples, topk scaled proportionally
      shrink<N0>_<N1> : linearly anneal candidates from N0 to N1 across opt_steps
      gd              : gradient-based planning (GDPlanner / Adam-SGD through wm.rollout)
    """
    if ":" not in spec:
        raise ValueError(f"planner spec must be '<kind>:<variant>', got {spec!r}")
    kind, variant = spec.split(":", 1)
    if kind == "cem":
        return {"name": spec, "kind": "cem", "variant": variant}
    if kind == "gd":
        return {"name": spec, "kind": "gd", "variant": variant}
    if kind.startswith("few"):
        n = int(kind[3:])
        if n < 2:
            raise ValueError(f"few<N> requires N >= 2, got {spec!r}")
        return {"name": spec, "kind": "few", "N": n, "variant": variant}
    if kind.startswith("shrink") and "_" in kind:
        a, b = kind[6:].split("_", 1)
        n0, n1 = int(a), int(b)
        if n0 < 2 or n1 < 2:
            raise ValueError(f"shrink requires N0, N1 >= 2, got {spec!r}")
        return {"name": spec, "kind": "shrink", "N0": n0, "N1": n1, "variant": variant}
    raise ValueError(f"unsupported planner kind in {spec!r}")


def iter_budget(parsed, it, opt_steps, default_N, default_topk):
    """Return (N_it, topk_it) for CEM iteration it in [0, opt_steps)."""
    ratio = default_topk / default_N
    kind = parsed["kind"]
    if kind == "cem":
        return default_N, default_topk
    if kind == "few":
        n = parsed["N"]
        return n, max(2, min(n - 1, round(n * ratio)))
    if kind == "shrink":
        frac = 0.0 if opt_steps <= 1 else it / (opt_steps - 1)
        n = int(round(parsed["N0"] + frac * (parsed["N1"] - parsed["N0"])))
        n = max(2, n)
        return n, max(2, min(n - 1, round(n * ratio)))
    raise ValueError(f"iter_budget does not apply to kind={kind!r}")


def paired_speedup_ci(base_lat, lat, n_boot=5000, seed=0):
    """Ratio of mean latencies with 95% paired bootstrap CI."""
    a = np.asarray(base_lat, dtype=np.float64)
    b = np.asarray(lat, dtype=np.float64)
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("base_lat and lat must be same non-zero length")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    boot = a[idx].mean(axis=1) / b[idx].mean(axis=1)
    return float(a.mean() / b.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def aggregate_mpc_results(runs_by_planner, base_name=None):
    """Aggregate per-seed episode metrics across all seeds for each planner.
    runs_by_planner: {planner_name: list of per-seed dicts with episode_plan_s, step_plan_s,
                      success, final_state_dist, steps_to_success, peak_mem_alloc_mb, total_candidates}."""
    names = list(runs_by_planner.keys())
    if not names:
        return {}
    base_name = base_name or names[0]
    base_ep = np.concatenate([np.asarray(r["episode_plan_s"], dtype=np.float64) for r in runs_by_planner[base_name]])

    out = {}
    for name, seed_runs in runs_by_planner.items():
        ep_lat = np.concatenate([np.asarray(r["episode_plan_s"], dtype=np.float64) for r in seed_runs])
        st_lat = np.concatenate([np.asarray(r["step_plan_s"], dtype=np.float64) for r in seed_runs])
        succ = np.concatenate([np.asarray(r["success"], dtype=np.float64) for r in seed_runs])
        dist = np.concatenate([np.asarray(r["final_state_dist"], dtype=np.float64) for r in seed_runs])
        tot_cand = sum(int(r.get("total_candidates", 0)) for r in seed_runs)
        sp, lo, hi = paired_speedup_ci(base_ep, ep_lat)
        seed_succ = [float(np.mean(r["success"])) for r in seed_runs]
        out[name] = {
            "n_seeds": len(seed_runs),
            "n_tasks_total": int(len(ep_lat)),
            "episode_median_s": float(np.median(ep_lat)),
            "episode_mean_s": float(np.mean(ep_lat)),
            "episode_p95_s": float(np.percentile(ep_lat, 95)),
            "step_median_s": float(np.median(st_lat)),
            "step_p95_s": float(np.percentile(st_lat, 95)),
            "candidates_per_s": float(tot_cand / max(float(ep_lat.sum()), 1e-9)),
            "peak_mem_alloc_mb": float(max(r["peak_mem_alloc_mb"] for r in seed_runs)),
            "speedup_vs_base": sp,
            "speedup_ci95": [lo, hi],
            "success_rate": float(np.mean(succ)),
            "success_rate_seed_std": float(np.std(seed_succ, ddof=1)) if len(seed_succ) > 1 else 0.0,
            "mean_final_state_dist": float(np.mean(dist)),
        }
    return out
