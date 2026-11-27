#!/usr/bin/env python
"""Multi-GPU candidate-split benchmark for DINO-WM CEM planning on PushT.

Because rollout cost is strictly linear in candidate count K >= 8 on an RTX 5090, splitting the K
candidates across 2 GPUs (replicated frozen world model, broadcast cached start/goal latents once
per plan, concurrent threaded rollouts, gather only K/2 scalar losses per CEM iteration) approaches
linear multi-GPU scaling with negligible PCIe communication overhead.
"""
import argparse, copy, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def split_bounds(K, n_devices):
    """Balanced contiguous slice bounds splitting K candidates across n_devices."""
    if n_devices < 1 or K < 1:
        raise ValueError(f"invalid K={K}, n_devices={n_devices}")
    base, rem = divmod(K, n_devices)
    bounds, off = [], 0
    for d in range(n_devices):
        sz = base + (1 if d < rem else 0)
        bounds.append((off, off + sz))
        off += sz
    return bounds


def run_benchmark(args):
    import torch
    sys.path.insert(0, str(Path(__file__).parent))
    import planbench as pb

    devices = [torch.device(d.strip()) for d in args.devices.split(",") if d.strip()]
    if len(devices) < 2 and devices[0].type == "cuda":
        raise ValueError("multigpu_bench requires at least 2 devices (e.g. cuda:0,cuda:1)")

    S = pb.load(args, devices[0])
    from utils import move_to_device
    wm0, pl = S["wm"], S["planner"]
    tr = pl.preprocessor.transform_obs
    N, topk, steps, H, A, n = args.num_samples, args.topk, args.opt_steps, pl.horizon, pl.action_dim, args.n_tasks
    seed_of = lambda i: args.seed * 100003 + i

    # Replicate frozen world model across all target devices
    wms = [wm0] + [copy.deepcopy(wm0).to(d) for d in devices[1:]]
    bounds = split_bounds(N, len(devices))

    def sync_all():
        for d in devices:
            pb.sync(d)

    def plan_single_gpu(var0, i):
        o0 = move_to_device(tr({k: v[i:i + 1] for k, v in S["obs_0"].items()}), devices[0])
        og = move_to_device(tr({k: v[i:i + 1] for k, v in S["obs_g"].items()}), devices[0])
        var0.new_plan()
        with torch.no_grad():
            zg = wm0.encode_obs(og)
        torch.manual_seed(seed_of(i))
        mu, sigma = torch.zeros(H, A, device=devices[0]), torch.ones(H, A, device=devices[0])
        obs_rep = {k: v.expand(N, *v.shape[1:]) for k, v in o0.items()}
        zg_rep = {k: v.expand(N, *v.shape[1:]) for k, v in zg.items()}
        for _ in range(steps):
            act = torch.randn(N, H, A).to(devices[0]) * sigma + mu
            act[0] = mu
            with torch.no_grad():
                zo, _ = wm0.rollout(obs_rep, act)
            loss = pl.objective_fn(zo, zg_rep)
            top = torch.argsort(loss)[:topk]
            mu, sigma = act[top].mean(0), act[top].std(0)
        return mu

    def plan_multi_gpu(vars_list, pool, i):
        o0_0 = move_to_device(tr({k: v[i:i + 1] for k, v in S["obs_0"].items()}), devices[0])
        og_0 = move_to_device(tr({k: v[i:i + 1] for k, v in S["obs_g"].items()}), devices[0])
        for v in vars_list:
            v.new_plan()
        with torch.no_grad():
            z0_0 = wm0.encode_obs(o0_0)
            zg_0 = wm0.encode_obs(og_0)
        # Broadcast tiny cached start & goal latents once per plan
        o0_reps, zg_reps = [], []
        for d_idx, (dev, var_d, (a, b)) in enumerate(zip(devices, vars_list, bounds)):
            sz = b - a
            var_d.z_cache["z"] = {k: v.to(dev, non_blocking=True) for k, v in z0_0.items()}
            o0_d = {k: v.to(dev, non_blocking=True).expand(sz, *v.shape[1:]) for k, v in o0_0.items()}
            zg_d = {k: v.to(dev, non_blocking=True).expand(sz, *v.shape[1:]) for k, v in zg_0.items()}
            o0_reps.append(o0_d); zg_reps.append(zg_d)

        torch.manual_seed(seed_of(i))
        mu, sigma = torch.zeros(H, A, device=devices[0]), torch.ones(H, A, device=devices[0])

        def worker(d_idx, act_slice):
            dev = devices[d_idx]
            if dev.type == "cuda":
                torch.cuda.set_device(dev)
            a_d = act_slice.to(dev, non_blocking=True)
            with torch.no_grad():
                zo, _ = wms[d_idx].rollout(o0_reps[d_idx], a_d)
                l_d = pl.objective_fn(zo, zg_reps[d_idx])
            return l_d.to(devices[0], non_blocking=True)

        for _ in range(steps):
            act = torch.randn(N, H, A).to(devices[0]) * sigma + mu
            act[0] = mu
            futs = [pool.submit(worker, d_idx, act[a:b]) for d_idx, (a, b) in enumerate(bounds)]
            loss = torch.cat([f.result() for f in futs], dim=0)
            top = torch.argsort(loss)[:topk]
            mu, sigma = act[top].mean(0), act[top].std(0)
        return mu

    variant = args.variant if devices[0].type == "cuda" else "cache_sdpa_unmasked_reuse"
    R = {"args": vars(args), "devices": [str(d) for d in devices], "variant": variant, "bounds": bounds}

    # Enter Variant on each GPU independently (each gets its own ReusePredict instance)
    vars_list = []
    for wm_d in wms:
        pb.Variant._reuse.clear()  # ensure each device builds its own ReusePredict bound to wm_d
        v = pb.Variant(wm_d, variant)
        v.__enter__()
        vars_list.append(v)

    try:
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            # Runnable check: 1-GPU vs multi-GPU on task 0
            mu_1 = plan_single_gpu(vars_list[0], 0)
            mu_m = plan_multi_gpu(vars_list, pool, 0)
            chk_diff = float((mu_1 - mu_m).abs().max().item())
            R["selfcheck_1gpu_vs_mgpu_max_action_diff"] = chk_diff
            print(f"selfcheck 1gpu_vs_mgpu max_action_diff={chk_diff:.3e}", flush=True)

            # Time 1-GPU baseline vs multi-GPU on paired tasks
            for label, fn in (("1gpu", lambda i: plan_single_gpu(vars_list[0], i)),
                              (f"{len(devices)}gpu", lambda i: plan_multi_gpu(vars_list, pool, i))):
                for _ in range(args.warmup):
                    fn(0)
                for d in devices:
                    if d.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(d)
                lats, mus = [], []
                for i in range(n):
                    sync_all(); t0 = time.perf_counter()
                    mu = fn(i)
                    sync_all(); lats.append(time.perf_counter() - t0)
                    mus.append(mu[None])
                out_dir = Path(args.out).resolve() / label
                out_dir.mkdir(parents=True, exist_ok=True)
                import os
                cwd = os.getcwd(); os.chdir(out_dir)
                logs, succ, _, e_states = S["evaluator"].eval_actions(torch.cat(mus, 0), filename="rollouts")
                os.chdir(cwd)
                peak_mb = [torch.cuda.max_memory_allocated(d) / 2**20 if d.type == "cuda" else 0.0 for d in devices]
                R[label] = {
                    "latency_s": lats,
                    "latency_median_s": float(np.median(lats)),
                    "latency_p95_s": float(np.percentile(lats, 95)),
                    "success_rate": float(np.mean(succ)),
                    "peak_mem_per_gpu_mb": peak_mb,
                    "mean_state_dist": float(np.mean(np.linalg.norm(S["state_g"] - e_states[:, -1], axis=1))),
                }
                print(f"[{label}] median={np.median(lats):.3f}s p95={np.percentile(lats, 95):.3f}s "
                      f"succ={np.mean(succ):.3f} peak_mb={[round(x) for x in peak_mb]}", flush=True)
    finally:
        for v in vars_list:
            v.__exit__(None, None, None)
        S["env"].close()

    m_key = f"{len(devices)}gpu"
    R["speedup_mgpu_vs_1gpu"] = float(np.mean(R["1gpu"]["latency_s"]) / np.mean(R[m_key]["latency_s"]))
    print(f"==> {m_key} vs 1gpu speedup: {R['speedup_mgpu_vs_1gpu']:.2f}x", flush=True)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "multigpu.json").write_text(json.dumps(R, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="fp16_unmasked_reuse")
    ap.add_argument("--devices", default="cuda:0,cuda:1")
    ap.add_argument("--model-name", default="pusht")
    ap.add_argument("--model-epoch", default="latest")
    ap.add_argument("--n-tasks", type=int, default=16)
    ap.add_argument("--goal-H", type=int, default=5)
    ap.add_argument("--num-samples", type=int, default=300)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--opt-steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--n-plot", type=int, default=0)
    ap.add_argument("--eval-mode", action="store_true")
    args = ap.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
