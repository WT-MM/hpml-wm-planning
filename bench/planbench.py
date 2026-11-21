#!/usr/bin/env python
"""Latency, task-success and profile benchmark for DINO-WM CEM planning on PushT.

The world model, CEM planner, objective, simulator and evaluator are the official DINO-WM code (unmodified, pinned
commit) with the released PushT checkpoint. This file only adds: paired task sampling, timing, and the variants below,
which are applied by patching at run time so every variant plans the same tasks with the same candidate noise.

  official        as released: every CEM iteration re-encodes the (identical) start frame for all candidates
  cache           encode the start frame once per plan and expand the latents ("cached encodings" strong baseline)
  cache_sdpa      cache + predictor attention through F.scaled_dot_product_attention (fused kernel, same frame mask)
  cache_sdpa_fp16 cache_sdpa + fp16 autocast inside the world-model rollout (changes numerics)

One plan = one task: opt_steps CEM iterations x num_samples candidate action sequences x horizon steps.
"""
import argparse, json, os, random, subprocess, sys, time, types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

W = Path(os.environ["HPML_W"])  # folder holding dino_wm/, ckpt/outputs/, data/pusht_noise/
DINO_WM = W / "dino_wm"
VARIANTS = {  # name -> (cache, sdpa, amp dtype)
    "official": (False, False, None),
    "cache": (True, False, None),
    "cache_sdpa": (True, True, None),
    "cache_sdpa_fp16": (True, True, torch.float16),
}


# ---------------------------------------------------------------- variants
def rollout_tail(wm, z, action):
    """The prediction loop of VWorldModel.rollout, verbatim, starting from already-encoded z."""
    t, inc = 0, 1
    while t < action.shape[1]:
        z_pred = wm.predict(z[:, -wm.num_hist:])
        z_new = wm.replace_actions_from_z(z_pred[:, -inc:, ...], action[:, t:t + inc, :])
        z = torch.cat([z, z_new], dim=1)
        t += inc
    z = torch.cat([z, wm.predict(z[:, -wm.num_hist:])[:, -1:, ...]], dim=1)
    return wm.separate_emb(z)[0], z


def make_cached_rollout(wm, cache):
    """rollout() for a planner that passes the same start observation repeated B times: encode one copy, expand."""
    def rollout(obs_0, act):
        if "z" not in cache:
            assert torch.equal(obs_0["visual"][0], obs_0["visual"][-1]), "cached rollout needs identical rows"
            cache["z"] = wm.encode_obs({k: v[:1] for k, v in obs_0.items()})
        B, n = act.shape[0], obs_0["visual"].shape[1]
        vis = cache["z"]["visual"].expand(B, -1, -1, -1)
        prop, act_emb = cache["z"]["proprio"].expand(B, -1, -1), wm.encode_act(act[:, :n])
        if wm.concat_dim == 0:
            z = torch.cat([vis, prop.unsqueeze(2), act_emb.unsqueeze(2)], dim=2)
        else:  # same tiling as VWorldModel.encode
            P = vis.shape[2]
            z = torch.cat([vis, prop.unsqueeze(2).expand(-1, -1, P, -1).repeat(1, 1, 1, wm.num_proprio_repeat),
                           act_emb.unsqueeze(2).expand(-1, -1, P, -1).repeat(1, 1, 1, wm.num_action_repeat)], dim=3)
        return rollout_tail(wm, z, act[:, n:])
    return rollout


def sdpa_forward(self, x):
    """Drop-in for models.vit.Attention.forward: same projections, scale (dim_head**-0.5) and frame-causal mask."""
    B, T, _ = x.size()
    q, k, v = (t.view(B, T, self.heads, -1).transpose(1, 2) for t in self.to_qkv(self.norm(x)).chunk(3, dim=-1))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=self.bias[:, :, :T, :T] != 0,
                                         dropout_p=self.dropout.p if self.training else 0.0)
    return self.to_out(out.transpose(1, 2).reshape(B, T, -1))


class Variant:
    """Context manager that applies a variant to the world model for the duration of a plan."""
    def __init__(self, wm, name):
        self.wm, (self.cache, self.sdpa, self.amp) = wm, VARIANTS[name]
        self.z_cache = {}

    def new_plan(self):
        self.z_cache.clear()

    def __enter__(self):
        import models.vit as vit
        self._attn_fwd = vit.Attention.forward
        if self.sdpa:
            vit.Attention.forward = sdpa_forward
        inner = make_cached_rollout(self.wm, self.z_cache) if self.cache else type(self.wm).rollout.__get__(self.wm)
        if self.amp is not None:
            def rollout(obs_0, act, _inner=inner):
                with torch.autocast("cuda", dtype=self.amp):
                    z_obses, z = _inner(obs_0, act)
                return {k: v.float() for k, v in z_obses.items()}, z.float()
        else:
            rollout = inner
        self.wm.rollout = rollout  # instance attribute shadows the class method
        return self

    def __exit__(self, *exc):
        import models.vit as vit
        vit.Attention.forward = self._attn_fwd
        del self.wm.rollout


# ---------------------------------------------------------------- setup (official code paths)
def load(args, device):
    sys.path.insert(0, str(DINO_WM))
    sys.path.insert(0, str(Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"))  # pickled encoder classes
    # env/__init__.py eagerly imports the PointMaze env (mujoco-py, d4rl); PushT does not need it.
    stub = types.ModuleType("env.pointmaze"); stub.U_MAZE = None; sys.modules["env.pointmaze"] = stub
    import gym, hydra
    from omegaconf import OmegaConf
    import env  # noqa: F401  registers "pusht"
    from env.venv import SubprocVectorEnv
    from plan import load_model
    from preprocessor import Preprocessor
    from planning.evaluator import PlanEvaluator
    from planning.objectives import create_objective_fn
    from planning.cem import CEMPlanner

    model_path = W / "ckpt" / "outputs" / args.model_name
    cfg = OmegaConf.load(model_path / "hydra.yaml")
    cfg.env.dataset.data_path = str(W / "data" / "pusht_noise")
    _, dset = hydra.utils.call(cfg.env.dataset, num_hist=cfg.num_hist, num_pred=cfg.num_pred, frameskip=cfg.frameskip)
    dset = dset["valid"]
    n, fs, H = args.n_tasks, cfg.frameskip, args.goal_H
    vec_env = SubprocVectorEnv([lambda: gym.make(cfg.env.name, *cfg.env.args, **cfg.env.kwargs) for _ in range(n)])
    wm = load_model(model_path / "checkpoints" / f"model_{args.model_epoch}.pth", cfg, cfg.num_action_repeat, device)
    flags = {k: getattr(wm, k).training for k in ("encoder", "predictor", "action_encoder", "proprio_encoder")}
    if args.eval_mode:
        wm.eval()

    pre = Preprocessor(action_mean=dset.action_mean, action_std=dset.action_std, state_mean=dset.state_mean,
                       state_std=dset.state_std, proprio_mean=dset.proprio_mean, proprio_std=dset.proprio_std,
                       transform=dset.transform)
    # Tasks: as PlanWorkspace.prepare_targets with goal_source='dset' (a validation segment's start state, and the
    # state its recorded actions reach after goal_H * frameskip env steps), read from the state/action tensors so no
    # video has to be decoded. The random draws are ours, so the tasks are not the official run's exact 50.
    rng, traj_len, states, actions, info = random.Random(args.seed), fs * H + 1, [], [], []
    while len(states) < n:
        i = rng.randrange(len(dset)); T = dset.get_seq_length(i)
        if T < traj_len:
            continue
        off = rng.randint(0, T - traj_len)
        states.append(dset.states[i, off:off + traj_len].numpy()); actions.append(dset.actions[i, off:off + fs * H])
        info.append({"shape": dset.shapes[i]})
    vec_env.update_env(info)
    eval_seed = [args.seed * k + 1 for k in range(n)]
    actions = torch.stack(actions)
    init_state = np.array([s[0] for s in states])
    obses, r_states = vec_env.rollout(eval_seed, init_state, pre.denormalize_actions(actions).numpy())
    obs_0 = {k: np.expand_dims(v[:, 0], 1) for k, v in obses.items()}
    obs_g = {k: np.expand_dims(v[:, -1], 1) for k, v in obses.items()}
    state_g = r_states[:, -1]
    from einops import rearrange
    gt_actions = rearrange(actions, "b (t f) d -> b t (f d)", f=fs)

    evaluator = PlanEvaluator(obs_0=obs_0, obs_g=obs_g, state_0=init_state, state_g=state_g, env=vec_env, wm=wm,
                              frameskip=fs, seed=eval_seed, preprocessor=pre, n_plot_samples=args.n_plot)
    rec = types.SimpleNamespace(rows=[], log=lambda d: rec.rows.append(d))
    planner = CEMPlanner(horizon=H, topk=args.topk, num_samples=args.num_samples, var_scale=1.0,
                         opt_steps=args.opt_steps, eval_every=1, wm=wm, action_dim=dset.action_dim * fs,
                         objective_fn=create_objective_fn(alpha=1, base=2, mode="last"), preprocessor=pre,
                         evaluator=None,  # no simulator calls or early stopping inside CEM
                         wandb_run=rec, log_filename=None)
    return dict(wm=wm, cfg=cfg, env=vec_env, evaluator=evaluator, planner=planner, rec=rec, obs_0=obs_0, obs_g=obs_g,
                state_g=state_g, gt_actions=gt_actions, train_flags=flags)


def meta(args, S, device):
    q = lambda *a: subprocess.run(["nvidia-smi", *a, "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    from omegaconf import OmegaConf
    sel = lambda c, k: OmegaConf.select(c, k, default="?")
    wm, cfg = S["wm"], S["cfg"]
    P = (cfg.img_size // 16) ** 2
    return {
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu (functional dry run)", "gpu_index_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "driver": q("--query-gpu=driver_version").splitlines()[0], "cpu": next(
            (l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?"),
        "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "python": sys.version.split()[0], "dtype": "float32 (fp16 autocast in *_fp16 variants)",
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark, "backend": "PyTorch eager, no torch.compile",
        "model": f"DINO-WM {args.model_name} (model_{args.model_epoch}.pth), encoder {sel(cfg, 'encoder.name')}, "
                 f"predictor ViT depth {sel(cfg, 'predictor.depth')} heads {sel(cfg, 'predictor.heads')}",
        "params_predictor": sum(p.numel() for p in wm.predictor.parameters()),
        "params_encoder": sum(p.numel() for p in wm.encoder.parameters()),
        "train_flags_as_loaded": S["train_flags"], "eval_mode_forced": args.eval_mode,
        "batch_candidates": args.num_samples, "cem_opt_steps": args.opt_steps, "cem_topk": args.topk,
        "horizon": args.goal_H, "frameskip": int(cfg.frameskip), "num_hist": int(cfg.num_hist),
        "tokens_per_frame": P, "context_tokens_max": P * int(cfg.num_hist), "img_size": int(cfg.img_size),
        "n_tasks": args.n_tasks, "seed": args.seed, "warmup_plans": args.warmup, "repetitions": "1 plan per task",
        "timer": "time.perf_counter around planner.plan with torch.cuda.synchronize before and after",
        "dino_wm_commit": subprocess.run(["git", "-C", str(DINO_WM), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "argv": sys.argv[1:],
    }


def gpu_state():
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip().splitlines()
    a = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip().splitlines()
    return {"gpus": r, "compute_apps": a, "loadavg": os.getloadavg()}


# ---------------------------------------------------------------- measurement
def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def plan_task(S, var, i, seed, device):
    o0 = {k: v[i:i + 1] for k, v in S["obs_0"].items()}
    og = {k: v[i:i + 1] for k, v in S["obs_g"].items()}
    var.new_plan(); S["rec"].rows.clear()
    torch.manual_seed(seed * 100003 + i)  # same candidate noise for every variant
    sync(device); t0 = time.perf_counter()
    mu, _ = S["planner"].plan(obs_0=o0, obs_g=og)
    sync(device)
    return time.perf_counter() - t0, mu.detach(), [r[next(k for k in r if k.endswith("/loss"))] for r in S["rec"].rows]


def timed(fn, acc, key, device):
    def f(*a, **k):
        sync(device); t = time.perf_counter()
        out = fn(*a, **k)
        sync(device); acc[key] = acc.get(key, 0.0) + time.perf_counter() - t
        return out
    return f


def run_variant(S, name, args, device, out):
    wm, n = S["wm"], args.n_tasks
    res = {"variant": name, "gpu_state_start": gpu_state()}
    with Variant(wm, name) as var:
        for _ in range(args.warmup):
            plan_task(S, var, 0, args.seed, device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        lat, mus, losses = [], [], []
        for i in range(n):
            dt, mu, ls = plan_task(S, var, i, args.seed, device)
            lat.append(dt); mus.append(mu); losses.append(ls)
            print(f"[{name}] task {i:3d} plan {dt:7.3f}s final_cem_loss {ls[-1]:.5f}", flush=True)
        cuda = device.type == "cuda"
        res["peak_mem_alloc_mb"] = torch.cuda.max_memory_allocated(device) / 2**20 if cuda else 0.0
        res["peak_mem_reserved_mb"] = torch.cuda.max_memory_reserved(device) / 2**20 if cuda else 0.0
        # phase breakdown: separate instrumented pass (synchronising timers perturb end-to-end latency)
        phases, planner = {}, S["planner"]
        orig = (wm.encode_obs, wm.predict, planner.objective_fn)
        wm.encode_obs, wm.predict = timed(orig[0], phases, "encode_obs", device), timed(orig[1], phases, "predict", device)
        planner.objective_fn = timed(orig[2], phases, "objective", device)
        tot = sum(plan_task(S, var, i, args.seed, device)[0] for i in range(min(args.phase_tasks, n)))
        del wm.encode_obs, wm.predict
        planner.objective_fn = orig[2]
        phases["other(sampling,topk,preprocess,act-encode)"] = tot - sum(phases.values())
        res["phase_seconds_per_plan"] = {k: v / min(args.phase_tasks, n) for k, v in phases.items()}
        res["phase_fraction"] = {k: v / tot for k, v in phases.items()}
    res["gpu_state_end"] = gpu_state()
    # task success: the planned open-loop actions executed in the simulator (official evaluator, official rollout)
    actions = torch.cat(mus, dim=0)
    os.makedirs(out / name, exist_ok=True); cwd = os.getcwd(); os.chdir(out / name)
    logs, succ, _, e_states = S["evaluator"].eval_actions(actions, filename="rollouts")
    os.chdir(cwd)
    lat = np.array(lat)
    res.update({
        "latency_s": lat.tolist(), "latency_median_s": float(np.median(lat)), "latency_mean_s": float(lat.mean()),
        "latency_std_s": float(lat.std(ddof=1)) if n > 1 else 0.0, "latency_p95_s": float(np.percentile(lat, 95)),
        "latency_max_s": float(lat.max()), "per_cem_iter_median_ms": float(np.median(lat)) / args.opt_steps * 1e3,
        "candidates_per_s": args.num_samples * args.opt_steps / float(np.median(lat)),
        "success": [bool(s) for s in succ], "success_rate": float(np.mean(succ)),
        "state_dist": np.linalg.norm(S["state_g"] - e_states[:, -1], axis=1).tolist(),
        "eval_logs": {k: float(v) for k, v in logs.items()}, "cem_loss_trace": losses,
        "actions": actions.cpu().tolist(),
    })
    return res


def reference_rows(S, out):
    """Sanity rows that need no planning: the recorded dataset actions (should reach the goal) and zero actions."""
    rows = {}
    for name, a in (("dataset_actions", S["gt_actions"]), ("zero_actions", torch.zeros_like(S["gt_actions"]))):
        os.makedirs(out / name, exist_ok=True); cwd = os.getcwd(); os.chdir(out / name)
        logs, succ, _, _ = S["evaluator"].eval_actions(a.to(next(S["wm"].parameters()).device), filename="rollouts")
        os.chdir(cwd)
        rows[name] = {"success_rate": float(np.mean(succ)), "eval_logs": {k: float(v) for k, v in logs.items()}}
    return rows


def selfcheck(S, args, device):
    """Runnable check: cached and fused-attention rollouts match the official rollout on the same inputs (eval mode)."""
    wm = S["wm"]; was = {m: m.training for m in wm.modules()}; wm.eval()
    pre, pl = S["planner"].preprocessor, S["planner"]
    from utils import move_to_device
    o0 = move_to_device(pre.transform_obs({k: v[:1] for k, v in S["obs_0"].items()}), device)
    obs = {k: v.expand(args.num_samples, *v.shape[1:]).contiguous() for k, v in o0.items()}
    torch.manual_seed(0); act = torch.randn(args.num_samples, pl.horizon, pl.action_dim, device=device)
    out = {}
    with torch.no_grad():
        ref = wm.rollout(obs, act)[0]["visual"]
        for name in ("cache", "cache_sdpa", "cache_sdpa_fp16"):
            with Variant(wm, name):
                got = wm.rollout(obs, act)[0]["visual"]
            out[name] = {"max_abs_diff": (got - ref).abs().max().item(), "ref_abs_mean": ref.abs().mean().item()}
    for m, t in was.items():
        m.training = t
    print("selfcheck", json.dumps(out))
    assert all(out[k]["max_abs_diff"] < 1e-3 for k in ("cache", "cache_sdpa")), "fp32 variant diverges from official rollout"
    return out


def profile(S, name, args, device, out):
    from torch.profiler import profile as prof, ProfilerActivity
    with Variant(S["wm"], name) as var:
        plan_task(S, var, 0, args.seed, device)  # warm-up
        with prof(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            plan_task(S, var, 0, args.seed, device)
    ka = p.key_averages()
    dev = lambda e: getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0))
    total = sum(dev(e) for e in ka) or 1
    rows = sorted(({"op": e.key, "self_gpu_ms": dev(e) / 1e3, "self_gpu_frac": dev(e) / total, "calls": e.count,
                    "self_cpu_ms": e.self_cpu_time_total / 1e3} for e in ka), key=lambda r: -r["self_gpu_ms"])[:25]
    (out / f"profile_{name}.txt").write_text(ka.table(sort_by="self_cuda_time_total", row_limit=30, max_name_column_width=60))
    return rows


def scaling(S, args, device):
    """World-model cost of one CEM iteration against the number of candidates K: latency of one rollout call
    (start-frame latents already cached in the cache variants, as in every iteration of a plan after the first)."""
    from utils import move_to_device
    wm, pl = S["wm"], S["planner"]
    o0 = move_to_device(pl.preprocessor.transform_obs({k: v[:1] for k, v in S["obs_0"].items()}), device)
    rows = []
    for name in args.variants.split(","):
        with Variant(wm, name) as var:
            for K in [int(k) for k in args.scaling_k.split(",")]:
                obs = {k: v.expand(K, *v.shape[1:]).contiguous() for k, v in o0.items()}
                torch.manual_seed(K); act = torch.randn(K, pl.horizon, pl.action_dim, device=device)
                var.new_plan(); ts = []
                if device.type == "cuda":
                    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
                try:
                    for _ in range(args.scaling_warmup + args.scaling_reps):
                        sync(device); t = time.perf_counter()
                        with torch.no_grad():
                            wm.rollout(obs, act)
                        sync(device); ts.append(time.perf_counter() - t)
                except torch.OutOfMemoryError:
                    rows.append({"variant": name, "K": K, "oom": True}); continue
                ts = np.array(ts[args.scaling_warmup:]) * 1e3
                rows.append({"variant": name, "K": K, "median_ms": float(np.median(ts)), "min_ms": float(ts.min()),
                             "p95_ms": float(np.percentile(ts, 95)), "ms_per_candidate": float(np.median(ts)) / K,
                             "peak_mem_alloc_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0,
                             "warmup": args.scaling_warmup, "reps": args.scaling_reps})
                print("scaling", json.dumps(rows[-1]), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--model-name", default="pusht"); ap.add_argument("--model-epoch", default="latest")
    ap.add_argument("--n-tasks", type=int, default=50); ap.add_argument("--goal-H", type=int, default=5)
    ap.add_argument("--num-samples", type=int, default=300); ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--opt-steps", type=int, default=30); ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--warmup", type=int, default=1); ap.add_argument("--phase-tasks", type=int, default=5)
    ap.add_argument("--n-plot", type=int, default=10); ap.add_argument("--profile", action="store_true")
    ap.add_argument("--scaling", action="store_true", help="also time one rollout call against the candidate count K")
    ap.add_argument("--scaling-k", default="1,2,4,8,16,32,64,128,256,300"); ap.add_argument("--skip-plans", action="store_true")
    ap.add_argument("--scaling-warmup", type=int, default=3); ap.add_argument("--scaling-reps", type=int, default=20)
    ap.add_argument("--device", default="cuda:0", help="cpu = functional dry run only (fp16 variant needs cuda)")
    ap.add_argument("--eval-mode", action="store_true", help="call wm.eval() (the official plan.py does not)")
    args = ap.parse_args()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    S = load(args, device)
    M = meta(args, S, device); print(json.dumps(M, indent=1), flush=True)
    result = {"meta": M, "selfcheck": selfcheck(S, args, device), "reference": reference_rows(S, out), "variants": {}}
    if args.scaling:
        result["scaling"] = scaling(S, args, device)
        (out / "results.json").write_text(json.dumps(result))
    for name in [] if args.skip_plans else args.variants.split(","):
        if args.profile:
            result.setdefault("profile_top_ops", {})[name] = profile(S, name, args, device, out)
        result["variants"][name] = run_variant(S, name, args, device, out)
        (out / "results.json").write_text(json.dumps(result))
        v = result["variants"][name]
        print(f"== {name}: median {v['latency_median_s']:.3f}s p95 {v['latency_p95_s']:.3f}s success "
              f"{v['success_rate']:.2f} peak {v['peak_mem_alloc_mb']:.0f}MB phases {v['phase_fraction']}", flush=True)
    S["env"].close()


if __name__ == "__main__":
    main()
