#!/usr/bin/env python
"""Closed-loop receding-horizon MPC benchmark harness for DINO-WM on PushT.

Supports leak-free episode splits (val, heldout train episodes never seen by G, random_state goals),
multiple seeds (e.g. 3 seeds x 200 paired tasks), warm-started receding-horizon replanning, and
paired comparison of CEM variants, fewer-sample baselines, shrinking-budget CEM, and GD planning.
"""
import argparse, json, os, sys, time, types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import mpc_utils as mu_util


def load_env_and_wm(args, device):
    """Load DINO-WM model, PushT train+val datasets, Preprocessor, and single-env simulator."""
    import torch
    import planbench as pb

    sys.path.insert(0, str(pb.DINO_WM))
    sys.path.insert(0, str(Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"))
    stub = types.ModuleType("env.pointmaze"); stub.U_MAZE = None; sys.modules["env.pointmaze"] = stub
    import gym, hydra
    from omegaconf import OmegaConf
    import env  # noqa: F401
    from plan import load_model
    from preprocessor import Preprocessor
    from planning.objectives import create_objective_fn

    model_path = pb.W / "ckpt" / "outputs" / args.model_name
    cfg = OmegaConf.load(model_path / "hydra.yaml")
    cfg.env.dataset.data_path = str(pb.W / "data" / "pusht_noise")
    _, dsets = hydra.utils.call(cfg.env.dataset, num_hist=cfg.num_hist, num_pred=cfg.num_pred, frameskip=cfg.frameskip)
    wm = load_model(model_path / "checkpoints" / f"model_{args.model_epoch}.pth", cfg, cfg.num_action_repeat, device)
    if args.eval_mode:
        wm.eval()
    d0 = dsets["valid"]
    pre = Preprocessor(
        action_mean=d0.action_mean, action_std=d0.action_std, state_mean=d0.state_mean,
        state_std=d0.state_std, proprio_mean=d0.proprio_mean, proprio_std=d0.proprio_std,
        transform=d0.transform,
    )
    sim = gym.make(cfg.env.name, *cfg.env.args, **cfg.env.kwargs)
    obj_fn = create_objective_fn(alpha=1, base=2, mode="last")
    return {
        "wm": wm, "cfg": cfg, "dsets": dsets, "pre": pre, "sim": sim,
        "obj_fn": obj_fn, "fs": int(cfg.frameskip), "action_dim": int(d0.action_dim * cfg.frameskip),
    }


def prepare_seed_tasks(S, args, seed):
    """Build n_tasks paired (init_state, obs_0, state_g, obs_g, env_info, gt_actions) for one seed."""
    import torch
    from einops import rearrange

    sim, pre, fs, H_goal = S["sim"], S["pre"], S["fs"], args.goal_H
    traj_len = fs * H_goal + 1
    tasks = []
    if args.task_source == "random_state":
        for i in range(args.n_tasks):
            ev_seed = seed * 1009 + i + 1
            sim.update_env({"shape": "T"})
            s0, sg = sim.sample_random_init_goal_states(ev_seed)
            o0, s0 = sim.prepare(ev_seed, s0)
            og, sg = sim.prepare(ev_seed, sg)
            tasks.append({
                "meta": {"split": "random_state", "eval_seed": ev_seed},
                "eval_seed": ev_seed, "env_info": {"shape": "T"},
                "state_0": s0, "state_g": sg,
                "obs_0": {k: np.expand_dims(v, (0, 1)) for k, v in o0.items()},
                "obs_g": {k: np.expand_dims(v, (0, 1)) for k, v in og.items()},
                "gt_actions": None,
            })
        return tasks

    seq_lens = {k: list(S["dsets"][k].seq_lengths) for k in ("train", "valid")}
    specs = mu_util.sample_dset_tasks(seq_lens, args.task_source, args.n_tasks, traj_len, seed, heldout_n=args.heldout_n)
    for i, sp in enumerate(specs):
        ds = S["dsets"][sp["split"]]
        ep, off = sp["episode"], sp["offset"]
        ev_seed = seed * 1009 + i + 1
        env_info = {"shape": ds.shapes[ep]}
        sim.update_env(env_info)
        init_state = ds.states[ep, off].numpy()
        act_norm = ds.actions[ep, off:off + fs * H_goal]
        act_raw = pre.denormalize_actions(act_norm).numpy()
        obses, states = sim.rollout(ev_seed, init_state, act_raw)
        gt_macro = rearrange(act_norm, "(t f) d -> t (f d)", f=fs)
        tasks.append({
            "meta": {**sp, "eval_seed": ev_seed},
            "eval_seed": ev_seed, "env_info": env_info,
            "state_0": init_state, "state_g": states[-1],
            "obs_0": {k: np.expand_dims(v[0], (0, 1)) for k, v in obses.items()},
            "obs_g": {k: np.expand_dims(v[-1], (0, 1)) for k, v in obses.items()},
            "gt_actions": gt_macro,
        })
    return tasks


def eval_open_loop_ref(S, tasks, args, use_gt=True):
    """Sanity rows without planning: recorded dataset actions (if available) or zero actions."""
    from einops import rearrange
    sim, pre, fs = S["sim"], S["pre"], S["fs"]
    act_d = S["action_dim"] // fs
    succs, dists = [], []
    for t in tasks:
        if use_gt and t["gt_actions"] is None:
            return None
        sim.update_env(t["env_info"])
        if use_gt:
            act_raw = pre.denormalize_actions(rearrange(t["gt_actions"], "h (f d) -> (h f) d", f=fs)).numpy()
        else:
            n_macro = t["gt_actions"].shape[0] if t["gt_actions"] is not None else args.n_taken * args.mpc_steps
            act_raw = np.zeros((fs * n_macro, act_d), dtype=np.float32)
        _, states = sim.rollout(t["eval_seed"], t["state_0"], act_raw)
        ev = sim.eval_state(t["state_g"], states[-1])
        succs.append(bool(ev["success"]))
        dists.append(float(ev["state_dist"]))
    return {"success_rate": float(np.mean(succs)), "mean_state_dist": float(np.mean(dists))}


def plan_cem_step(S, var, obs_0_t, zg, parsed, args, device, noise_seed, init_mu=None):
    """One receding-horizon CEM plan from obs_0_t toward zg; returns (mu, final_loss, n_candidates_evaluated)."""
    import torch
    wm, obj_fn = S["wm"], S["obj_fn"]
    H, A = args.horizon, S["action_dim"]
    var.new_plan()
    torch.manual_seed(noise_seed)
    mu = torch.zeros(H, A, device=device) if init_mu is None else init_mu.clone()
    sigma = torch.ones(H, A, device=device)
    total_cands = 0
    final_loss = float("nan")

    for it in range(args.opt_steps):
        N_it, k_it = mu_util.iter_budget(parsed, it, args.opt_steps, args.num_samples, args.topk)
        action = torch.randn(N_it, H, A).to(device) * sigma + mu
        action[0] = mu
        obs_rep = {k: v.expand(N_it, *v.shape[1:]) for k, v in obs_0_t.items()}
        zg_rep = {k: v.expand(N_it, *v.shape[1:]) for k, v in zg.items()}
        with torch.no_grad():
            z_obs, _ = wm.rollout(obs_rep, action)
        loss = obj_fn(z_obs, zg_rep)
        top = torch.argsort(loss)[:k_it]
        final_loss = float(loss[top[0]].item())
        mu = action[top].mean(dim=0)
        sigma = action[top].std(dim=0)
        total_cands += N_it
    return mu, final_loss, total_cands


def plan_gd_step(S, var, obs_0_t, zg, args, device, noise_seed, init_mu=None):
    """One receding-horizon gradient-based plan (SGD/Adam + noise, matching GDPlanner)."""
    import torch
    wm, obj_fn = S["wm"], S["obj_fn"]
    H, A, B = args.horizon, S["action_dim"], args.gd_restarts
    var.new_plan()
    torch.manual_seed(noise_seed)
    if init_mu is None:
        act = torch.randn(B, H, A, device=device)
    else:
        act = init_mu[None].repeat(B, 1, 1)
        if B > 1:
            act[1:] = act[1:] + 0.5 * torch.randn(B - 1, H, A, device=device)
    act = act.detach().clone().requires_grad_(True)
    obs_rep = {k: v.expand(B, *v.shape[1:]) for k, v in obs_0_t.items()}
    zg_rep = {k: v.expand(B, *v.shape[1:]).detach() for k, v in zg.items()}
    if var.cache:  # populate start-frame cache detached so backward doesn't retain encoder graph across GD steps
        with torch.no_grad():
            var.z_cache["z"] = wm.encode_obs(obs_0_t)
    opt = torch.optim.Adam([act], lr=args.gd_lr) if args.gd_opt == "adam" else torch.optim.SGD([act], lr=args.gd_lr)

    best_loss = None
    for _ in range(args.gd_steps):
        opt.zero_grad()
        z_obs, _ = wm.rollout(obs_rep, act)
        loss = obj_fn(z_obs, zg_rep)
        loss.sum().backward()
        opt.step()
        if args.gd_noise > 0:
            with torch.no_grad():
                act.add_(torch.randn_like(act) * args.gd_noise)
        best_loss = loss.detach()
    best_i = int(best_loss.argmin().item())
    return act[best_i].detach(), float(best_loss[best_i].item()), B * args.gd_steps


def run_planner_on_seed(S, tasks, spec, args, device, seed):
    """Run one planner on all tasks for one seed in closed-loop receding-horizon MPC."""
    import torch
    from einops import rearrange
    from utils import move_to_device
    import planbench as pb

    parsed = mu_util.parse_planner_spec(spec)
    wm, sim, pre, fs = S["wm"], S["sim"], S["pre"], S["fs"]
    H, A = args.horizon, S["action_dim"]
    tr = pre.transform_obs

    def run_episode(t, task_idx):
        sim.update_env(t["env_info"])
        o0_raw, cur_state = sim.prepare(t["eval_seed"], t["state_0"])
        cur_obs = {k: np.expand_dims(v, (0, 1)) for k, v in o0_raw.items()}
        with torch.no_grad():
            zg = wm.encode_obs(move_to_device(tr(t["obs_g"]), device))
        memo = None
        step_lats, step_objs, total_cands = [], [], 0
        solved_step = None
        ev = sim.eval_state(t["state_g"], cur_state)

        for mpc_it in range(args.mpc_steps):
            obs_0_t = move_to_device(tr(cur_obs), device)
            init_mu = None
            if args.warm_start and memo is not None and memo.shape[0] > 0:
                rem = H - memo.shape[0]
                init_mu = torch.cat([memo, torch.zeros(rem, A, device=device)], dim=0) if rem > 0 else memo[:H]
            noise_seed = seed * 100003 + task_idx * 997 + mpc_it

            pb.sync(device)
            t0 = time.perf_counter()
            if parsed["kind"] == "gd":
                mu, obj_val, n_c = plan_gd_step(S, var, obs_0_t, zg, args, device, noise_seed, init_mu=init_mu)
            else:
                mu, obj_val, n_c = plan_cem_step(S, var, obs_0_t, zg, parsed, args, device, noise_seed, init_mu=init_mu)
            pb.sync(device)
            step_lats.append(time.perf_counter() - t0)
            step_objs.append(obj_val)
            total_cands += n_c

            taken = mu[:args.n_taken].detach()
            memo = mu[args.n_taken:].detach()
            exec_raw = pre.denormalize_actions(rearrange(taken.cpu(), "h (f d) -> (h f) d", f=fs)).numpy()
            obses, _, _, infos = sim.step_multiple(exec_raw)
            cur_state = infos["state"][-1]
            cur_obs = {k: np.expand_dims(v[-1], (0, 1)) for k, v in obses.items()}
            ev = sim.eval_state(t["state_g"], cur_state)
            if ev["success"] and solved_step is None:
                solved_step = mpc_it + 1
                if args.early_stop:
                    break

        return {
            "episode_plan_s": float(sum(step_lats)),
            "step_plan_s": step_lats,
            "final_cem_obj": step_objs[-1],
            "total_candidates": total_cands,
            "success": bool(ev["success"]),
            "any_step_success": solved_step is not None,
            "steps_to_success": solved_step,
            "final_state_dist": float(ev["state_dist"]),
        }

    with pb.Variant(wm, parsed["variant"]) as var:
        for _ in range(args.warmup):
            run_episode(tasks[0], 0)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        ep_plan_s, step_plan_s, succ, any_succ, dist, objs, steps_to = [], [], [], [], [], [], []
        tot_cands = 0
        for i, t in enumerate(tasks):
            r = run_episode(t, i)
            ep_plan_s.append(r["episode_plan_s"])
            step_plan_s.extend(r["step_plan_s"])
            succ.append(r["success"])
            any_succ.append(r["any_step_success"])
            dist.append(r["final_state_dist"])
            objs.append(r["final_cem_obj"])
            steps_to.append(r["steps_to_success"])
            tot_cands += r["total_candidates"]
            if (i + 1) % max(1, len(tasks) // 4) == 0 or i == len(tasks) - 1:
                print(f"[{spec} seed={seed}] {i + 1:3d}/{len(tasks)} succ={np.mean(succ):.3f} "
                      f"ep_med={np.median(ep_plan_s):.3f}s dist={np.mean(dist):.1f}", flush=True)
        cuda = device.type == "cuda"
        return {
            "seed": seed,
            "planner": spec,
            "episode_plan_s": ep_plan_s,
            "step_plan_s": step_plan_s,
            "success": succ,
            "any_step_success": any_succ,
            "steps_to_success": steps_to,
            "final_state_dist": dist,
            "final_cem_obj": objs,
            "total_candidates": tot_cands,
            "peak_mem_alloc_mb": torch.cuda.max_memory_allocated(device) / 2**20 if cuda else 0.0,
            "peak_mem_reserved_mb": torch.cuda.max_memory_reserved(device) / 2**20 if cuda else 0.0,
        }


def selfcheck_sim_incremental(S, task):
    """Runnable check: live sim.prepare + step_multiple chunks match full sim.rollout bit-for-bit
    (preserving hidden pymunk velocities across MPC steps)."""
    from einops import rearrange
    sim, pre, fs = S["sim"], S["pre"], S["fs"]
    sim.update_env(task["env_info"])
    if task["gt_actions"] is not None:
        raw = pre.denormalize_actions(rearrange(task["gt_actions"], "h (f d) -> (h f) d", f=fs)).numpy()
    else:
        rng = np.random.default_rng(0)
        raw = rng.normal(0.0, 0.2, size=(fs * 2, 2)).astype(np.float32)
    _, full_states = sim.rollout(task["eval_seed"], task["state_0"], raw)
    mid = fs
    sim.prepare(task["eval_seed"], task["state_0"])
    _, _, _, info1 = sim.step_multiple(raw[:mid])
    _, _, _, info2 = sim.step_multiple(raw[mid:])
    diff = float(np.max(np.abs(full_states[-1] - info2["state"][-1])))
    assert diff < 1e-5, f"incremental simulator state mismatch: {diff}"
    return {"sim_incremental_max_state_diff": diff}


def print_summary(agg):
    print("\n| planner | tasks (seeds) | ep median (s) | step median / p95 (s) | speedup [95% CI] | cands/s | peak MiB | success (seed sd) | state dist |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, d in agg.items():
        lo, hi = d["speedup_ci95"]
        print(f"| {name} | {d['n_tasks_total']} ({d['n_seeds']}) | {d['episode_median_s']:.3f} | "
              f"{d['step_median_s']:.3f} / {d['step_p95_s']:.3f} | {d['speedup_vs_base']:.2f}x [{lo:.2f}, {hi:.2f}] | "
              f"{d['candidates_per_s']:.0f} | {d['peak_mem_alloc_mb']:.0f} | "
              f"{d['success_rate']:.3f} ({d['success_rate_seed_std']:.3f}) | {d['mean_final_state_dist']:.1f} |")


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--planners", default="cem:fp16_reuse,few150:fp16_reuse,shrink300_60:fp16_reuse")
    ap.add_argument("--task-source", default="val+heldout", choices=["val", "heldout", "val+heldout", "random_state"])
    ap.add_argument("--heldout-n", type=int, default=mu_util.HELDOUT_TRAIN_N)
    ap.add_argument("--seeds", default="99,100,101")
    ap.add_argument("--n-tasks", type=int, default=200)
    ap.add_argument("--goal-H", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--n-taken", type=int, default=2)
    ap.add_argument("--mpc-steps", type=int, default=5)
    ap.add_argument("--warm-start", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--early-stop", action="store_true")
    ap.add_argument("--num-samples", type=int, default=300)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--opt-steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--gd-steps", type=int, default=60)
    ap.add_argument("--gd-lr", type=float, default=0.1)
    ap.add_argument("--gd-noise", type=float, default=0.003)
    ap.add_argument("--gd-restarts", type=int, default=8)
    ap.add_argument("--gd-opt", default="adam", choices=["adam", "sgd"])
    ap.add_argument("--model-name", default="pusht")
    ap.add_argument("--model-epoch", default="latest")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eval-mode", action="store_true")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    S = load_env_and_wm(args, device)
    seeds = [int(s) for s in args.seeds.split(",")]
    planners = [p.strip() for p in args.planners.split(",") if p.strip()]
    for p in planners:
        mu_util.parse_planner_spec(p)  # validate upfront

    R = {
        "args": vars(args),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu (functional dry run)",
        "runs": {p: [] for p in planners},
        "reference": {},
        "task_metas": {},
    }
    for si, seed in enumerate(seeds):
        tasks = prepare_seed_tasks(S, args, seed)
        if si == 0:
            R["selfcheck"] = selfcheck_sim_incremental(S, tasks[0])
            print("selfcheck", json.dumps(R["selfcheck"]), flush=True)
        R["task_metas"][str(seed)] = [t["meta"] for t in tasks]
        R["reference"][str(seed)] = {
            "dataset_actions": eval_open_loop_ref(S, tasks, args, use_gt=True),
            "zero_actions": eval_open_loop_ref(S, tasks, args, use_gt=False),
        }
        for spec in planners:
            R["runs"][spec].append(run_planner_on_seed(S, tasks, spec, args, device, seed))
            (out / "mpceval.json").write_text(json.dumps(R))

    R["summary"] = mu_util.aggregate_mpc_results(R["runs"], base_name=planners[0])
    (out / "mpceval.json").write_text(json.dumps(R))
    print_summary(R["summary"])
    S["sim"].close()


if __name__ == "__main__":
    main()
