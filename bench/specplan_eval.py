#!/usr/bin/env python
"""End-to-end speculative planning benchmark (open-loop and closed-loop MPC) comparing:
  - full            : full-model CEM (fp16_unmasked_reuse or fp16_reuse)
  - few<N>          : full model with N samples
  - online:m<M>     : zero-training GPU OnlineRidgeBuffer draft-and-verify (<0.5 ms draft!)
  - refg:std        : standalone trained RefBasisG (T_F(1) + T_G(K))
  - refg:m<M>       : draft-and-verify with trained RefBasisG
  - norefg:std      : standalone equal-cost NoRefBasisG (T_G(K), no reference rollout)
  - norefg:m<M>     : draft-and-verify with equal-cost NoRefBasisG
  - hybrid:m<M>     : trained RefBasisG + GPU online verified residual draft-and-verify
"""
import argparse, json, sys, time, types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import specplan_models as smod


def make_unmasked_reuse(pb, wm, compile_fn=False):
    class UnmaskedReusePredict(pb.ReusePredict):
        def _slide_frame_attn(self, attn, x, P):
            q, k, v = self._qkv(attn, x)
            n_frames = x.shape[1] // P
            dp = attn.dropout.p if attn.training else 0.0
            outs = [
                F.scaled_dot_product_attention(
                    q[:, :, f * P:(f + 1) * P], k[:, :, :(f + 1) * P], v[:, :, :(f + 1) * P],
                    attn_mask=None, dropout_p=dp,
                )
                for f in range(n_frames)
            ]
            o = torch.cat(outs, dim=2)
            return attn.to_out(o.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)) + x

        def _slide(self, x, mask, P):
            for attn, ff in self.layers[:-1]:
                x = self._slide_frame_attn(attn, x, P)
                x = ff(x) + x
            attn, ff = self.layers[-1]
            q, k, v = self._qkv(attn, x)
            return self.pred.transformer.norm(self._layer(attn, ff, x[:, -P:], q[:, :, -P:], k, v, None))

    return UnmaskedReusePredict(wm, compile=compile_fn)


def load_ckpt(path, device):
    if not path:
        return None
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    if ck["kind"] in ("refg", "refg_lin"):
        m = smod.RefBasisG(**cfg).to(device)
    else:
        cfg_nr = {k: v for k, v in cfg.items() if k != "linear_only"}
        m = smod.NoRefBasisG(**cfg_nr).to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


def load_wm_and_tasks(args, device):
    import planbench as pb
    import mpc_utils as mu_util

    sys.path.insert(0, str(pb.DINO_WM))
    sys.path.insert(0, str(Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"))
    stub = types.ModuleType("env.pointmaze"); stub.U_MAZE = None; sys.modules["env.pointmaze"] = stub
    import gym, hydra
    from omegaconf import OmegaConf
    from einops import rearrange
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
    fs = int(cfg.frameskip)
    seq_lens = {k: list(dsets[k].seq_lengths) for k in ("train", "valid")}

    def build_tasks(seed):
        tasks = []
        if args.task_source == "random_state":
            for i in range(args.n_tasks):
                ev_seed = seed * 1009 + i + 1
                sim.update_env({"shape": "T"})
                s0, sg = sim.sample_random_init_goal_states(ev_seed)
                o0, s0 = sim.prepare(ev_seed, s0)
                og, sg = sim.prepare(ev_seed, sg)
                tasks.append({
                    "eval_seed": ev_seed, "env_info": {"shape": "T"}, "state_0": s0, "state_g": sg,
                    "obs_0": {k: np.expand_dims(v, (0, 1)) for k, v in o0.items()},
                    "obs_g": {k: np.expand_dims(v, (0, 1)) for k, v in og.items()},
                })
            return tasks
        specs = mu_util.sample_dset_tasks(seq_lens, args.task_source, args.n_tasks, fs * args.goal_H + 1, seed, heldout_n=args.heldout_n)
        for i, sp in enumerate(specs):
            ds = dsets[sp["split"]]
            ep, off = sp["episode"], sp["offset"]
            ev_seed = seed * 1009 + i + 1
            env_info = {"shape": ds.shapes[ep]}
            sim.update_env(env_info)
            s0 = ds.states[ep, off].numpy()
            raw = pre.denormalize_actions(ds.actions[ep, off:off + fs * args.goal_H]).numpy()
            obses, states = sim.rollout(ev_seed, s0, raw)
            tasks.append({
                "eval_seed": ev_seed, "env_info": env_info, "state_0": s0, "state_g": states[-1],
                "obs_0": {k: np.expand_dims(v[0], (0, 1)) for k, v in obses.items()},
                "obs_g": {k: np.expand_dims(v[-1], (0, 1)) for k, v in obses.items()},
            })
        return tasks

    return {
        "wm": wm, "pre": pre, "sim": sim, "fs": fs, "action_dim": int(d0.action_dim * fs),
        "obj_fn": create_objective_fn(alpha=1, base=2, mode="last"), "build_tasks": build_tasks,
    }


def parse_method(spec):
    """Parse method spec:
      full, few<N>,
      <kind>:std, <kind>:m<M>, <kind>:m<M0>-<M1>, <kind>:k<K>:m<M>, <kind>:k<K>:m<M0>-<M1>
    where kind in {'online', 'refg', 'norefg', 'hybrid'}."""
    if spec == "full":
        return {"name": spec, "kind": "full", "N": None, "m": None, "m_sched": None}
    if spec.startswith("few"):
        return {"name": spec, "kind": "few", "N": int(spec[3:]), "m": None, "m_sched": None}
    parts = spec.split(":")
    if len(parts) < 2 or parts[0] not in ("online", "refg", "norefg", "hybrid"):
        raise ValueError(f"unknown method spec: {spec}")
    out = {"name": spec, "kind": parts[0], "N": None, "m": None, "m_sched": None}
    for p in parts[1:]:
        if p == "std":
            out["m"] = None
        elif p.startswith("k"):
            out["N"] = int(p[1:])
        elif p.startswith("m") and "-" in p:
            a, b = p[1:].split("-", 1)
            out["m"] = int(a)
            out["m_sched"] = (int(a), int(b))
        elif p.startswith("m"):
            out["m"] = int(p[1:])
        else:
            raise ValueError(f"unknown method spec token {p!r} in {spec}")
    return out


def plan_one_step(S, var, obs_0_t, zg, ug, parsed, models, args, device, noise_seed, init_mu=None):
    wm = S["wm"]
    H, A = args.horizon, S["action_dim"]
    N = parsed["N"] if parsed["N"] is not None else args.num_samples
    topk = max(2, min(N - 1, round(args.topk * N / args.num_samples))) if parsed["kind"] == "few" else args.topk

    var.new_plan()
    with torch.no_grad():
        z0_dict = wm.encode_obs(obs_0_t)
        if var.cache:
            var.z_cache["z"] = z0_dict
        z0_vis = z0_dict["visual"][:, 0]    # (1, 196, 384)
        z0_prop = z0_dict["proprio"][:, 0]  # (1, 10)

    def rollout_packed(act_b):
        B = act_b.shape[0]
        obs_rep = {k: v.expand(B, *v.shape[1:]) for k, v in obs_0_t.items()}
        with torch.no_grad():
            zo, _ = wm.rollout(obs_rep, act_b)
        U = smod.pack_latent(zo["visual"][:, -1], zo["proprio"][:, -1])
        return U, zo["visual"][:, -1], zo["proprio"][:, -1]

    def verify_excluding_ref(action, draft_loss, m_ver, loss_ref):
        """Always preserve exact mu (index 0) for free and verify top m_ver non-reference candidates."""
        non_ref = draft_loss[1:]
        cand = torch.argsort(non_ref)[:min(m_ver, len(non_ref))] + 1
        U_ver, _, _ = rollout_packed(action[cand])
        loss_ver = ((U_ver - ug[None]) ** 2).sum(dim=-1)
        loss = torch.full((len(action),), float("inf"), device=device)
        loss[0] = loss_ref
        loss[cand] = loss_ver
        return loss, cand, U_ver

    torch.manual_seed(noise_seed)
    mu = torch.zeros(H, A, device=device) if init_mu is None else init_mu.clone()
    sigma = torch.ones(H, A, device=device)
    ridge = smod.OnlineRidgeBuffer(max_iters=args.ridge_hist, lam=args.ridge_lam)
    full_evals = 0

    for it in range(args.opt_steps):
        action = torch.randn(N, H, A).to(device) * sigma + mu
        action[0] = mu
        act_flat = action.reshape(N, H * A)
        kind = parsed["kind"]
        if parsed["m_sched"] is not None:
            m0, m1 = parsed["m_sched"]
            m_ver = m0 if it < args.opt_steps // 3 else m1
        else:
            m_ver = parsed["m"]

        if kind in ("full", "few"):
            U, _, _ = rollout_packed(action)
            full_evals += N
            loss = ((U - ug[None]) ** 2).sum(dim=-1)
        elif kind == "online":
            if len(ridge) == 0:
                # Cold start: iter 0 runs full model on base num_samples (or N if smaller) to seed verified buffer
                n_seed = min(N, args.num_samples)
                U, _, _ = rollout_packed(action[:n_seed])
                full_evals += n_seed
                l_seed = ((U - ug[None]) ** 2).sum(dim=-1)
                loss = torch.full((N,), float("inf"), device=device)
                loss[:n_seed] = l_seed
                top_seed = torch.argsort(l_seed)[:min(max(m_ver, 60), n_seed)]
                ridge.update(act_flat[top_seed], U[top_seed])
            else:
                U_ref, _, _ = rollout_packed(mu[None])
                full_evals += 1
                loss_ref = ((U_ref[0] - ug) ** 2).sum()
                draft_loss = ridge.score_online(mu.reshape(-1), U_ref[0], ug, act_flat)
                loss, cand, U_ver = verify_excluding_ref(action, draft_loss, m_ver, loss_ref)
                full_evals += len(cand)
                ridge.update(act_flat[cand], U_ver)
        elif kind in ("refg", "hybrid"):
            U_ref, zref_v, zref_p = rollout_packed(mu[None])
            full_evals += 1
            loss_ref = ((U_ref[0] - ug) ** 2).sum()
            with torch.no_grad():
                if kind == "refg":
                    draft_loss, _, _ = models["refg"].score_candidates(z0_vis, z0_prop, zref_v, zref_p, mu.reshape(1, -1), ug, act_flat)
                else:
                    draft_loss = ridge.score_hybrid(models["refg"], z0_vis, z0_prop, zref_v, zref_p, mu.reshape(-1), ug, act_flat)
            if m_ver is None:
                loss = draft_loss
                loss[0] = loss_ref
            else:
                loss, cand, U_ver = verify_excluding_ref(action, draft_loss, m_ver, loss_ref)
                full_evals += len(cand)
                if kind == "hybrid":
                    ridge.update(act_flat[cand], U_ver)
        elif kind == "norefg":
            with torch.no_grad():
                draft_loss, _, _ = models["norefg"].score_candidates(z0_vis, z0_prop, ug, act_flat)
            if m_ver is None:
                loss = draft_loss
            else:
                cand = torch.argsort(draft_loss)[:m_ver]
                U_ver, _, _ = rollout_packed(action[cand])
                full_evals += len(cand)
                loss = torch.full((N,), float("inf"), device=device)
                loss[cand] = ((U_ver - ug[None]) ** 2).sum(dim=-1)

        top = torch.argsort(loss)[:topk]
        mu = action[top].mean(dim=0)
        sigma = action[top].std(dim=0)

    # Evaluate final planned mu with exact full model for comparable final objective
    U_fin, _, _ = rollout_packed(mu[None])
    fin_obj = float(((U_fin[0] - ug) ** 2).sum().item())
    return mu, fin_obj, full_evals


def main():
    import planbench as pb
    import mpc_utils as mu_util
    from einops import rearrange

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", default="full,few150,online:m45,refg:m45,norefg:m45,hybrid:m45")
    ap.add_argument("--refg-ckpt", default=None)
    ap.add_argument("--norefg-ckpt", default=None)
    ap.add_argument("--base-variant", default="fp16_reuse", help="fp16_reuse or fp16_reuse_compile")
    ap.add_argument("--unmasked", action=argparse.BooleanOptionalAction, default=True, help="use Flash UnmaskedReusePredict")
    ap.add_argument("--task-source", default="val+heldout", choices=["val", "heldout", "val+heldout", "random_state"])
    ap.add_argument("--heldout-n", type=int, default=mu_util.HELDOUT_TRAIN_N)
    ap.add_argument("--seeds", default="99")
    ap.add_argument("--n-tasks", type=int, default=16)
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
    ap.add_argument("--ridge-hist", type=int, default=2)
    ap.add_argument("--ridge-lam", type=float, default=1e-2)
    ap.add_argument("--model-name", default="pusht")
    ap.add_argument("--model-epoch", default="latest")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eval-mode", action="store_true")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    S = load_wm_and_tasks(args, device)
    from utils import move_to_device
    models = {
        "refg": load_ckpt(args.refg_ckpt, device),
        "norefg": load_ckpt(args.norefg_ckpt, device),
    }
    methods = [parse_method(m.strip()) for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m["kind"] in ("refg", "hybrid") and models["refg"] is None:
            raise ValueError(f"{m['name']} requires --refg-ckpt")
        if m["kind"] == "norefg" and models["norefg"] is None:
            raise ValueError(f"{m['name']} requires --norefg-ckpt")

    variant = args.base_variant if device.type == "cuda" else "cache_sdpa_reuse"
    seeds = [int(s) for s in args.seeds.split(",")]
    R = {"args": vars(args), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
         "runs": {m["name"]: [] for m in methods}}

    wm, sim, pre, fs = S["wm"], S["sim"], S["pre"], S["fs"]
    tr = pre.transform_obs
    H, A = args.horizon, S["action_dim"]

    with pb.Variant(wm, variant) as var:
        if args.unmasked:
            unmasked = make_unmasked_reuse(pb, wm, compile_fn=("compile" in variant))
            wm.predict = unmasked
            inner = pb.make_cached_rollout(wm, var.z_cache, unmasked.reset)
            if var.amp is not None:
                def _ro(o, a, _in=inner, _amp=var.amp):
                    with torch.autocast("cuda", dtype=_amp):
                        zo, z = _in(o, a)
                    return {k: v.float() for k, v in zo.items()}, z.float()
                wm.rollout = _ro
            else:
                wm.rollout = inner

        for seed in seeds:
            tasks = S["build_tasks"](seed)
            for parsed in methods:
                name = parsed["name"]

                def run_ep(t, tid):
                    sim.update_env(t["env_info"])
                    o0_raw, cur_state = sim.prepare(t["eval_seed"], t["state_0"])
                    cur_obs = {k: np.expand_dims(v, (0, 1)) for k, v in o0_raw.items()}
                    with torch.no_grad():
                        zg = wm.encode_obs(move_to_device(tr(t["obs_g"]), device))
                        ug = smod.pack_latent(zg["visual"][0, 0], zg["proprio"][0, 0])
                    memo, step_lats, objs, tot_fe = None, [], [], 0
                    for mpc_it in range(args.mpc_steps):
                        obs_0_t = move_to_device(tr(cur_obs), device)
                        init_mu = None
                        if args.warm_start and memo is not None and memo.shape[0] > 0:
                            rem = H - memo.shape[0]
                            init_mu = torch.cat([memo, torch.zeros(rem, A, device=device)], dim=0) if rem > 0 else memo[:H]
                        pb.sync(device); t0 = time.perf_counter()
                        mu, fin_obj, fe = plan_one_step(
                            S, var, obs_0_t, zg, ug, parsed, models, args, device,
                            seed * 100003 + tid * 997 + mpc_it, init_mu=init_mu,
                        )
                        pb.sync(device); step_lats.append(time.perf_counter() - t0)
                        objs.append(fin_obj); tot_fe += fe
                        taken = mu[:args.n_taken].detach()
                        memo = mu[args.n_taken:].detach()
                        raw = pre.denormalize_actions(rearrange(taken.cpu(), "h (f d) -> (h f) d", f=fs)).numpy()
                        obses, _, _, infos = sim.step_multiple(raw)
                        cur_state = infos["state"][-1]
                        cur_obs = {k: np.expand_dims(v[-1], (0, 1)) for k, v in obses.items()}
                        if args.early_stop and sim.eval_state(t["state_g"], cur_state)["success"]:
                            break
                    ev = sim.eval_state(t["state_g"], cur_state)
                    return {
                        "episode_plan_s": float(sum(step_lats)), "step_plan_s": step_lats,
                        "success": bool(ev["success"]), "final_state_dist": float(ev["state_dist"]),
                        "final_obj": objs[-1], "full_evals": tot_fe,
                    }

                for _ in range(args.warmup):
                    run_ep(tasks[0], 0)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                ep_s, st_s, succ, dist, objs, tot_fe = [], [], [], [], [], 0
                for i, t in enumerate(tasks):
                    r = run_ep(t, i)
                    ep_s.append(r["episode_plan_s"]); st_s.extend(r["step_plan_s"])
                    succ.append(r["success"]); dist.append(r["final_state_dist"])
                    objs.append(r["final_obj"]); tot_fe += r["full_evals"]
                cuda = device.type == "cuda"
                R["runs"][name].append({
                    "seed": seed, "episode_plan_s": ep_s, "step_plan_s": st_s,
                    "success": succ, "final_state_dist": dist, "final_obj": objs,
                    "total_candidates": tot_fe,
                    "peak_mem_alloc_mb": torch.cuda.max_memory_allocated(device) / 2**20 if cuda else 0.0,
                })
                print(f"[{name} seed={seed}] ep_med={np.median(ep_s):.3f}s succ={np.mean(succ):.3f} "
                      f"dist={np.mean(dist):.1f} obj={np.mean(objs):.4f}", flush=True)
                (out / "specplan_eval.json").write_text(json.dumps(R))

    R["summary"] = mu_util.aggregate_mpc_results(R["runs"], base_name=methods[0]["name"])
    (out / "specplan_eval.json").write_text(json.dumps(R))
    print("\n| method | ep median (s) | step median / p95 (s) | speedup [95% CI] | peak MiB | success | state dist |")
    print("|---|---|---|---|---|---|---|")
    for name, d in R["summary"].items():
        lo, hi = d["speedup_ci95"]
        print(f"| {name} | {d['episode_median_s']:.3f} | {d['step_median_s']:.3f} / {d['step_p95_s']:.3f} | "
              f"{d['speedup_vs_base']:.2f}x [{lo:.2f}, {hi:.2f}] | {d['peak_mem_alloc_mb']:.0f} | "
              f"{d['success_rate']:.3f} | {d['mean_final_state_dist']:.1f} |")
    sim.close()


if __name__ == "__main__":
    main()
