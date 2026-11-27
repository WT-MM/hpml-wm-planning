#!/usr/bin/env python
"""Collect compact fp16 on-policy CEM rollout dataset strictly from train_G episodes ([0..18184])
for training RefBasisG and equal-cost NoRefBasisG.

Saves one .pt shard per chunk of episodes (~3.6 GB total for 120 episodes x 6 CEM iters x 32 cands),
keeping disk footprint lean and allowing full in-RAM training.
"""
import argparse, json, random, sys, time, types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import specplan_math as sm


def load_train_wm(args, device):
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
    d_tr = dsets["train"]
    pre = Preprocessor(
        action_mean=d_tr.action_mean, action_std=d_tr.action_std, state_mean=d_tr.state_mean,
        state_std=d_tr.state_std, proprio_mean=d_tr.proprio_mean, proprio_std=d_tr.proprio_std,
        transform=d_tr.transform,
    )
    sim = gym.make(cfg.env.name, *cfg.env.args, **cfg.env.kwargs)
    return {
        "wm": wm, "cfg": cfg, "d_tr": d_tr, "pre": pre, "sim": sim,
        "obj_fn": create_objective_fn(alpha=1, base=2, mode="last"),
        "fs": int(cfg.frameskip), "action_dim": int(d_tr.action_dim * cfg.frameskip),
    }


def main():
    import torch
    from utils import move_to_device
    import planbench as pb

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-episodes", type=int, default=120)
    ap.add_argument("--heldout-n", type=int, default=sm.HELDOUT_TRAIN_N)
    ap.add_argument("--sample-iters", default="0,3,7,12,18,25")
    ap.add_argument("--k-save", type=int, default=32)
    ap.add_argument("--goal-H", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--num-samples", type=int, default=300)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--opt-steps", type=int, default=30)
    ap.add_argument("--variant", default="fp16_reuse")
    ap.add_argument("--shard-size", type=int, default=30, help="episodes per shard file")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model-name", default="pusht")
    ap.add_argument("--model-epoch", default="latest")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eval-mode", action="store_true")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    S = load_train_wm(args, device)
    wm, d_tr, pre, sim, fs = S["wm"], S["d_tr"], S["pre"], S["sim"], S["fs"]
    H, A, N, topk, steps = args.horizon, S["action_dim"], args.num_samples, args.topk, args.opt_steps
    sample_iters = sorted({int(x) for x in args.sample_iters.split(",") if 0 <= int(x) < steps})
    variant = args.variant if device.type == "cuda" else "cache_sdpa_reuse"

    n_train_total = len(d_tr.seq_lengths)
    cut = n_train_total - args.heldout_n
    traj_len = fs * args.goal_H + 1
    eligible = [i for i in range(cut) if d_tr.seq_lengths[i] >= traj_len]
    rng = random.Random(args.seed)
    chosen_eps = [eligible[rng.randrange(len(eligible))] for _ in range(args.n_episodes)]
    sm.assert_train_g_only(chosen_eps, n_train_total, heldout_n=args.heldout_n)

    t_start = time.perf_counter()
    shard_records, shard_idx, manifest_shards = [], 0, []

    def flush_shard():
        nonlocal shard_records, shard_idx
        if not shard_records:
            return
        fname = f"shard_{shard_idx:03d}.pt"
        torch.save(shard_records, out / fname)
        manifest_shards.append({"file": fname, "n_groups": len(shard_records)})
        print(f"[collect] wrote {fname} ({len(shard_records)} groups, elapsed {time.perf_counter() - t_start:.1f}s)", flush=True)
        shard_records = []
        shard_idx += 1

    with pb.Variant(wm, variant) as var:
        for ei, ep in enumerate(chosen_eps):
            T = d_tr.seq_lengths[ep]
            off = rng.randint(0, T - traj_len)
            ev_seed = args.seed * 10007 + ei + 1
            sim.update_env({"shape": d_tr.shapes[ep]})
            init_state = d_tr.states[ep, off].numpy()
            act_raw = pre.denormalize_actions(d_tr.actions[ep, off:off + fs * args.goal_H]).numpy()
            obses, _ = sim.rollout(ev_seed, init_state, act_raw)
            o0_np = {k: np.expand_dims(v[0], (0, 1)) for k, v in obses.items()}
            og_np = {k: np.expand_dims(v[-1], (0, 1)) for k, v in obses.items()}
            o0 = move_to_device(pre.transform_obs(o0_np), device)
            og = move_to_device(pre.transform_obs(og_np), device)

            var.new_plan()
            with torch.no_grad():
                z0 = wm.encode_obs(o0)
                zg = wm.encode_obs(og)
            z0_vis_cpu = z0["visual"][0, 0].detach().half().cpu()     # (196, 384)
            z0_prop_cpu = z0["proprio"][0, 0].detach().half().cpu()   # (10,)
            zg_vis_cpu = zg["visual"][0, 0].detach().half().cpu()
            zg_prop_cpu = zg["proprio"][0, 0].detach().half().cpu()

            torch.manual_seed(args.seed * 100003 + ei)
            mu = torch.zeros(H, A, device=device)
            sigma = torch.ones(H, A, device=device)
            obs_rep = {k: v.expand(N, *v.shape[1:]) for k, v in o0.items()}
            zg_rep = {k: v.expand(N, *v.shape[1:]) for k, v in zg.items()}

            for it in range(steps):
                action = torch.randn(N, H, A).to(device) * sigma + mu
                action[0] = mu
                with torch.no_grad():
                    z_obs, _ = wm.rollout(obs_rep, action)
                loss = S["obj_fn"](z_obs, zg_rep)
                top = torch.argsort(loss)[:topk]

                if it in sample_iters:
                    sel = sm.stratified_candidate_indices(loss.detach().cpu().numpy(), args.k_save, rng)
                    sel_t = torch.as_tensor(sel, device=device, dtype=torch.long)
                    shard_records.append({
                        "episode": int(ep),
                        "offset": int(off),
                        "cem_iter": int(it),
                        "sigma_mean": float(sigma.mean().item()),
                        "z0_vis": z0_vis_cpu,
                        "z0_prop": z0_prop_cpu,
                        "zg_vis": zg_vis_cpu,
                        "zg_prop": zg_prop_cpu,
                        "mu": mu.detach().reshape(-1).float().cpu(),
                        "zref_vis": z_obs["visual"][0, -1].detach().half().cpu(),
                        "zref_prop": z_obs["proprio"][0, -1].detach().half().cpu(),
                        "cand_act": action[sel_t].detach().reshape(len(sel), -1).float().cpu(),
                        "cand_vis": z_obs["visual"][sel_t, -1].detach().half().cpu(),
                        "cand_prop": z_obs["proprio"][sel_t, -1].detach().half().cpu(),
                    })
                mu = action[top].mean(dim=0)
                sigma = action[top].std(dim=0)

            if (ei + 1) % args.shard_size == 0 or ei == len(chosen_eps) - 1:
                flush_shard()

    manifest = {
        "args": vars(args),
        "n_train_total": n_train_total,
        "train_g_cut": cut,
        "sample_iters": sample_iters,
        "shards": manifest_shards,
        "elapsed_s": time.perf_counter() - t_start,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[collect] done: {sum(s['n_groups'] for s in manifest_shards)} groups in {manifest['elapsed_s']:.1f}s", flush=True)
    sim.close()


if __name__ == "__main__":
    main()
