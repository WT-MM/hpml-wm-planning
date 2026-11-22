#!/usr/bin/env python
"""Draft-and-verify scan for CEM planning with DINO-WM (exploration; not part of the benchmark tables).

Idea under test: CEM only needs to know which candidates are the elites. A cheap "draft" rollout scores all N
candidates, the full model re-scores only the draft's top m, and the elites are the best k of those m by the full
model. Whenever the true elites are inside the draft's top m, CEM proceeds exactly as with the full model.

scan   plan with the full model exactly as CEMPlanner does; at every iteration also score the same candidates with
       each draft and record elite recall (share of the true top k inside the draft's top m) and the draft's cost.
plans  end-to-end plans on the same tasks with the same noise: draft-and-verify, and the baseline it has to beat,
       the full model with fewer samples. Quality = the full model's objective for the final plan, and the simulator.

Drafts need no training: the same predictor on a random subset of the patch tokens (tokR, as in Sparse Imagination,
arXiv 2506.01392) and/or with a shorter context window (ctxC). Spec examples: tok0.25, ctx1, tok0.25+ctx1.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import planbench as pb


class DraftPredict(pb.ReusePredict):
    """ReusePredict on a subset idx of the patch tokens (the same tokens in every frame). idx = all tokens gives
    ReusePredict itself."""
    def __init__(self, wm, idx, P_full):
        super().__init__(wm); self.idx, self.P_full = idx, P_full

    def _pos(self, a, b):  # position embeddings of frames [a, b) at the kept tokens
        f = torch.arange(a, b, device=self.idx.device)[:, None] * self.P_full + self.idx
        return self.pred.pos_embedding[:, f.flatten()]

    def _mask(self, a, T, P):  # frame-causal: queries in frames [a, T), keys in frames [0, T)
        f = torch.arange(T, device=self.idx.device).repeat_interleave(P)
        return f[a * P:, None] >= f[None, :]

    def __call__(self, zw):
        T, P = zw.shape[1], zw.shape[2]
        if T > self.done:
            x = self.pred.dropout(zw[:, self.done:].flatten(1, 2) + self._pos(self.done, T))
            out, self.kv = self.grow(x, self.kv, None if T - self.done == 1 else self._mask(self.done, T, P)); self.done = T
        else:
            self.kv = None
            out = self.slide(self.pred.dropout(zw.flatten(1, 2) + self._pos(0, T)), self._mask(0, T, P), P)
        return out[:, None, -P:]


def make_draft(wm, spec, P, device, seed):
    r, ctx = 1.0, wm.num_hist
    for part in spec.split("+"):
        if part.startswith("tok"):
            r = float(part[3:])
        elif part.startswith("ctx"):
            ctx = int(part[3:])
        elif part != "full":
            raise ValueError(spec)
    g = torch.Generator().manual_seed(seed)  # one fixed token subset per draft for the whole run
    idx = (torch.arange(P) if r >= 1 else torch.randperm(P, generator=g)[:max(1, round(r * P))].sort().values).to(device)
    return DraftPredict(wm, idx, P), idx, ctx


def final_latents(wm, pred, z, action, ctx):
    """The rollout loop of VWorldModel.rollout with a context window of ctx frames; returns the final frame only."""
    pred.reset()
    for t in range(action.shape[1]):
        z = torch.cat([z, wm.replace_actions_from_z(pred(z[:, -ctx:]), action[:, t:t + 1])], 1)
    return pred(z[:, -ctx:])


class Scorer:
    """Scores candidate action sequences for one task (its start observation and goal)."""
    def __init__(self, S, var, i, device, amp):
        from utils import move_to_device
        self.wm, self.pl, self.var, self.device = S["wm"], S["planner"], var, device
        self.amp = amp if device.type == "cuda" else None
        tr = self.pl.preprocessor.transform_obs
        self.o0 = move_to_device(tr({k: v[i:i + 1] for k, v in S["obs_0"].items()}), device)
        var.new_plan()
        with torch.no_grad():
            self.zg = self.wm.encode_obs(move_to_device(tr({k: v[i:i + 1] for k, v in S["obs_g"].items()}), device))

    @staticmethod
    def _rep(d, B):
        return {k: v.expand(B, *v.shape[1:]) for k, v in d.items()}

    def full(self, action):
        with torch.no_grad():
            z_obs, _ = self.wm.rollout(self._rep(self.o0, len(action)), action)
        return self.pl.objective_fn(z_obs, self._rep(self.zg, len(action)))

    def draft(self, action, d):
        pred, idx, ctx = d; B = len(action)
        with torch.no_grad(), torch.autocast("cuda", dtype=self.amp, enabled=self.amp is not None):
            z, n = pb.start_latents(self.wm, self.var.z_cache, self._rep(self.o0, B), action)
            zf = final_latents(self.wm, pred, z[:, :, idx], action[:, n:], ctx)
        zo = {k: v.float() for k, v in self.wm.separate_emb(zf)[0].items()}
        zg = self._rep(self.zg, B)
        return self.pl.objective_fn(zo, {"visual": zg["visual"][:, :, idx], "proprio": zg["proprio"]})


def cem(seed, N, topk, steps, H, A, device, select):
    """CEMPlanner.plan for one task (same sampling, noise and elite update), with a pluggable
    select(iteration, action) -> loss per candidate; inf marks a candidate that cannot be an elite."""
    torch.manual_seed(seed)
    mu, sigma, best = torch.zeros(H, A, device=device), torch.ones(H, A, device=device), []
    for it in range(steps):
        action = torch.randn(N, H, A).to(device) * sigma + mu
        action[0] = mu
        loss = select(it, action)
        top = torch.argsort(loss)[:topk]
        best.append(loss[top[0]].item()); mu, sigma = action[top].mean(0), action[top].std(0)
    return mu, best


def timed(fn, device):
    pb.sync(device); t = time.perf_counter(); out = fn(); pb.sync(device)
    return out, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--variant", default="fp16_reuse", help="the full model's implementation")
    ap.add_argument("--drafts", default="tok0.5,tok0.25,tok0.125,ctx1,tok0.25+ctx1")
    ap.add_argument("--plan-drafts", default="tok0.25,tok0.125,ctx1"); ap.add_argument("--verify-m", default="45,60,90")
    ap.add_argument("--few", default="150,100,60", help="fewer-samples baselines (full model), top k scaled with N")
    ap.add_argument("--recall-m", default="30,45,60,90,150")
    ap.add_argument("--model-name", default="pusht"); ap.add_argument("--model-epoch", default="latest")
    ap.add_argument("--n-tasks", type=int, default=16); ap.add_argument("--goal-H", type=int, default=5)
    ap.add_argument("--num-samples", type=int, default=300); ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--opt-steps", type=int, default=30); ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--n-plot", type=int, default=4); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eval-mode", action="store_true")
    args = ap.parse_args()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    S = pb.load(args, device); wm, pl = S["wm"], S["planner"]
    N, k, steps, H, A, n = args.num_samples, args.topk, args.opt_steps, pl.horizon, pl.action_dim, args.n_tasks
    amp = pb.VARIANTS[args.variant].get("amp"); P = (S["cfg"].img_size // 16) ** 2
    ms = [int(m) for m in args.recall_m.split(",") if int(m) <= N]
    seed_of = lambda i: args.seed * 100003 + i  # as planbench.plan_task: the same candidate noise
    R = {"args": vars(args), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu (functional dry run)",
         "tokens_per_frame": P, "num_hist": wm.num_hist}

    with pb.Variant(wm, args.variant) as var:
        drafts = {s: make_draft(wm, s, P, device, args.seed) for s in dict.fromkeys(["full"] + args.drafts.split(","))}
        # ---- the one runnable check: our CEM loop reproduces the official planner, and the all-token draft is the full model
        _, mu_ref, _ = pb.plan_task(S, var, 0, args.seed, device)
        sc = Scorer(S, var, 0, device, amp)
        mu, _ = cem(seed_of(0), N, k, steps, H, A, device, lambda it, a: sc.full(a))
        torch.manual_seed(0); a = torch.randn(N, H, A, device=device)
        lf, ld = sc.full(a), sc.draft(a, drafts["full"])
        R["check"] = {"cem_vs_official_max_abs_action_diff": (mu - mu_ref[0]).abs().max().item(),
                      "all_token_draft_vs_full_max_abs_loss_diff": (lf - ld).abs().max().item(), "loss_spread": (lf.max() - lf.min()).item()}
        print("check", json.dumps(R["check"]), flush=True)
        assert R["check"]["cem_vs_official_max_abs_action_diff"] < 1e-3, "our CEM loop differs from CEMPlanner"
        assert R["check"]["all_token_draft_vs_full_max_abs_loss_diff"] < 1e-2 * R["check"]["loss_spread"], "DraftPredict(all tokens) != full"
        del drafts["full"]

        # ---- scan: full-model CEM; every iteration also scored by each draft
        rec = {s: {m: np.zeros((n, steps)) for m in ms} for s in drafts}; rho = {s: np.zeros((n, steps)) for s in drafts}
        t_d, t_f, mu_full = {s: 0.0 for s in drafts}, 0.0, []
        for i in range(n):
            sc = Scorer(S, var, i, device, amp)

            def select(it, a):
                nonlocal t_f
                lf, dt = timed(lambda: sc.full(a), device); t_f += dt
                true = set(torch.argsort(lf)[:k].tolist()); rf = torch.argsort(torch.argsort(lf)).float()
                for s, d in drafts.items():
                    ld, dt = timed(lambda: sc.draft(a, d), device); t_d[s] += dt
                    order = torch.argsort(ld)
                    for m in ms:
                        rec[s][m][i, it] = len(true & set(order[:m].tolist())) / k
                    rho[s][i, it] = torch.corrcoef(torch.stack([rf, torch.argsort(order).float()]))[0, 1].item()  # Spearman
                return lf
            mu_full.append(cem(seed_of(i), N, k, steps, H, A, device, select)[0])
            print(f"[scan] task {i} " + " ".join(f"{s}:r@{ms[min(2, len(ms) - 1)]}={rec[s][ms[min(2, len(ms) - 1)]][i].mean():.2f}" for s in drafts), flush=True)
        R["scan"] = {s: {"tokens": len(drafts[s][1]), "ctx": drafts[s][2], "cost_vs_full": t_d[s] / t_f,
                         "draft_ms_per_iter": 1e3 * t_d[s] / (n * steps), "spearman_mean": float(rho[s].mean()),
                         "recall": {m: {"mean": float(rec[s][m].mean()), "min": float(rec[s][m].min()),
                                        "exact_share": float((rec[s][m] == 1).mean()),
                                        "by_iteration": rec[s][m].mean(0).round(4).tolist()} for m in ms}} for s in drafts}
        R["full_ms_per_iter"] = 1e3 * t_f / (n * steps)
        (out / "draftscan.json").write_text(json.dumps(R))

        # ---- plans: end to end, same tasks and noise
        configs = [("full", N, k, None, None)]
        configs += [(f"few{f}", int(f), max(2, round(k * int(f) / N)), None, None) for f in args.few.split(",") if f]
        configs += [(f"dv:{s}:m{m}", N, k, s, int(m)) for s in args.plan_drafts.split(",") if s in drafts for m in args.verify_m.split(",")]
        R["plans"] = {}
        for name, Nc, kc, s, m in configs:
            lat, mus, J = [], [], []
            for i in ([0] + list(range(n))):  # the first pass over task 0 is a discarded warm-up
                sc = Scorer(S, var, i, device, amp)
                if s is None:
                    select = lambda it, a: sc.full(a)
                else:
                    def select(it, a, d=drafts[s]):
                        cand = torch.argsort(sc.draft(a, d))[:m]
                        loss = torch.full((len(a),), float("inf"), device=device); loss[cand] = sc.full(a[cand])
                        return loss
                (mu, _), dt = timed(lambda: cem(seed_of(i), Nc, kc, steps, H, A, device, select), device)
                lat.append(dt); mus.append(mu); J.append(sc.full(mu[None]).item())  # quality judged by the full model
            lat, mus, J = lat[1:], mus[1:], J[1:]
            cwd = Path.cwd(); (out / name.replace(":", "_")).mkdir(exist_ok=True)
            import os; os.chdir(out / name.replace(":", "_"))
            logs, succ, _, e_states = S["evaluator"].eval_actions(torch.stack(mus), filename="rollouts"); os.chdir(cwd)
            R["plans"][name] = {"N": Nc, "topk": kc, "draft": s, "verify_m": m, "latency_s": lat, "full_objective": J,
                                "success": [bool(x) for x in succ],
                                "state_dist": np.linalg.norm(S["state_g"] - e_states[:, -1], axis=1).tolist()}
            print(f"== {name}: median {np.median(lat):.3f}s objective {np.mean(J):.5f} success {np.mean(succ):.2f}", flush=True)
            (out / "draftscan.json").write_text(json.dumps(R))
    S["env"].close()

    # ---- summary
    print(f"\nFull model ({args.variant}): {R['full_ms_per_iter']:.1f} ms per CEM iteration at N = {N}; {n} tasks x {steps} iterations.\n")
    print("| draft | tokens/frame | context | cost vs full | Spearman | " + " | ".join(f"recall@{m} mean (min)" for m in ms) + " |")
    print("|---|---|---|---|---|" + "---|" * len(ms))
    for s, d in R["scan"].items():
        print(f"| {s} | {d['tokens']} | {d['ctx']} | {d['cost_vs_full']:.2f} | {d['spearman_mean']:.3f} | "
              + " | ".join(f"{d['recall'][m]['mean']:.3f} ({d['recall'][m]['min']:.2f})" for m in ms) + " |")
    base = R["plans"]["full"]; bl, bj = np.array(base["latency_s"]), np.array(base["full_objective"])
    print("\n| plan | median latency (s) | speedup | final objective, full model | paired difference vs full (se) | success | mean state dist |")
    print("|---|---|---|---|---|---|---|")
    for name, p in R["plans"].items():
        dj = np.array(p["full_objective"]) - bj
        print(f"| {name} | {np.median(p['latency_s']):.2f} | {np.median(bl) / np.median(p['latency_s']):.2f}x | {np.mean(p['full_objective']):.5f} | "
              f"{dj.mean():+.5f} ({dj.std(ddof=1) / np.sqrt(len(dj)) if len(dj) > 1 else 0:.5f}) | {sum(p['success'])}/{len(p['success'])} | {np.mean(p['state_dist']):.1f} |")


if __name__ == "__main__":
    main()
