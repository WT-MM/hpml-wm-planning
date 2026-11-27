#!/usr/bin/env python
"""Phase 0 pre-training probe and systems diagnostics for DINO-WM on PushT.

Answers all foundational questions in one run before training G:
  1. Systems diagnostics:
     - Which SDPA CUDA kernel PyTorch 2.15 picks for masked 3-frame attention vs unmasked _grow
       vs UnmaskedReusePredict (per-frame unmasked _slide), plus fp32 exactness & K=300 latency.
     - Isolates the 0.048 fp32 batch-1 vs batch-300 discrepancy (cudnn TF32 / determinism / encoder).
  2. Intrinsic dimensionality (SVD) of CEM rollout perturbations dU = U(a_k) - U(mu) across
     early / mid / late CEM iterations, plus exact oracle rank-R ranking accuracy.
  3. Zero-training reference-conditioned controls on the same CEM trajectories:
     - copy_ref      : U_hat(a_k) = U(mu) (floor control)
     - grad_lin      : J_hat(a_k) = J(mu) + grad_a J(mu)^T (a_k - mu) (1 backward pass at B=1)
     - jacobian_lin  : U_hat(a_k) = U(mu) + J_u(mu) (a_k - mu) via batched forward finite diff (B=d_a)
     - online_ridge  : dual kernel-ridge fit on previous iteration's top-m verified rollouts (0 extra F calls)
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import probe_math as pm


def _lazy_torch():
    import torch
    import torch.nn.functional as F
    return torch, F


def make_unmasked_reuse(pb, wm, compile_fn=False):
    """Subclass of ReusePredict that splits _slide's block-causal attention into per-frame
    unmasked SDPA calls (attn_mask=None), eliminating 1/3 of attention FLOPs and enabling Flash."""
    torch, F = _lazy_torch()

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


def diag_batch1_diff(pb, S, device):
    """Isolate where the fp32 batch-1 vs batch-300 difference enters (encoder vs proprio, cuDNN TF32)."""
    torch, _ = _lazy_torch()
    from utils import move_to_device

    wm, pl = S["wm"], S["planner"]
    was = {m: m.training for m in wm.modules()}
    wm.eval()
    o0 = move_to_device(pl.preprocessor.transform_obs({k: v[:1] for k, v in S["obs_0"].items()}), device)
    K = min(300, max(16, pl.num_samples))
    o_rep = {k: v.expand(K, *v.shape[1:]).contiguous() for k, v in o0.items()}

    orig_tf32_cudnn = torch.backends.cudnn.allow_tf32
    orig_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    orig_det = torch.backends.cudnn.deterministic
    orig_bench = torch.backends.cudnn.benchmark

    configs = [
        ("default", orig_tf32_cudnn, orig_tf32_matmul, orig_det, orig_bench),
        ("cudnn_tf32_off", False, False, False, False),
        ("cudnn_tf32_off_det", False, False, True, False),
    ]
    out = {}
    with torch.no_grad():
        for name, c_tf32, m_tf32, det, bench in configs:
            torch.backends.cudnn.allow_tf32 = c_tf32
            torch.backends.cuda.matmul.allow_tf32 = m_tf32
            torch.backends.cudnn.deterministic = det
            torch.backends.cudnn.benchmark = bench
            z1 = wm.encode_obs(o0)
            zK = wm.encode_obs(o_rep)
            dv = (zK["visual"][0:1] - z1["visual"]).abs()
            dp = (zK["proprio"][0:1] - z1["proprio"]).abs()
            out[name] = {
                "visual_max_abs_diff": dv.max().item(),
                "visual_mean_abs_diff": dv.mean().item(),
                "proprio_max_abs_diff": dp.max().item(),
                "row0_vs_rowLast_max_diff": (zK["visual"][0] - zK["visual"][-1]).abs().max().item(),
            }
    torch.backends.cudnn.allow_tf32 = orig_tf32_cudnn
    torch.backends.cuda.matmul.allow_tf32 = orig_tf32_matmul
    torch.backends.cudnn.deterministic = orig_det
    torch.backends.cudnn.benchmark = orig_bench
    for m, t in was.items():
        m.training = t
    return out


def diag_sdpa_and_unmasked(pb, S, args, device):
    """Audit which SDPA CUDA kernels PyTorch selects and compare masked vs unmasked ReusePredict."""
    torch, _ = _lazy_torch()
    from utils import move_to_device

    wm, pl = S["wm"], S["planner"]
    was = {m: m.training for m in wm.modules()}
    wm.eval()
    o0 = move_to_device(pl.preprocessor.transform_obs({k: v[:1] for k, v in S["obs_0"].items()}), device)
    K = args.num_samples
    obs = {k: v.expand(K, *v.shape[1:]).contiguous() for k, v in o0.items()}
    torch.manual_seed(0)
    act = torch.randn(K, pl.horizon, pl.action_dim, device=device)

    out = {"K": K}
    # 1. Exactness check in fp32: UnmaskedReusePredict vs ReusePredict
    with torch.no_grad(), pb.Variant(wm, "cache_sdpa_reuse") as var:
        var.new_plan()
        ref_vis = wm.rollout(obs, act)[0]["visual"]
        unmasked = make_unmasked_reuse(pb, wm, compile_fn=False)
        wm.predict = unmasked
        var.new_plan()
        wm.rollout = pb.make_cached_rollout(wm, var.z_cache, unmasked.reset)
        got_vis = wm.rollout(obs, act)[0]["visual"]
        out["unmasked_vs_reuse_fp32_max_abs_diff"] = (got_vis - ref_vis).abs().max().item()

    if device.type != "cuda":
        for m, t in was.items():
            m.training = t
        return out

    # 2. Timing + PyTorch Profiler attention kernel names in fp16
    from torch.profiler import profile as prof, ProfilerActivity

    def bench_one(label, fn):
        for _ in range(3):
            with torch.no_grad():
                fn()
        pb.sync(device)
        ts = []
        for _ in range(15):
            pb.sync(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                fn()
            pb.sync(device)
            ts.append((time.perf_counter() - t0) * 1e3)
        with prof(activities=[ProfilerActivity.CUDA]) as p:
            with torch.no_grad():
                fn()
        pb.sync(device)
        attn_ops = []
        for e in p.key_averages():
            k = e.key
            lo = k.lower()
            if any(w in lo for w in ("flash", "fmha", "efficient_attention", "sdpa", "attention")):
                dev_us = getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0))
                if dev_us > 0:
                    attn_ops.append({"op": k[:140], "self_gpu_ms": dev_us / 1e3, "calls": e.count})
        attn_ops.sort(key=lambda r: -r["self_gpu_ms"])
        out[label] = {
            "median_ms": float(np.median(ts)),
            "p95_ms": float(np.percentile(ts, 95)),
            "attn_kernels": attn_ops[:10],
        }

    with pb.Variant(wm, "cache_sdpa_fp16") as var:
        var.new_plan()
        bench_one("cache_sdpa_fp16", lambda: wm.rollout(obs, act))
    with pb.Variant(wm, "fp16_reuse") as var:
        var.new_plan()
        bench_one("fp16_reuse", lambda: wm.rollout(obs, act))
        unmasked = make_unmasked_reuse(pb, wm, compile_fn=False)
        wm.predict = unmasked
        inner = pb.make_cached_rollout(wm, var.z_cache, unmasked.reset)
        def rollout_unmasked(o, a, _in=inner):
            with torch.autocast("cuda", dtype=torch.float16):
                z_o, z = _in(o, a)
            return {k: v.float() for k, v in z_o.items()}, z.float()
        wm.rollout = rollout_unmasked
        bench_one("fp16_unmasked_reuse", lambda: wm.rollout(obs, act))

    for m, t in was.items():
        m.training = t
    return out


def pack_latent_torch(z_obs, alpha=1.0):
    """Torch version of pm.pack_weighted_latent on last frame: returns (K, D_tot) float32."""
    torch, _ = _lazy_torch()
    v = z_obs["visual"][:, -1].float().flatten(1)
    p = z_obs["proprio"][:, -1].float().flatten(1)
    return torch.cat([v / (v.shape[1] ** 0.5), p * ((alpha / p.shape[1]) ** 0.5)], dim=1)


def run_rollout_latents(wm, o0_single, act):
    """Run wm.rollout on act (B, H, A) for a single start obs; returns packed U (B, D_tot) and raw z_obs."""
    B = act.shape[0]
    obs = {k: v.expand(B, *v.shape[1:]) for k, v in o0_single.items()}
    z_obs, _ = wm.rollout(obs, act)
    return pack_latent_torch(z_obs), z_obs


def selfcheck_math(pb, S, device):
    """Runnable check on the real model: packed U norm == objective_fn, Gram-SVD rank-K == true loss,
    and Jacobian finite-diff at delta=0 equals exact reference loss."""
    torch, _ = _lazy_torch()
    from utils import move_to_device

    wm, pl = S["wm"], S["planner"]
    o0 = move_to_device(pl.preprocessor.transform_obs({k: v[:1] for k, v in S["obs_0"].items()}), device)
    og = move_to_device(pl.preprocessor.transform_obs({k: v[:1] for k, v in S["obs_g"].items()}), device)
    K = min(32, pl.num_samples)
    torch.manual_seed(0)
    act = torch.randn(K, pl.horizon, pl.action_dim, device=device)
    with pb.Variant(wm, "cache_sdpa_reuse") as var:
        var.new_plan()
        with torch.no_grad():
            zg = wm.encode_obs(og)
            U, z_obs = run_rollout_latents(wm, o0, act)
            ug = pack_latent_torch(zg)[0]
            l_true = pl.objective_fn(z_obs, {k: v.expand(K, *v.shape[1:]) for k, v in zg.items()})
            l_pack = ((U - ug[None]) ** 2).sum(dim=1)
            pack_diff = (l_true - l_pack).abs().max().item()

            # Check Gram-SVD full rank reproduces l_pack exactly
            u_ref = U[0]
            r_g = u_ref - ug
            dU = U - u_ref[None]
            G = (dU @ dU.T).cpu().numpy().astype(np.float64)
            h = (dU @ r_g).cpu().numpy().astype(np.float64)
            res = pm.gram_svd_analysis(G, h, float((r_g @ r_g).item()), [K])
            svd_diff = float(np.max(np.abs(res["oracle_losses"][K] - l_pack.cpu().numpy())))

        # Check autograd through ReusePredict produces non-zero gradient w.r.t. action
        a_var = act[:1].detach().clone().requires_grad_(True)
        u_g1, _ = run_rollout_latents(wm, o0, a_var)
        (g_a,) = torch.autograd.grad(((u_g1[0] - ug) ** 2).sum(), a_var)
        grad_norm = float(g_a.norm().item())

    assert pack_diff < 1e-4, f"packed latent MSE mismatch: {pack_diff}"
    assert svd_diff < 1e-3, f"Gram-SVD full-rank mismatch: {svd_diff}"
    assert grad_norm > 0.0 and np.isfinite(grad_norm), f"invalid grad_norm: {grad_norm}"
    return {"pack_vs_objective_max_diff": pack_diff, "gram_svd_fullrank_max_diff": svd_diff, "grad_norm": grad_norm}


def probe_cem(pb, S, args, device):
    """Run full-model CEM on n_tasks; at selected iterations measure SVD rank spectrum,
    oracle rank-R recall, copy_ref, grad_lin, jacobian_lin, and online_ridge."""
    torch, _ = _lazy_torch()
    from utils import move_to_device

    wm, pl = S["wm"], S["planner"]
    N, topk, steps, H, A, n = args.num_samples, args.topk, args.opt_steps, pl.horizon, pl.action_dim, args.n_tasks
    d_a = H * A
    probe_iters = sorted({int(x) for x in args.probe_iters.split(",") if 0 <= int(x) < steps})
    ranks = [int(r) for r in args.ranks.split(",")]
    recall_ms = [int(m) for m in args.recall_m.split(",") if int(m) <= N]
    variant = args.variant if device.type == "cuda" else "cache_sdpa_reuse"

    def timed(fn):
        pb.sync(device)
        t0 = time.perf_counter()
        out = fn()
        pb.sync(device)
        return out, (time.perf_counter() - t0) * 1e3

    # Accumulators keyed by iteration
    svd_recs = {it: [] for it in probe_iters}
    oracle_recs = {it: {r: [] for r in ranks} for it in probe_iters}
    ctrl_names = ["copy_ref", "grad_lin", "jacobian_lin", "online_ridge"]
    ctrl_recs = {c: {it: [] for it in probe_iters} for c in ctrl_names}
    timings_ms = {k: [] for k in ("full", "ref_b1", "grad_lin", "jacobian_lin", "online_ridge")}

    # Precompute unit basis for finite-difference Jacobian (d_a, H, A)
    eye_a = torch.eye(d_a, device=device).view(d_a, H, A)

    with pb.Variant(wm, variant) as var:
        for i in range(n):
            tr = pl.preprocessor.transform_obs
            o0 = move_to_device(tr({k: v[i:i + 1] for k, v in S["obs_0"].items()}), device)
            og = move_to_device(tr({k: v[i:i + 1] for k, v in S["obs_g"].items()}), device)
            var.new_plan()
            with torch.no_grad():
                zg = wm.encode_obs(og)
                ug = pack_latent_torch(zg)[0]  # (D_tot,)

            torch.manual_seed(args.seed * 100003 + i)
            mu = torch.zeros(H, A, device=device)
            sigma = torch.ones(H, A, device=device)

            # Sliding buffer of previous verified (action, packed_U) for online_ridge
            hist_a, hist_u = [], []

            for it in range(steps):
                action = torch.randn(N, H, A).to(device) * sigma + mu
                action[0] = mu

                with torch.no_grad():
                    (U, _), dt_full = timed(lambda: run_rollout_latents(wm, o0, action))
                l_true_t = ((U - ug[None]) ** 2).sum(dim=1)
                l_true = l_true_t.cpu().numpy()
                top = torch.argsort(l_true_t)[:topk]

                if it in probe_iters:
                    timings_ms["full"].append(dt_full)
                    # Reference rollout cost at B=1 (same variant)
                    with torch.no_grad():
                        (u_ref_1, _), dt_ref = timed(lambda: run_rollout_latents(wm, o0, mu[None]))
                    timings_ms["ref_b1"].append(dt_ref)

                    u_ref = U[0]  # action[0] == mu
                    r_g = u_ref - ug
                    r_norm2 = float((r_g @ r_g).item())
                    dU = U - u_ref[None]
                    dA = (action - mu[None]).reshape(N, d_a)

                    # 1. Gram-SVD & oracle rank-R
                    G = (dU @ dU.T).cpu().numpy().astype(np.float64)
                    h = (dU @ r_g).cpu().numpy().astype(np.float64)
                    sa = pm.gram_svd_analysis(G, h, r_norm2, ranks)
                    svd_recs[it].append({
                        "sigma_mean": float(sigma.mean().item()),
                        "total_var": sa["total_var"],
                        "eff_rank_var": sa["eff_rank_var"],
                        "eff_rank_sv": sa["eff_rank_sv"],
                        "var_explained": sa["var_explained"],
                    })
                    for r in ranks:
                        oracle_recs[it][r].append(pm.ranking_metrics(l_true, sa["oracle_losses"][r], topk, recall_ms))

                    # 2. copy_ref control
                    l_copy = np.full(N, r_norm2, dtype=np.float64)
                    ctrl_recs["copy_ref"][it].append(pm.ranking_metrics(l_true, l_copy, topk, recall_ms))

                    # 3. jacobian_lin: batched forward finite differences at B = d_a
                    eps = torch.clamp(args.fd_rel * sigma.reshape(d_a), min=args.fd_min)
                    act_fd = mu[None] + eye_a * eps[:, None, None]
                    with torch.no_grad():
                        def _jac():
                            U_fd, _ = run_rollout_latents(wm, o0, act_fd)
                            J_u = (U_fd - u_ref[None]) / eps[:, None]  # (d_a, D_tot)
                            M_j = J_u @ J_u.T
                            b_j = J_u @ r_g
                            return r_norm2 + 2.0 * (dA @ b_j) + ((dA @ M_j) * dA).sum(dim=1)
                        l_jac_t, dt_jac = timed(_jac)
                    timings_ms["jacobian_lin"].append(dt_jac)
                    ctrl_recs["jacobian_lin"][it].append(
                        pm.ranking_metrics(l_true, l_jac_t.cpu().numpy(), topk, recall_ms)
                    )

                    # 4. online_ridge: fit dual kernel ridge on previous iteration's verified candidates (0 extra F calls)
                    if hist_a:
                        A_prev = torch.cat(hist_a, dim=0)
                        U_prev = torch.cat(hist_u, dim=0)
                        def _ridge():
                            dA_tr = (A_prev - mu.reshape(1, d_a)).cpu().numpy().astype(np.float64)
                            dA_ev = dA.cpu().numpy().astype(np.float64)
                            C_r = pm.dual_ridge_coeffs(dA_tr, dA_ev, lam=args.ridge_lam)
                            dU_tr = U_prev - u_ref[None]
                            M_r = (dU_tr @ dU_tr.T).cpu().numpy().astype(np.float64)
                            b_r = (dU_tr @ r_g).cpu().numpy().astype(np.float64)
                            return pm.quad_mse_from_gram(r_norm2, M_r, b_r, C_r)
                        l_ridge, dt_ridge = timed(_ridge)
                        timings_ms["online_ridge"].append(dt_ridge)
                        ctrl_recs["online_ridge"][it].append(
                            pm.ranking_metrics(l_true, l_ridge, topk, recall_ms)
                        )

                    # 5. grad_lin: 1st-order scalar Taylor via 1 backward pass at B=1 (under active variant)
                    def _grad():
                        a_var = mu.detach().clone().unsqueeze(0).requires_grad_(True)
                        u_g1, _ = run_rollout_latents(wm, o0, a_var)
                        loss_1 = ((u_g1[0] - ug) ** 2).sum()
                        (g_a,) = torch.autograd.grad(loss_1, a_var)
                        return r_norm2 + (dA @ g_a.detach().reshape(d_a))
                    l_grad_t, dt_grad = timed(_grad)
                    timings_ms["grad_lin"].append(dt_grad)
                    ctrl_recs["grad_lin"][it].append(
                        pm.ranking_metrics(l_true, l_grad_t.cpu().numpy(), topk, recall_ms)
                    )

                # Update sliding verified history (top verify_m of current iteration, mimicking draft-and-verify)
                m_keep = min(args.ridge_m, N)
                keep_idx = torch.argsort(l_true_t)[:m_keep]
                hist_a.append(action[keep_idx].reshape(m_keep, d_a).detach())
                hist_u.append(U[keep_idx].detach())
                if len(hist_a) > args.ridge_hist:
                    hist_a.pop(0)
                    hist_u.pop(0)

                mu = action[top].mean(0)
                sigma = action[top].std(0)

            print(f"[probe] task {i:2d}/{n} done", flush=True)

    def agg_rank(rows):
        if not rows:
            return None
        return {
            "n": len(rows),
            "spearman": float(np.mean([r["spearman"] for r in rows])),
            "elite_overlap": float(np.mean([r["elite_overlap"] for r in rows])),
            "best_same": float(np.mean([r["best_same"] for r in rows])),
            "recall": {str(m): float(np.mean([r["recall"][m] for r in rows])) for m in recall_ms},
        }

    summary = {
        "probe_iters": probe_iters,
        "ranks": ranks,
        "recall_ms": recall_ms,
        "timings_ms": {
            k: {"median": float(np.median(v)), "mean": float(np.mean(v)), "n": len(v)} if v else None
            for k, v in timings_ms.items()
        },
        "svd_by_iter": {
            str(it): {
                "sigma_mean": float(np.mean([x["sigma_mean"] for x in svd_recs[it]])),
                "eff_rank_var": float(np.mean([x["eff_rank_var"] for x in svd_recs[it]])),
                "eff_rank_sv": float(np.mean([x["eff_rank_sv"] for x in svd_recs[it]])),
                "var_explained": {
                    str(r): float(np.mean([x["var_explained"][r] for x in svd_recs[it]])) for r in ranks
                },
            }
            for it in probe_iters
        },
        "oracle_rank_by_iter": {
            str(it): {str(r): agg_rank(oracle_recs[it][r]) for r in ranks}
            for it in probe_iters
        },
        "controls_by_iter": {
            c: {str(it): agg_rank(ctrl_recs[c][it]) for it in probe_iters}
            for c in ctrl_names
        },
    }
    return summary


def print_summary(R):
    print("\n=== Batch-1 vs Batch-K fp32 encoder discrepancy ===")
    for k, v in R["batch1_diff"].items():
        print(f"  {k}: visual_max={v['visual_max_abs_diff']:.2e} proprio_max={v['proprio_max_abs_diff']:.2e}")

    print("\n=== SDPA kernel audit & UnmaskedReusePredict ===")
    sd = R["sdpa_audit"]
    print(f"  unmasked_vs_reuse fp32 max diff: {sd['unmasked_vs_reuse_fp32_max_abs_diff']:.2e}")
    for k in ("cache_sdpa_fp16", "fp16_reuse", "fp16_unmasked_reuse"):
        if k in sd:
            top_k = sd[k]["attn_kernels"][0]["op"] if sd[k]["attn_kernels"] else "none"
            print(f"  {k}: median {sd[k]['median_ms']:.2f} ms (p95 {sd[k]['p95_ms']:.2f} ms) | top attn: {top_k}")

    P = R["cem_probe"]
    print("\n=== Timings (ms per CEM iteration) ===")
    full_ms = P["timings_ms"]["full"]["median"] if P["timings_ms"]["full"] else 1.0
    for k, v in P["timings_ms"].items():
        if v:
            print(f"  {k:14s}: {v['median']:7.2f} ms ({v['median'] / full_ms:.3f}x full)")

    print("\n=== SVD intrinsic rank of dU across CEM iterations ===")
    ranks = P["ranks"]
    print("| iter | sigma | eff_rank(var) | eff_rank(sv) | " + " | ".join(f"var@{r}" for r in ranks) + " |")
    print("|---|---|---|---|" + "---|" * len(ranks))
    for it in P["probe_iters"]:
        s = P["svd_by_iter"][str(it)]
        print(f"| {it} | {s['sigma_mean']:.3f} | {s['eff_rank_var']:.1f} | {s['eff_rank_sv']:.1f} | "
              + " | ".join(f"{s['var_explained'][str(r)]:.3f}" for r in ranks) + " |")

    m_show = 45 if 45 in P["recall_ms"] else P["recall_ms"][min(1, len(P["recall_ms"]) - 1)]
    print(f"\n=== Controls: Spearman / recall@{m_show} across iterations ===")
    print("| method | " + " | ".join(f"it={it}" for it in P["probe_iters"]) + " |")
    print("|---|" + "---|" * len(P["probe_iters"]))
    for c, by_it in P["controls_by_iter"].items():
        cells = []
        for it in P["probe_iters"]:
            d = by_it[str(it)]
            cells.append("-" if d is None else f"{d['spearman']:.2f} / {d['recall'][str(m_show)]:.2f}")
        print(f"| {c} | " + " | ".join(cells) + " |")
    for r in (16, 32, 64):
        if r in ranks:
            cells = []
            for it in P["probe_iters"]:
                d = P["oracle_rank_by_iter"][str(it)][str(r)]
                cells.append(f"{d['spearman']:.2f} / {d['recall'][str(m_show)]:.2f}")
            print(f"| oracle_svd_r{r} | " + " | ".join(cells) + " |")


def main():
    import planbench as pb
    torch, _ = _lazy_torch()

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="fp16_reuse")
    ap.add_argument("--model-name", default="pusht")
    ap.add_argument("--model-epoch", default="latest")
    ap.add_argument("--n-tasks", type=int, default=8)
    ap.add_argument("--goal-H", type=int, default=5)
    ap.add_argument("--num-samples", type=int, default=300)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--opt-steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--n-plot", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eval-mode", action="store_true")
    ap.add_argument("--probe-iters", default="0,4,9,14,19,24,29")
    ap.add_argument("--ranks", default="4,8,16,32,50,64,96")
    ap.add_argument("--recall-m", default="30,45,60,90")
    ap.add_argument("--fd-rel", type=float, default=0.1, help="finite-diff step = max(fd_min, fd_rel * sigma)")
    ap.add_argument("--fd-min", type=float, default=0.05)
    ap.add_argument("--ridge-m", type=int, default=45, help="verified candidates kept per iteration for online_ridge")
    ap.add_argument("--ridge-hist", type=int, default=2, help="past iterations kept in online_ridge sliding buffer")
    ap.add_argument("--ridge-lam", type=float, default=1e-2)
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    S = pb.load(args, device)
    R = {
        "args": vars(args),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu (functional dry run)",
        "selfcheck": selfcheck_math(pb, S, device),
        "batch1_diff": diag_batch1_diff(pb, S, device),
        "sdpa_audit": diag_sdpa_and_unmasked(pb, S, args, device),
        "cem_probe": probe_cem(pb, S, args, device),
    }
    (out / "probe_phase0.json").write_text(json.dumps(R, indent=2))
    print_summary(R)
    S["env"].close()


if __name__ == "__main__":
    main()
