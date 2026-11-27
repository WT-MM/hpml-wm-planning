#!/usr/bin/env python
"""Train RefBasisG (reference-conditioned low-rank basis predictor) or equal-cost NoRefBasisG
on compact on-policy CEM rollout shards collected by collect_cem_data.py.

Splits train/dev strictly by episode ID, optimizes exact low-rank weighted latent MSE (with
per-reference scale normalization so late-CEM small-sigma steps get equal weight) plus optional
planning-aware elite ranking loss, and evaluates real CEM Spearman & top-k recall every epoch.
"""
import argparse, json, math, random, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import specplan_models as smod


def load_dataset(data_dirs, dev_frac=0.1, seed=0):
    records, manifests = [], []
    for d in str(data_dirs).split(","):
        data_dir = Path(d.strip())
        manifest = json.loads((data_dir / "manifest.json").read_text())
        manifests.append(manifest)
        for sh in manifest["shards"]:
            records.extend(torch.load(data_dir / sh["file"], map_location="cpu", weights_only=False))
    eps = sorted({int(r["episode"]) for r in records})
    rng = random.Random(seed)
    rng.shuffle(eps)
    n_dev = max(1, int(round(len(eps) * dev_frac))) if len(eps) > 1 else 0
    dev_eps = set(eps[:n_dev])
    train_recs = [r for r in records if int(r["episode"]) not in dev_eps]
    dev_recs = [r for r in records if int(r["episode"]) in dev_eps] or train_recs[:min(8, len(train_recs))]
    return train_recs, dev_recs, manifests


def collate_batch(recs, device):
    return {
        "z0_vis": torch.stack([r["z0_vis"] for r in recs]).to(device, non_blocking=True),
        "z0_prop": torch.stack([r["z0_prop"] for r in recs]).to(device, non_blocking=True),
        "zg_vis": torch.stack([r["zg_vis"] for r in recs]).to(device, non_blocking=True),
        "zg_prop": torch.stack([r["zg_prop"] for r in recs]).to(device, non_blocking=True),
        "mu": torch.stack([r["mu"] for r in recs]).to(device, non_blocking=True),
        "zref_vis": torch.stack([r["zref_vis"] for r in recs]).to(device, non_blocking=True),
        "zref_prop": torch.stack([r["zref_prop"] for r in recs]).to(device, non_blocking=True),
        "cand_act": torch.stack([r["cand_act"] for r in recs]).to(device, non_blocking=True),
        "cand_vis": torch.stack([r["cand_vis"] for r in recs]).to(device, non_blocking=True),
        "cand_prop": torch.stack([r["cand_prop"] for r in recs]).to(device, non_blocking=True),
        "cem_iter": [int(r["cem_iter"]) for r in recs],
    }


def forward_batch(model, kind, batch, args):
    """Compute exact low-rank MSE loss + predicted/true candidate objective scores for ranking metrics."""
    u_cand = smod.pack_latent(batch["cand_vis"], batch["cand_prop"], alpha=1.0)  # (B, K, D)
    ug = smod.pack_latent(batch["zg_vis"], batch["zg_prop"], alpha=1.0)          # (B, D)
    true_obj = ((u_cand - ug.unsqueeze(1)) ** 2).sum(dim=-1)                     # (B, K)

    if kind in ("refg", "refg_lin"):
        basis, ctx, W = model.encode_reference(
            batch["z0_vis"], batch["z0_prop"], batch["zref_vis"], batch["zref_prop"], batch["mu"],
        )
        dA = batch["cand_act"] - batch["mu"].unsqueeze(1)
        C = model.candidate_coeffs(ctx, W, dA)
        u_base = smod.pack_latent(batch["zref_vis"], batch["zref_prop"], alpha=1.0)
    else:
        basis, ctx, W = model.encode_start(batch["z0_vis"], batch["z0_prop"])
        C = model.candidate_coeffs(ctx, W, batch["cand_act"])
        u_base = smod.pack_latent(batch["z0_vis"], batch["z0_prop"], alpha=1.0)

    dU = u_cand - u_base.unsqueeze(1)
    mse_loss, raw_se = smod.lowrank_mse_loss(basis, C, dU, normalize_scale=args.norm_scale)

    # Exact predicted candidate objectives via quadratic Gram form: (B, K)
    B32, C32 = basis.float(), C.float()
    r_g = u_base - ug                                             # (B, D)
    M = torch.bmm(B32, B32.transpose(1, 2))                       # (B, R, R)
    b = torch.bmm(B32, r_g.unsqueeze(-1)).squeeze(-1)             # (B, R)
    r_n2 = (r_g * r_g).sum(dim=-1, keepdim=True)                  # (B, 1)
    pred_obj = r_n2 + 2.0 * torch.bmm(C32, b.unsqueeze(-1)).squeeze(-1) + (torch.bmm(C32, M) * C32).sum(dim=-1)

    total_loss = mse_loss
    rank_loss = torch.zeros((), device=mse_loss.device)
    if args.rank_weight > 0.0:
        rank_loss = smod.pairwise_rank_loss(pred_obj, true_obj, topk=max(4, u_cand.shape[1] // 4))
        total_loss = total_loss + args.rank_weight * rank_loss

    return total_loss, mse_loss, float(raw_se.mean().item()), float(rank_loss.item()), pred_obj.detach(), true_obj.detach()


def spearman_and_recall(pred_obj, true_obj):
    """Per-reference Spearman and top-25% elite overlap (pred/true shape (B, K))."""
    p_np, t_np = pred_obj.cpu().numpy(), true_obj.cpu().numpy()
    rhos, overlaps = [], []
    k_top = max(2, p_np.shape[1] // 4)
    for p, t in zip(p_np, t_np):
        rp = np.argsort(np.argsort(p)).astype(np.float64)
        rt = np.argsort(np.argsort(t)).astype(np.float64)
        rho = 0.0 if (rp.std() == 0 or rt.std() == 0) else float(np.corrcoef(rp, rt)[0, 1])
        ov = len(set(np.argsort(p)[:k_top]) & set(np.argsort(t)[:k_top])) / k_top
        rhos.append(rho)
        overlaps.append(ov)
    return rhos, overlaps


def selfcheck_model(model, kind, sample_rec, device):
    """Runnable check: exact zero residual at dA=0 (for RefBasisG) and lowrank_mse_loss == dense MSE."""
    model.eval()
    batch = collate_batch([sample_rec], device)
    with torch.no_grad():
        if kind in ("refg", "refg_lin"):
            basis, ctx, W = model.encode_reference(
                batch["z0_vis"], batch["z0_prop"], batch["zref_vis"], batch["zref_prop"], batch["mu"],
            )
            c0 = model.candidate_coeffs(ctx, W, torch.zeros(1, 1, batch["mu"].shape[-1], device=device))
            assert c0.abs().max().item() < 1e-6, f"RefBasisG nonzero at dA=0: {c0.abs().max().item()}"
            dA = batch["cand_act"] - batch["mu"].unsqueeze(1)
            C = model.candidate_coeffs(ctx, W, dA)
            u_base = smod.pack_latent(batch["zref_vis"], batch["zref_prop"])
        else:
            basis, ctx, W = model.encode_start(batch["z0_vis"], batch["z0_prop"])
            C = model.candidate_coeffs(ctx, W, batch["cand_act"])
            u_base = smod.pack_latent(batch["z0_vis"], batch["z0_prop"])
        u_cand = smod.pack_latent(batch["cand_vis"], batch["cand_prop"])
        dU = u_cand - u_base.unsqueeze(1)
        _, se_lr = smod.lowrank_mse_loss(basis, C, dU, normalize_scale=False)
        dense_se = ((torch.bmm(C.float(), basis.float()) - dU.float()) ** 2).sum(dim=-1)
        diff = (se_lr - dense_se).abs().max().item()
        assert diff < 1e-3, f"lowrank_mse_loss mismatch vs dense: {diff}"
    return {"lowrank_vs_dense_max_diff": diff}


def evaluate(model, kind, recs, args, device):
    model.eval()
    losses, raw_mses, rhos, overlaps = [], [], [], []
    by_iter = {}
    with torch.no_grad():
        for i in range(0, len(recs), args.batch_size):
            b = collate_batch(recs[i:i + args.batch_size], device)
            tot, mse, raw_mse, _, p_obj, t_obj = forward_batch(model, kind, b, args)
            losses.append(float(tot.item()))
            raw_mses.append(raw_mse)
            r_list, ov_list = spearman_and_recall(p_obj, t_obj)
            rhos.extend(r_list)
            overlaps.extend(ov_list)
            for it, rh, ov in zip(b["cem_iter"], r_list, ov_list):
                by_iter.setdefault(it, []).append((rh, ov))
    return {
        "loss": float(np.mean(losses)),
        "raw_mse": float(np.mean(raw_mses)),
        "spearman": float(np.mean(rhos)),
        "elite_overlap": float(np.mean(overlaps)),
        "spearman_by_iter": {str(k): float(np.mean([x[0] for x in v])) for k, v in sorted(by_iter.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kind", default="refg", choices=["refg", "refg_lin", "norefg"])
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--n-blocks", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--norm-scale", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--rank-weight", type=float, default=0.0, help="0.0 = base latent MSE; >0 = planning-aware ranking loss")
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    train_recs, dev_recs, manifest = load_dataset(args.data, dev_frac=args.dev_frac, seed=args.seed)
    r0 = train_recs[0]
    P, d_v = r0["z0_vis"].shape
    d_p = r0["z0_prop"].shape[0]
    d_a = r0["mu"].shape[0]
    grid = int(round(math.sqrt(P)))
    assert grid * grid == P, f"expected square patch grid, got P={P}"

    if args.kind in ("refg", "refg_lin"):
        model = smod.RefBasisG(
            d_v=d_v, d_p=d_p, d_a=d_a, grid=grid, rank=args.rank,
            hidden=args.hidden, n_blocks=args.n_blocks, linear_only=(args.kind == "refg_lin"),
        ).to(device)
    else:
        model = smod.NoRefBasisG(
            d_v=d_v, d_p=d_p, d_a=d_a, grid=grid, rank=args.rank,
            hidden=args.hidden, n_blocks=args.n_blocks,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    chk = selfcheck_model(model, args.kind, r0, device)
    print(f"[train] kind={args.kind} params={n_params:,} train_groups={len(train_recs)} dev_groups={len(dev_recs)} selfcheck={chk}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_rho = -2.0
    history = []
    t0 = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train_recs)))
        random.shuffle(order)
        tr_losses = []
        for i in range(0, len(order), args.batch_size):
            batch = collate_batch([train_recs[j] for j in order[i:i + args.batch_size]], device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                tot, _, _, _, _, _ = forward_batch(model, args.kind, batch, args)
            scaler.scale(tot).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            tr_losses.append(float(tot.item()))
        sched.step()

        dev = evaluate(model, args.kind, dev_recs, args, device)
        row = {"epoch": ep, "train_loss": float(np.mean(tr_losses)), "elapsed_s": time.perf_counter() - t0, **dev}
        history.append(row)
        if dev["spearman"] > best_rho:
            best_rho = dev["spearman"]
            torch.save({
                "kind": args.kind,
                "config": {"d_v": d_v, "d_p": d_p, "d_a": d_a, "grid": grid, "rank": args.rank,
                           "hidden": args.hidden, "n_blocks": args.n_blocks, "linear_only": args.kind == "refg_lin"},
                "state_dict": model.state_dict(),
                "best_dev": dev,
                "args": vars(args),
            }, out / "best.pt")
        if ep == 1 or ep % max(1, args.epochs // 10) == 0 or ep == args.epochs:
            print(f"[ep {ep:3d}/{args.epochs}] tr_loss={row['train_loss']:.4f} dev_loss={dev['loss']:.4f} "
                  f"dev_rho={dev['spearman']:.3f} ov={dev['elite_overlap']:.3f} by_it={dev['spearman_by_iter']}", flush=True)

    (out / "train_log.json").write_text(json.dumps({
        "args": vars(args), "n_params": n_params, "selfcheck": chk,
        "train_groups": len(train_recs), "dev_groups": len(dev_recs),
        "best_dev_spearman": best_rho, "history": history,
    }, indent=2))
    print(f"[train] finished in {time.perf_counter() - t0:.1f}s, best dev Spearman={best_rho:.3f}", flush=True)


if __name__ == "__main__":
    main()
