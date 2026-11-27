"""Pure-NumPy math helpers for Phase 1 speculative planning: low-rank training loss identity,
hybrid basis stacking (G + online verified residual), stratified candidate selection, and split guards.
"""
import numpy as np

HELDOUT_TRAIN_N = 500


def lowrank_mse_loss_np(B, C, dU):
    """Exact mean squared error mean_k ||C[k] @ B - dU[k]||_2^2 computed without forming (K, D).
    B: (R, D), C: (K, R), dU: (K, D). Returns per-candidate squared error (K,)."""
    M = B @ B.T                    # (R, R)
    P = dU @ B.T                   # (K, R)
    pred_n2 = np.einsum("kr,rs,ks->k", C, M, C)
    cross = np.sum(C * P, axis=1)
    tgt_n2 = np.sum(dU * dU, axis=1)
    return np.maximum(pred_n2 - 2.0 * cross + tgt_n2, 0.0)


def hybrid_quad_mse_np(r_g, B_g, C_ev, C_tr, Y_tr, W_ev):
    """Exact MSE of hybrid prediction: dU_hyb = C_ev @ B_g + W_ev @ (Y_tr - C_tr @ B_g)
    using stacked basis [B_g; Y_tr] without materializing (M, D) residuals."""
    B_cat = np.concatenate([B_g, Y_tr], axis=0)            # (R + M, D)
    C_cat = np.concatenate([C_ev - W_ev @ C_tr, W_ev], axis=1)  # (K, R + M)
    M_cat = B_cat @ B_cat.T
    b_cat = B_cat @ r_g
    r_n2 = float(r_g @ r_g)
    return r_n2 + 2.0 * (C_cat @ b_cat) + np.einsum("kr,rs,ks->k", C_cat, M_cat, C_cat)


def stratified_candidate_indices(losses, k_save, rng):
    """Select k_save candidate indices (excluding index 0 = mu): half from top elites,
    half uniformly stratified across the remaining candidates for full-distribution coverage."""
    losses = np.asarray(losses, dtype=np.float64)
    N = len(losses)
    order = np.argsort(losses, kind="mergesort")
    order = order[order != 0]  # exclude reference mu (index 0)
    k_save = min(int(k_save), len(order))
    n_top = k_save // 2
    n_rest = k_save - n_top
    top_idx = order[:n_top]
    rest_pool = order[n_top:]
    if n_rest == 0 or len(rest_pool) == 0:
        return top_idx.astype(np.int64)
    # Stratified quantiles across rest_pool with random jitter inside each bin
    bins = np.array_split(rest_pool, n_rest)
    rest_idx = np.array([b[rng.randrange(len(b))] for b in bins if len(b) > 0], dtype=np.int64)
    return np.concatenate([top_idx.astype(np.int64), rest_idx])


def assert_train_g_only(episode_ids, n_train_total, heldout_n=HELDOUT_TRAIN_N):
    """Guard ensuring data collection / training never touches held-out evaluation episodes."""
    cut = n_train_total - heldout_n
    bad = [int(e) for e in episode_ids if int(e) < 0 or int(e) >= cut]
    if bad:
        raise ValueError(f"Episode leakage detected: {bad[:5]} outside train_G [0, {cut})")
