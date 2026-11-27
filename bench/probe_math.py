"""Pure-NumPy math for Phase 0 probes: weighted latent packing, exact Gram-SVD rank-R MSE,
low-rank quadratic basis scoring, dual kernel-ridge local surrogate, and CEM ranking metrics.
"""
import numpy as np


def pack_weighted_latent(vis, prop, alpha=1.0):
    """Flatten (K, ..., visual) and (K, ..., proprio) into U of shape (K, D_tot) such that
    ||U[k] - U_g||_2^2 == MSE(vis[k], vis_g) + alpha * MSE(prop[k], prop_g) exactly."""
    vis = np.asarray(vis, dtype=np.float64)
    prop = np.asarray(prop, dtype=np.float64)
    K = vis.shape[0]
    v = vis.reshape(K, -1)
    p = prop.reshape(K, -1)
    return np.concatenate([v / np.sqrt(v.shape[1]), p * np.sqrt(alpha / p.shape[1])], axis=1)


def quad_mse_from_gram(r_norm2, M, b, C):
    """Exact MSE ||r_g + C @ B||_2^2 given r_norm2 = ||r_g||^2, M = B @ B.T (R, R),
    b = B @ r_g (R,), and candidate coefficients C (K, R)."""
    return r_norm2 + 2.0 * (C @ b) + np.einsum("kr,rs,ks->k", C, M, C)


def quad_mse(r_g, B, C):
    """Exact MSE ||r_g + C @ B||_2^2 for residual r_g (D,), basis B (R, D), and C (K, R)."""
    M = B @ B.T
    b = B @ r_g
    return quad_mse_from_gram(float(r_g @ r_g), M, b, C)


def gram_svd_analysis(G, h, r_norm2, ranks):
    """Given Gram matrix G = dU @ dU.T (K, K), linear term h = dU @ r_g (K,), and ||r_g||^2,
    return singular spectrum stats and exact oracle rank-R MSE losses without forming V."""
    eigvals, Q = np.linalg.eigh(0.5 * (G + G.T))
    order = np.argsort(eigvals)[::-1]
    lam = np.maximum(eigvals[order], 0.0)
    Q = Q[:, order]
    tot = float(lam.sum())
    if tot <= 0.0:
        zeros = np.full(len(h), r_norm2, dtype=np.float64)
        return {
            "total_var": 0.0,
            "eff_rank_var": 0.0,
            "eff_rank_sv": 0.0,
            "var_explained": {int(r): 1.0 for r in ranks},
            "oracle_losses": {int(r): zeros.copy() for r in ranks},
        }
    p_var = lam / tot
    p_var_nz = p_var[p_var > 1e-15]
    eff_rank_var = float(np.exp(-np.sum(p_var_nz * np.log(p_var_nz))))
    sv = np.sqrt(lam)
    p_sv = sv / float(sv.sum())
    p_sv_nz = p_sv[p_sv > 1e-15]
    eff_rank_sv = float(np.exp(-np.sum(p_sv_nz * np.log(p_sv_nz))))

    cum = np.cumsum(lam) / tot
    K = len(h)
    qth = Q.T @ h  # (K,)
    var_explained, oracle_losses = {}, {}
    for r in ranks:
        r_eff = min(int(r), K)
        var_explained[int(r)] = float(cum[r_eff - 1]) if r_eff > 0 else 0.0
        nz = lam[:r_eff] > 1e-12 * tot
        Qr = Q[:, :r_eff][:, nz]
        lam_r = lam[:r_eff][nz]
        qth_r = qth[:r_eff][nz]
        if Qr.shape[1] == 0:
            oracle_losses[int(r)] = np.full(K, r_norm2, dtype=np.float64)
        else:
            lin = Qr @ qth_r
            quad = (Qr * Qr) @ lam_r
            oracle_losses[int(r)] = r_norm2 + 2.0 * lin + quad
    return {
        "total_var": tot,
        "eff_rank_var": eff_rank_var,
        "eff_rank_sv": eff_rank_sv,
        "var_explained": var_explained,
        "oracle_losses": oracle_losses,
    }


def dual_ridge_coeffs(dA_train, dA_eval, lam=1e-2):
    """Dual kernel-ridge weights C (K_eval, M) such that dU_hat = C @ dU_train minimizes
    ||dA_train @ J - dU_train||_F^2 + lam ||J||_F^2. Never forms (d_a, D_tot)."""
    K_mat = dA_train @ dA_train.T
    m = K_mat.shape[0]
    reg = lam * max(float(np.trace(K_mat)) / max(m, 1), 1e-8)
    inv_term = np.linalg.solve(K_mat + reg * np.eye(m), np.eye(m))
    return (dA_eval @ dA_train.T) @ inv_term


def _ranks_desc_tie_avg(x):
    """Average-tie ranks (0 = smallest). Used for Spearman when scores can tie (e.g. copy_ref)."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    xs = x[order]
    i, n = 0, len(xs)
    while i < n:
        j = i + 1
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks


def ranking_metrics(true_loss, pred_loss, topk, recall_ms):
    """Spearman correlation, recall@m of true top-k inside pred top-m, elite overlap (recall@k),
    and best-candidate agreement."""
    t = np.asarray(true_loss, dtype=np.float64)
    p = np.asarray(pred_loss, dtype=np.float64)
    true_order = np.argsort(t, kind="mergesort")
    pred_order = np.argsort(p, kind="mergesort")
    true_top = set(true_order[:topk].tolist())

    rt = _ranks_desc_tie_avg(t)
    rp = _ranks_desc_tie_avg(p)
    st, sp = rt.std(), rp.std()
    spearman = 0.0 if (st == 0.0 or sp == 0.0) else float(np.corrcoef(rt, rp)[0, 1])

    recall = {}
    for m in recall_ms:
        m_int = min(int(m), len(p))
        recall[int(m)] = len(true_top & set(pred_order[:m_int].tolist())) / topk
    return {
        "spearman": spearman,
        "elite_overlap": recall.get(topk, len(true_top & set(pred_order[:topk].tolist())) / topk),
        "best_same": bool(pred_order[0] == true_order[0]),
        "recall": recall,
    }
