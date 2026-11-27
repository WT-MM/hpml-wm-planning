"""Reference-conditioned low-rank basis predictor (RefBasisG), equal-cost unconditioned
baseline (NoRefBasisG), GPU-native OnlineRidgeDraft, and HybridDraft (G + online residual).

Every scorer operates in packed weighted latent space U of dim D_tot = P*D_v + D_p where
||U(a) - U_g||_2^2 == MSE(vis, vis_g) + alpha * MSE(prop, prop_g) identically.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def pack_latent(vis, prop, alpha=1.0):
    """Pack visual (..., P, D_v) and proprio (..., D_p) into (..., D_tot) float32."""
    v = vis.float().flatten(-2)
    p = prop.float()
    return torch.cat([v / math.sqrt(v.shape[-1]), p * math.sqrt(alpha / p.shape[-1])], dim=-1)


def quad_mse_from_gram(r_norm2, M, b, C):
    """Exact MSE ||r_g + C @ B||_2^2 given r_norm2 (scalar), M = B @ B.T (R, R),
    b = B @ r_g (R,), and C (K, R)."""
    return r_norm2 + 2.0 * (C @ b) + ((C @ M) * C).sum(dim=-1)


def lowrank_mse_loss(B, C, dU, normalize_scale=True, eps=1e-4):
    """Exact batched low-rank MSE loss without forming (B_ref, K, D_tot).
    B: (B_ref, R, D), C: (B_ref, K, R), dU: (B_ref, K, D).
    Returns scalar mean loss and per-candidate squared error (B_ref, K)."""
    B32, C32, dU32 = B.float(), C.float(), dU.float()
    M = torch.bmm(B32, B32.transpose(1, 2))               # (B_ref, R, R)
    P = torch.bmm(dU32, B32.transpose(1, 2))              # (B_ref, K, R)
    pred_n2 = (torch.bmm(C32, M) * C32).sum(dim=-1)       # (B_ref, K)
    cross = (C32 * P).sum(dim=-1)                         # (B_ref, K)
    tgt_n2 = (dU32 * dU32).sum(dim=-1)                    # (B_ref, K)
    se = (pred_n2 - 2.0 * cross + tgt_n2).clamp_min(0.0)
    if normalize_scale:
        scale = tgt_n2.mean(dim=-1, keepdim=True) + eps
        return (se / scale).mean(), se
    return se.mean(), se


def pairwise_rank_loss(pred_score, true_score, topk=8, margin=0.0):
    """Planning-aware pairwise logistic ranking loss focusing on separating top-k elites from non-elites.
    pred_score, true_score: (B_ref, K) lower-is-better losses."""
    B_ref, K = true_score.shape
    order = torch.argsort(true_score, dim=-1)
    k = min(topk, K // 2)
    elites = order[:, :k]
    non_elites = order[:, k:]
    p_e = torch.gather(pred_score, 1, elites).unsqueeze(2)      # (B, k, 1)
    p_n = torch.gather(pred_score, 1, non_elites).unsqueeze(1)  # (B, 1, K-k)
    t_e = torch.gather(true_score, 1, elites).unsqueeze(2)
    t_n = torch.gather(true_score, 1, non_elites).unsqueeze(1)
    # Scale-invariant margin normalized by true loss spread per reference
    spread = (true_score.max(dim=-1, keepdim=True).values - true_score.min(dim=-1, keepdim=True).values).clamp_min(1e-4)
    diff = (p_e - p_n) / spread.unsqueeze(2)
    # Only penalize pairs where true_n > true_e
    valid = (t_n > t_e).float()
    loss = F.softplus(5.0 * (diff + margin)) * valid
    return loss.sum() / valid.sum().clamp_min(1.0)


class _ResConvBlock(nn.Module):
    def __init__(self, ch, cond_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.film = nn.Linear(cond_dim, 2 * ch)

    def forward(self, x, cond):
        scale, shift = self.film(cond).chunk(2, dim=-1)
        h = self.conv1(F.gelu(self.norm1(x)))
        h = self.norm2(h) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        return x + self.conv2(F.gelu(h))


class _BasisEncoder(nn.Module):
    """Spatial 14x14 encoder run ONCE per CEM iteration (B=1) to emit packed basis B (B, R, D_tot)
    and compact scene context ctx (B, d_ctx)."""
    def __init__(self, in_vis_ch, cond_dim, d_v=384, d_p=10, grid=14, rank=32, hidden=256, n_blocks=3, alpha=1.0):
        super().__init__()
        self.d_v, self.d_p, self.grid, self.rank, self.alpha = d_v, d_p, grid, rank, alpha
        self.P = grid * grid
        self.in_proj = nn.Conv2d(in_vis_ch, hidden, 1)
        self.blocks = nn.ModuleList([_ResConvBlock(hidden, cond_dim) for _ in range(n_blocks)])
        self.norm_out = nn.GroupNorm(8, hidden)
        self.vis_basis = nn.Conv2d(hidden, rank * d_v, 1)
        self.ctx_proj = nn.Sequential(
            nn.Linear(2 * hidden + cond_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.prop_basis = nn.Linear(hidden, rank * d_p)

    def forward(self, vis_in, cond):
        # vis_in: (B, in_vis_ch, grid, grid), cond: (B, cond_dim)
        B = vis_in.shape[0]
        h = self.in_proj(vis_in)
        for blk in self.blocks:
            h = blk(h, cond)
        h = F.gelu(self.norm_out(h))
        # Visual basis: (B, R*d_v, grid, grid) -> (B, R, P*d_v)
        bv = self.vis_basis(h).view(B, self.rank, self.d_v, self.P).transpose(2, 3).reshape(B, self.rank, self.P * self.d_v)
        # Global context from avg + max pool + conditioning vector
        ctx_in = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3)), cond], dim=-1)
        ctx = self.ctx_proj(ctx_in)
        bp = self.prop_basis(ctx).view(B, self.rank, self.d_p)
        # Apply exact DINO-WM objective weights so ||C @ basis||^2 matches weighted MSE
        basis = torch.cat([bv / math.sqrt(self.P * self.d_v), bp * math.sqrt(self.alpha / self.d_p)], dim=-1)
        return basis, ctx


class RefBasisG(nn.Module):
    """Reference-conditioned low-rank basis predictor G(z_0, z_ref, mu, dA) -> dU_hat = C @ B.
    Exact zero residual at dA = 0 by construction (c_lin(0) = 0 and c_nl(0) - c_nl(0) = 0)."""
    def __init__(self, d_v=384, d_p=10, d_a=50, grid=14, rank=32, hidden=256, n_blocks=3, alpha=1.0, linear_only=False):
        super().__init__()
        self.d_v, self.d_p, self.d_a, self.grid, self.rank, self.alpha = d_v, d_p, d_a, grid, rank, alpha
        self.linear_only = linear_only
        # Visual channels: [z0_vis, zref_vis, zref_vis - z0_vis]
        # Cond vector: [z0_prop, zref_prop, zref_prop - z0_prop, mu]
        self.encoder = _BasisEncoder(
            in_vis_ch=3 * d_v, cond_dim=3 * d_p + d_a,
            d_v=d_v, d_p=d_p, grid=grid, rank=rank, hidden=hidden, n_blocks=n_blocks, alpha=alpha,
        )
        # Amortized local Jacobian map W_lin(ctx): (B, R, d_a)
        self.w_lin = nn.Linear(hidden, rank * d_a)
        if not linear_only:
            self.nl_mlp = nn.Sequential(
                nn.Linear(hidden + d_a, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, rank),
            )
            # Small init on nonlinear branch so training starts near smooth linear regime
            nn.init.zeros_(self.nl_mlp[-1].weight)
            nn.init.zeros_(self.nl_mlp[-1].bias)

    def encode_reference(self, z0_vis, z0_prop, zref_vis, zref_prop, mu):
        """Run ONCE per CEM iteration (B_ref). Returns (basis (B, R, D_tot), ctx (B, hidden), W_lin (B, R, d_a))."""
        B, g = z0_vis.shape[0], self.grid
        v0 = z0_vis.float().transpose(1, 2).reshape(B, self.d_v, g, g)
        vr = zref_vis.float().transpose(1, 2).reshape(B, self.d_v, g, g)
        vis_in = torch.cat([v0, vr, vr - v0], dim=1)
        p0, pr = z0_prop.float(), zref_prop.float()
        cond = torch.cat([p0, pr, pr - p0, mu.float()], dim=-1)
        basis, ctx = self.encoder(vis_in, cond)
        W = self.w_lin(ctx).view(B, self.rank, self.d_a)
        return basis, ctx, W

    def candidate_coeffs(self, ctx, W, dA):
        """Run for all K candidates. ctx: (B, hidden), W: (B, R, d_a), dA: (B, K, d_a) -> C: (B, K, R)."""
        dA = dA.float()
        c_lin = torch.bmm(dA, W.transpose(1, 2))  # (B, K, R)
        if self.linear_only:
            return c_lin
        B, K, _ = dA.shape
        ctx_exp = ctx.unsqueeze(1).expand(B, K, -1)
        nl = self.nl_mlp(torch.cat([ctx_exp, dA], dim=-1))
        nl_zero = self.nl_mlp(torch.cat([ctx, torch.zeros(B, self.d_a, device=ctx.device, dtype=ctx.dtype)], dim=-1))
        return c_lin + (nl - nl_zero.unsqueeze(1))

    def score_candidates(self, z0_vis, z0_prop, zref_vis, zref_prop, mu, ug, actions):
        """Single-reference inference (B_ref=1) scoring K candidates: returns (losses (K,), basis (R, D), C (K, R))."""
        basis, ctx, W = self.encode_reference(z0_vis, z0_prop, zref_vis, zref_prop, mu)
        B0 = basis[0]                                           # (R, D_tot)
        u_ref = pack_latent(zref_vis[0], zref_prop[0], self.alpha)
        r_g = u_ref - ug
        M = B0 @ B0.T
        b = B0 @ r_g
        r_n2 = (r_g * r_g).sum()
        dA = (actions - mu).unsqueeze(0)                        # (1, K, d_a)
        C = self.candidate_coeffs(ctx, W, dA)[0]                # (K, R)
        return quad_mse_from_gram(r_n2, M, b, C), B0, C


class NoRefBasisG(nn.Module):
    """Equal-cost unconditioned baseline: identical architecture, depth, width, and rank R as RefBasisG,
    but conditioned ONLY on start state z_0 (NO reference rollout z_ref, NO mu).
    Predicts u(a_k) = u_0 + C(z_0, a_k) @ B(z_0)."""
    def __init__(self, d_v=384, d_p=10, d_a=50, grid=14, rank=32, hidden=256, n_blocks=3, alpha=1.0):
        super().__init__()
        self.d_v, self.d_p, self.d_a, self.grid, self.rank, self.alpha = d_v, d_p, d_a, grid, rank, alpha
        self.encoder = _BasisEncoder(
            in_vis_ch=d_v, cond_dim=d_p,
            d_v=d_v, d_p=d_p, grid=grid, rank=rank, hidden=hidden, n_blocks=n_blocks, alpha=alpha,
        )
        self.w_lin = nn.Linear(hidden, rank * d_a)
        self.nl_mlp = nn.Sequential(
            nn.Linear(hidden + d_a, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, rank),
        )

    def encode_start(self, z0_vis, z0_prop):
        B, g = z0_vis.shape[0], self.grid
        v0 = z0_vis.float().transpose(1, 2).reshape(B, self.d_v, g, g)
        basis, ctx = self.encoder(v0, z0_prop.float())
        W = self.w_lin(ctx).view(B, self.rank, self.d_a)
        return basis, ctx, W

    def candidate_coeffs(self, ctx, W, actions):
        a = actions.float()
        B, K, _ = a.shape
        c_lin = torch.bmm(a, W.transpose(1, 2))
        ctx_exp = ctx.unsqueeze(1).expand(B, K, -1)
        return c_lin + self.nl_mlp(torch.cat([ctx_exp, a], dim=-1))

    def score_candidates(self, z0_vis, z0_prop, ug, actions):
        """Single-start inference (B_ref=1) scoring K candidates without any reference rollout."""
        basis, ctx, W = self.encode_start(z0_vis, z0_prop)
        B0 = basis[0]
        u0 = pack_latent(z0_vis[0], z0_prop[0], self.alpha)
        r_g = u0 - ug
        M = B0 @ B0.T
        b = B0 @ r_g
        r_n2 = (r_g * r_g).sum()
        C = self.candidate_coeffs(ctx, W, actions.unsqueeze(0))[0]
        return quad_mse_from_gram(r_n2, M, b, C), B0, C


class OnlineRidgeBuffer:
    """Zero-extra-F-call GPU-native dual kernel-ridge draft (and hybrid residual adapter).
    Maintains a sliding window of verified exact (action, packed_U) pairs from recent CEM iterations."""
    def __init__(self, max_iters=2, lam=1e-2):
        self.max_iters = max_iters
        self.lam = lam
        self.hist_a = []
        self.hist_u = []

    def reset(self):
        self.hist_a.clear()
        self.hist_u.clear()

    def update(self, actions_flat, U_verified):
        self.hist_a.append(actions_flat.detach().float())
        self.hist_u.append(U_verified.detach().float())
        if len(self.hist_a) > self.max_iters:
            self.hist_a.pop(0)
            self.hist_u.pop(0)

    def __len__(self):
        return sum(x.shape[0] for x in self.hist_a)

    def _dual_weights(self, A_tr, mu, A_ev):
        dA_tr = A_tr - mu[None]
        dA_ev = A_ev - mu[None]
        K_mat = dA_tr @ dA_tr.T
        m = K_mat.shape[0]
        reg = self.lam * (torch.trace(K_mat) / max(m, 1)).clamp_min(1e-8)
        eye = torch.eye(m, device=K_mat.device, dtype=K_mat.dtype)
        # Solve (K + reg I) X = (dA_tr @ dA_ev.T) -> W_ev = X.T
        X = torch.linalg.solve(K_mat + reg * eye, dA_tr @ dA_ev.T)
        return X.T  # (K_ev, M)

    def score_online(self, mu, u_ref, ug, actions_flat):
        """Pure training-free GPU online-ridge scorer (<0.5 ms on RTX 5090)."""
        A_tr = torch.cat(self.hist_a, dim=0)
        U_tr = torch.cat(self.hist_u, dim=0)
        W_ev = self._dual_weights(A_tr, mu.float(), actions_flat.float())
        Y_tr = U_tr - u_ref[None]
        r_g = u_ref - ug
        M = Y_tr @ Y_tr.T
        b = Y_tr @ r_g
        r_n2 = (r_g * r_g).sum()
        return quad_mse_from_gram(r_n2, M, b, W_ev)

    def score_hybrid(self, model_g, z0_vis, z0_prop, zref_vis, zref_prop, mu, ug, actions_flat):
        """Hybrid scorer: RefBasisG prior + online verified residual via stacked basis [B_g; Y_tr]."""
        basis, ctx, W_lin = model_g.encode_reference(z0_vis, z0_prop, zref_vis, zref_prop, mu[None])
        B_g = basis[0]
        u_ref = pack_latent(zref_vis[0], zref_prop[0], model_g.alpha)
        r_g = u_ref - ug
        r_n2 = (r_g * r_g).sum()
        dA_ev = (actions_flat.float() - mu.float()[None]).unsqueeze(0)
        C_ev = model_g.candidate_coeffs(ctx, W_lin, dA_ev)[0]

        if not self.hist_a:
            return quad_mse_from_gram(r_n2, B_g @ B_g.T, B_g @ r_g, C_ev)

        A_tr = torch.cat(self.hist_a, dim=0)
        U_tr = torch.cat(self.hist_u, dim=0)
        dA_tr = (A_tr - mu.float()[None]).unsqueeze(0)
        C_tr = model_g.candidate_coeffs(ctx, W_lin, dA_tr)[0]
        W_ev = self._dual_weights(A_tr, mu.float(), actions_flat.float())
        Y_tr = U_tr - u_ref[None]

        B_cat = torch.cat([B_g, Y_tr], dim=0)
        C_cat = torch.cat([C_ev - W_ev @ C_tr, W_ev], dim=1)
        return quad_mse_from_gram(r_n2, B_cat @ B_cat.T, B_cat @ r_g, C_cat)
