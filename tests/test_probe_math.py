import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import probe_math as pm


class TestProbeMath(unittest.TestCase):
    def test_pack_weighted_latent_matches_dino_objective(self):
        rng = np.random.default_rng(0)
        K, P, Dv, Dp = 12, 16, 8, 4
        vis = rng.standard_normal((K, P, Dv))
        prop = rng.standard_normal((K, Dp))
        vis_g = rng.standard_normal((1, P, Dv))
        prop_g = rng.standard_normal((1, Dp))
        for alpha in (0.5, 1.0, 2.5):
            U = pm.pack_weighted_latent(vis, prop, alpha=alpha)
            ug = pm.pack_weighted_latent(vis_g, prop_g, alpha=alpha)[0]
            got = np.sum((U - ug[None]) ** 2, axis=1)
            want = np.mean((vis - vis_g) ** 2, axis=(1, 2)) + alpha * np.mean((prop - prop_g) ** 2, axis=1)
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)

    def test_quad_mse_matches_dense_mse(self):
        rng = np.random.default_rng(1)
        K, R, D = 40, 12, 500
        r_g = rng.standard_normal(D)
        B = rng.standard_normal((R, D))
        C = rng.standard_normal((K, R))
        got = pm.quad_mse(r_g, B, C)
        dense = np.sum((r_g[None, :] + C @ B) ** 2, axis=1)
        np.testing.assert_allclose(got, dense, rtol=1e-11, atol=1e-11)

    def test_gram_svd_oracle_matches_explicit_svd(self):
        rng = np.random.default_rng(2)
        K, D = 25, 120
        u_ref = rng.standard_normal(D)
        ug = rng.standard_normal(D)
        r_g = u_ref - ug
        # Construct a rank-10 perturbation matrix + small noise
        dU = rng.standard_normal((K, 10)) @ rng.standard_normal((10, D)) + 0.05 * rng.standard_normal((K, D))
        dU[0] = 0.0  # action[0] == mu
        G = dU @ dU.T
        h = dU @ r_g
        ranks = [2, 5, 10, 25]
        res = pm.gram_svd_analysis(G, h, float(r_g @ r_g), ranks)

        # Explicit SVD comparison
        U_svd, S_svd, Vt_svd = np.linalg.svd(dU, full_matrices=False)
        for r in ranks:
            dU_r = (U_svd[:, :r] * S_svd[:r]) @ Vt_svd[:r]
            want = np.sum((r_g[None, :] + dU_r) ** 2, axis=1)
            np.testing.assert_allclose(res["oracle_losses"][r], want, rtol=1e-10, atol=1e-10)
        # Rank-K oracle reproduces exact true MSE
        true_mse = np.sum((r_g[None, :] + dU) ** 2, axis=1)
        np.testing.assert_allclose(res["oracle_losses"][K], true_mse, rtol=1e-10, atol=1e-10)
        self.assertGreater(res["var_explained"][10], 0.95)
        self.assertLess(res["eff_rank_var"], 12.0)

    def test_dual_ridge_recovers_linear_dynamics_and_scores_quadratically(self):
        rng = np.random.default_rng(3)
        d_a, D, M, K = 10, 80, 40, 50
        J_true = rng.standard_normal((d_a, D))
        r_g = rng.standard_normal(D)
        dA_tr = rng.standard_normal((M, d_a))
        dU_tr = dA_tr @ J_true
        dA_ev = rng.standard_normal((K, d_a))
        dU_ev = dA_ev @ J_true

        C = pm.dual_ridge_coeffs(dA_tr, dA_ev, lam=1e-9)
        pred_loss = pm.quad_mse(r_g, dU_tr, C)
        true_loss = np.sum((r_g[None, :] + dU_ev) ** 2, axis=1)
        np.testing.assert_allclose(pred_loss, true_loss, rtol=1e-6, atol=1e-6)

    def test_ranking_metrics_perfect_and_constant(self):
        t = np.arange(100, dtype=np.float64)
        m_perf = pm.ranking_metrics(t, t * 2.0 + 3.0, topk=10, recall_ms=[10, 20])
        self.assertAlmostEqual(m_perf["spearman"], 1.0, places=6)
        self.assertEqual(m_perf["elite_overlap"], 1.0)
        self.assertTrue(m_perf["best_same"])
        self.assertEqual(m_perf["recall"][20], 1.0)

        # Constant prediction (copy_ref control): spearman = 0, deterministic tie-handling
        m_const = pm.ranking_metrics(t, np.ones(100), topk=10, recall_ms=[10, 20])
        self.assertEqual(m_const["spearman"], 0.0)


if __name__ == "__main__":
    unittest.main()
