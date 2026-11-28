import random
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import specplan_math as sm


class TestSpecplanMath(unittest.TestCase):
    def test_lowrank_mse_matches_dense(self):
        rng = np.random.default_rng(0)
        K, R, D = 32, 16, 400
        B = rng.standard_normal((R, D))
        C = rng.standard_normal((K, R))
        dU = rng.standard_normal((K, D))
        got = sm.lowrank_mse_loss_np(B, C, dU)
        want = np.sum((C @ B - dU) ** 2, axis=1)
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)

    def test_hybrid_quad_mse_matches_dense_and_boundaries(self):
        rng = np.random.default_rng(1)
        K, R, M, D = 24, 8, 12, 300
        r_g = rng.standard_normal(D)
        B_g = rng.standard_normal((R, D))
        C_ev = rng.standard_normal((K, R))
        C_tr = rng.standard_normal((M, R))
        Y_tr = rng.standard_normal((M, D))
        W_ev = rng.standard_normal((K, M))

        got = sm.hybrid_quad_mse_np(r_g, B_g, C_ev, C_tr, Y_tr, W_ev)
        # Explicit dense hybrid: C_ev @ B_g + W_ev @ (Y_tr - C_tr @ B_g)
        E_tr = Y_tr - C_tr @ B_g
        dU_hyb = C_ev @ B_g + W_ev @ E_tr
        want = np.sum((r_g[None, :] + dU_hyb) ** 2, axis=1)
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)

        # Boundary 1: if G is exact on history (Y_tr == C_tr @ B_g), hybrid reduces identically to G
        got_exact_g = sm.hybrid_quad_mse_np(r_g, B_g, C_ev, C_tr, C_tr @ B_g, W_ev)
        want_g = np.sum((r_g[None, :] + C_ev @ B_g) ** 2, axis=1)
        np.testing.assert_allclose(got_exact_g, want_g, rtol=1e-10, atol=1e-10)

        # Boundary 2: at dA=0 (C_ev=0, W_ev=0), hybrid returns exact ||r_g||^2
        got_zero = sm.hybrid_quad_mse_np(r_g, B_g, np.zeros((1, R)), C_tr, Y_tr, np.zeros((1, M)))
        np.testing.assert_allclose(got_zero, [float(r_g @ r_g)], rtol=1e-12, atol=1e-12)

    def test_stratified_candidates_exclude_ref_and_cover_top(self):
        losses = np.linspace(10.0, 0.0, 300)  # best candidates at high indices; idx 0 is mu
        idx = sm.stratified_candidate_indices(losses, k_save=32, rng=random.Random(0))
        self.assertEqual(len(idx), 32)
        self.assertNotIn(0, idx)
        self.assertEqual(len(set(idx.tolist())), 32)
        # Top 16 elites (indices 299..284) must all be included
        self.assertTrue(set(range(284, 300)).issubset(set(idx.tolist())))

    def test_train_g_leakage_guard(self):
        sm.assert_train_g_only([0, 100, 18184], n_train_total=18685, heldout_n=500)
        with self.assertRaises(ValueError):
            sm.assert_train_g_only([0, 18185], n_train_total=18685, heldout_n=500)
        with self.assertRaises(ValueError):
            sm.assert_train_g_only([-1], n_train_total=18685, heldout_n=500)


if __name__ == "__main__":
    unittest.main()
