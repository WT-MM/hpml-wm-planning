import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import mpc_utils as mu


class TestMpcUtils(unittest.TestCase):
    def test_episode_split_is_disjoint_and_complete(self):
        sp = mu.episode_split(n_train=18685, n_val=21, heldout_n=500)
        self.assertEqual(len(sp["train_G"]), 18185)
        self.assertEqual(len(sp["heldout_train"]), 500)
        self.assertEqual(len(sp["val"]), 21)
        self.assertEqual(set(sp["train_G"]) & set(sp["heldout_train"]), set())
        self.assertEqual(sp["train_G"][-1] + 1, sp["heldout_train"][0])

    def test_sample_dset_tasks_no_leakage_and_deterministic(self):
        seq_lens = {"train": [100] * 1000, "valid": [100] * 21}
        t1 = mu.sample_dset_tasks(seq_lens, "val+heldout", n_tasks=50, traj_len=51, seed=99, heldout_n=200)
        t2 = mu.sample_dset_tasks(seq_lens, "val+heldout", n_tasks=50, traj_len=51, seed=99, heldout_n=200)
        self.assertEqual(t1, t2)
        for t in t1:
            if t["split"] == "train":
                self.assertGreaterEqual(t["episode"], 800)  # never touches train_G [0, 800)
            self.assertLessEqual(t["offset"] + 51, 100)

        t_tr = mu.sample_dset_tasks(seq_lens, "train_G", n_tasks=30, traj_len=51, seed=99, heldout_n=200)
        for t in t_tr:
            self.assertEqual(t["split"], "train")
            self.assertLess(t["episode"], 800)

    def test_planner_spec_and_shrink_budget(self):
        p_full = mu.parse_planner_spec("cem:fp16_reuse")
        self.assertEqual(mu.iter_budget(p_full, 0, 30, 300, 30), (300, 30))

        p_few = mu.parse_planner_spec("few150:fp16_reuse")
        self.assertEqual(mu.iter_budget(p_few, 10, 30, 300, 30), (150, 15))

        p_sh = mu.parse_planner_spec("shrink300_60:fp16_reuse")
        ns = [mu.iter_budget(p_sh, it, 30, 300, 30)[0] for it in range(30)]
        self.assertEqual(ns[0], 300)
        self.assertEqual(ns[-1], 60)
        self.assertEqual(sum(ns), 30 * 180)  # linear from 300 to 60 -> avg 180

        p_gd = mu.parse_planner_spec("gd:cache_sdpa_fp16")
        self.assertEqual(p_gd["kind"], "gd")
        with self.assertRaises(ValueError):
            mu.parse_planner_spec("bad_spec")

    def test_aggregate_mpc_results(self):
        runs = {
            "cem:fp16_reuse": [
                {"episode_plan_s": [10.0, 10.0], "step_plan_s": [2.0] * 10, "success": [True, False],
                 "final_state_dist": [10.0, 40.0], "peak_mem_alloc_mb": 9000.0, "total_candidates": 9000},
                {"episode_plan_s": [10.0, 10.0], "step_plan_s": [2.0] * 10, "success": [True, True],
                 "final_state_dist": [12.0, 14.0], "peak_mem_alloc_mb": 9100.0, "total_candidates": 9000},
            ],
            "few150:fp16_reuse": [
                {"episode_plan_s": [5.0, 5.0], "step_plan_s": [1.0] * 10, "success": [True, False],
                 "final_state_dist": [11.0, 42.0], "peak_mem_alloc_mb": 5000.0, "total_candidates": 4500},
                {"episode_plan_s": [5.0, 5.0], "step_plan_s": [1.0] * 10, "success": [False, True],
                 "final_state_dist": [35.0, 15.0], "peak_mem_alloc_mb": 5000.0, "total_candidates": 4500},
            ],
        }
        agg = mu.aggregate_mpc_results(runs)
        self.assertEqual(agg["cem:fp16_reuse"]["n_tasks_total"], 4)
        self.assertAlmostEqual(agg["few150:fp16_reuse"]["speedup_vs_base"], 2.0, places=6)
        self.assertAlmostEqual(agg["cem:fp16_reuse"]["success_rate"], 0.75, places=6)
        self.assertAlmostEqual(agg["few150:fp16_reuse"]["success_rate"], 0.50, places=6)
        self.assertEqual(agg["cem:fp16_reuse"]["peak_mem_alloc_mb"], 9100.0)


if __name__ == "__main__":
    unittest.main()
