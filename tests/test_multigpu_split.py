import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import multigpu_bench as mg


class TestMultiGpuSplit(unittest.TestCase):
    def test_split_bounds_even_and_uneven(self):
        self.assertEqual(mg.split_bounds(300, 2), [(0, 150), (150, 300)])
        self.assertEqual(mg.split_bounds(301, 2), [(0, 151), (151, 301)])
        self.assertEqual(mg.split_bounds(10, 3), [(0, 4), (4, 7), (7, 10)])
        with self.assertRaises(ValueError):
            mg.split_bounds(0, 2)


if __name__ == "__main__":
    unittest.main()
