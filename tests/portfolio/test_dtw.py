import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from traffic_analysis.__main__ import benchmark, load_series
from traffic_analysis.dtw import exact_dtw, pairwise_dtw


class DTWTests(unittest.TestCase):
    def test_known_distances(self):
        self.assertEqual(exact_dtw([1, 2, 3], [1, 2, 3]), 0)
        self.assertEqual(exact_dtw([0], [2, 3]), 5)
        self.assertEqual(exact_dtw([1, 2], [1, 1, 2]), 0)

    def test_independent_full_table_reference(self):
        rng = np.random.default_rng(11)
        for n, m in [(1, 4), (3, 7), (6, 6)]:
            a, b = rng.normal(size=n), rng.normal(size=m)
            table = np.full((n + 1, m + 1), np.inf)
            table[0, 0] = 0
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    table[i, j] = abs(a[i - 1] - b[j - 1]) + min(
                        table[i - 1, j], table[i, j - 1], table[i - 1, j - 1]
                    )
            self.assertAlmostEqual(exact_dtw(a, b), table[-1, -1])

    def test_pair_calls_and_equivalence(self):
        values = [[i, i + 1, i + 2] for i in range(6)]
        with patch("traffic_analysis.dtw.exact_dtw", wraps=exact_dtw) as distance:
            result = pairwise_dtw(values)
            self.assertEqual(distance.call_count, 15)
        expected = [[exact_dtw(a, b) for b in values] for a in values]
        np.testing.assert_allclose(result, expected)

    def test_invalid_series(self):
        for values in [[], [np.nan], [np.inf], [[1, 2]]]:
            with self.assertRaises(ValueError):
                exact_dtw(values, [1])
        with self.assertRaises(ValueError):
            pairwise_dtw([])
        np.testing.assert_array_equal(pairwise_dtw([[3]]), [[0]])

    def test_benchmark(self):
        _, report = benchmark([[0, 1], [2, 1], [4, 2]], 2)
        self.assertEqual(report["max_absolute_error"], 0)
        self.assertEqual(report["baseline_calls"], 6)
        self.assertEqual(report["optimized_calls"], 3)

    def test_csv_validation_and_sorting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            frame = pd.DataFrame(
                {
                    "t": ["2024-01-01 00:01", "2024-01-01 00:00"],
                    "s": ["a", "a"],
                    "v": [2, 1],
                }
            )
            frame.to_csv(path, index=False)
            names, values, _ = load_series(path, "t", "s", "v")
            self.assertEqual(names, ["a"])
            np.testing.assert_array_equal(values[0], [1, 2])
            for invalid in [
                pd.concat([frame, frame]),
                frame.assign(v=[np.inf, 1]),
                frame.assign(t=["2024-01-01 00:03", "2024-01-01 00:00"]),
                frame.assign(s=[None, "a"]),
                frame.iloc[:0],
            ]:
                invalid.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    load_series(path, "t", "s", "v")


if __name__ == "__main__":
    unittest.main()
