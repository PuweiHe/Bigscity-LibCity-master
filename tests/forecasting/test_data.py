import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from traffic_forecasting.data import prepare


class DataTests(unittest.TestCase):
    def fixture(self, root):
        folders = [root / "施工前/data_11", root / "5.28施工期/data_5_28"]
        for folder in folders:
            folder.mkdir(parents=True)
        frames = {}
        for i in range(13):
            path = folders[0 if i < 7 else 1] / f"{i:02}.xlsx"
            path.write_bytes(f"fixture-{i}".encode())
            time = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
            times = [
                time + pd.Timedelta(minutes=m, seconds=s)
                for m in range(12)
                for s in [1, 30]
            ]
            frames[str(path)] = pd.DataFrame(
                {
                    "时间": times,
                    "行驶方向": "N-S",
                    "车道": 1,
                    "位置（离设备距离）": 3,
                    "速度 (km/h)": -80.0,
                    "车型": 8,
                    "车头时距(s)": 2,
                    "时间间距(s)": 2,
                    "轴数": 2,
                    "轴组数": 2,
                    "轴距(车长)": 4,
                }
            )
        return frames

    def test_session_split_and_partial_minute_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = self.fixture(root)
            with patch(
                "traffic_forecasting.data.pd.read_excel",
                side_effect=lambda p: frames[str(p)].copy(),
            ):
                data, audit = prepare(root)
            self.assertEqual(
                audit["split_counts"], {"train": 54, "validation": 12, "test": 12}
            )
            for a, b in [("train", "validation"), ("validation", "test")]:
                self.assertLess(
                    max(data["times"][data["split"] == a]),
                    min(data["times"][data["split"] == b]),
                )
                self.assertFalse(
                    set(data["groups"][data["split"] == a])
                    & set(data["groups"][data["split"] == b])
                )
            np.testing.assert_array_equal(data["X"][:, :, 0], 80)
            np.testing.assert_array_equal(data["X"][:, :, 1], 2)

    def test_duplicates_outliers_and_outages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = self.fixture(root)
            key = next(iter(frames))
            frames[key] = pd.concat(
                [frames[key], frames[key].iloc[[0]]], ignore_index=True
            )
            frames[key].loc[4, "速度 (km/h)"] = -999
            with patch(
                "traffic_forecasting.data.pd.read_excel",
                side_effect=lambda p: frames[str(p)].copy(),
            ):
                _, audit = prepare(root)
            self.assertEqual(audit["files"][0]["duplicates_removed"], 1)
            self.assertEqual(audit["files"][0]["speed_outliers_excluded"], 1)
            frames[key] = frames[key].drop(index=[6, 7])
            with patch(
                "traffic_forecasting.data.pd.read_excel",
                side_effect=lambda p: frames[str(p)].copy(),
            ):
                with self.assertRaisesRegex(ValueError, "Missing observation interval"):
                    prepare(root)
