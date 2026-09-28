"""Render the published aggregate results without exposing vehicle-level data."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
report = json.loads((ROOT / "docs/forecasting/test_report.json").read_text())["models"]
keys = ["persistence", "moving_average", "rf_0", "rf_mean_4", "gru_0", "gru_mean_1"]
labels = [
    "Persistence",
    "4-minute mean",
    "RF baseline",
    "RF tuned",
    "GRU baseline",
    "GRU tuned",
]
values = [report[k]["mae"] for k in keys]
errors = [report[k].get("mae_seed_std", 0) for k in keys]
fig, ax = plt.subplots(figsize=(9, 5.2))
fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.22)
colors = ["#9ca3af", "#6b7280", "#93b4d7", "#356b9c", "#a2b9a1", "#3a7350"]
ax.bar(labels, values, color=colors, yerr=errors, capsize=4, width=0.65)
for i, value in enumerate(values):
    ax.text(i, value + errors[i] + 0.12, f"{value:.2f}", ha="center", fontsize=10)
ax.set_ylim(0, 8.5)
ax.set_ylabel("Test MAE (km/h; lower is better)")
ax.set_title(
    "One-minute speed forecasting: chronological session holdout", loc="left", pad=16
)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.15)
ax.set_axisbelow(True)
fig.text(
    0.5,
    0.035,
    "112 windows, 2 held-out sessions. Model bars: 3-seed mean ± seed SD.\nTuned GRU improves 12.44% over baseline GRU, but only 0.97% over the 4-minute mean.",
    ha="center",
    fontsize=9,
)
fig.savefig(ROOT / "docs/forecasting/benchmark.png", dpi=180, bbox_inches="tight")
