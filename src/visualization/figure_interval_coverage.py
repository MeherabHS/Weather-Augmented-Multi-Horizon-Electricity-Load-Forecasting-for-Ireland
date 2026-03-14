# figure_interval_coverage.py

"""
Purpose
-------
Visualize probabilistic forecast calibration using
80% prediction interval coverage.

Why
---
Coverage measures how often the true value falls inside the predicted interval.

An 80% interval should ideally contain the true value 80% of the time.
"""

import json
import os
import matplotlib.pyplot as plt
import pandas as pd

GBR_PATH = "models/horizon_quantile_gbr_weather/all_horizon_summary.json"
DEEPAR_PATH = "models/horizon_deepar_weather/all_horizon_summary.json"

OUTPUT_DIR = "reports/figures"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "interval_coverage.png")


def load_coverage(path):

    with open(path) as f:
        data = json.load(f)

    rows = []

    for entry in data["horizons"]:

        if "test_metrics" in entry:
            coverage = entry["test_metrics"]["coverage_80_interval"]
        else:
            coverage = entry["test_coverage_80_interval"]

        rows.append({
            "horizon": entry["horizon_label"],
            "coverage": coverage
        })

    return pd.DataFrame(rows)


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    gbr = load_coverage(GBR_PATH)
    deepar = load_coverage(DEEPAR_PATH)

    horizon_order = ["t_plus_1", "t_plus_24", "t_plus_168"]

    gbr["horizon"] = pd.Categorical(gbr["horizon"], horizon_order)
    deepar["horizon"] = pd.Categorical(deepar["horizon"], horizon_order)

    gbr = gbr.sort_values("horizon")
    deepar = deepar.sort_values("horizon")

    plt.figure(figsize=(8,5))

    plt.plot(gbr["horizon"], gbr["coverage"], marker="o", label="GBR + Weather")
    plt.plot(deepar["horizon"], deepar["coverage"], marker="o", label="DeepAR + Weather")

    plt.axhline(0.8, linestyle="--", label="Ideal 80% Coverage")

    plt.ylabel("Interval Coverage")
    plt.xlabel("Forecast Horizon")

    plt.title("Prediction Interval Coverage (80%)")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300)

    print(f"\nFigure saved: {OUTPUT_FILE}\n")


if __name__ == "__main__":
    main()