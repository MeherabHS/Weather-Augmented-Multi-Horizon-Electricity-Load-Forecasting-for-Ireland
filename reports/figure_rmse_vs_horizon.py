# figure_rmse_vs_horizon.py

"""
Purpose
-------
Generate the primary research figure showing RMSE vs forecast horizon
for all trained models.

Why
---
This figure communicates the key experimental result:
how forecast error changes as the prediction horizon increases and
how different model families compare.

The script reads the already generated comparison report to ensure
the visualization is reproducible and consistent with the saved metrics.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

INPUT_PATH = "reports/model_comparison_table.csv"
OUTPUT_DIR = "reports/figures"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "rmse_vs_horizon.png")


def main():

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError("Model comparison table not found.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    horizon_order = ["t_plus_1", "t_plus_24", "t_plus_168"]
    df["horizon"] = pd.Categorical(df["horizon"], horizon_order)

    df = df.sort_values("horizon")

    plt.figure(figsize=(8,5))

    plt.plot(df["horizon"], df["GBR"], marker="o", label="GBR Baseline")
    plt.plot(df["horizon"], df["GBR_weather"], marker="o", label="GBR + Weather")
    plt.plot(df["horizon"], df["DeepAR_weather"], marker="o", label="DeepAR + Weather")

    plt.xlabel("Forecast Horizon")
    plt.ylabel("RMSE")
    plt.title("Forecast Error vs Prediction Horizon")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300)

    print(f"\nFigure saved: {OUTPUT_FILE}\n")


if __name__ == "__main__":
    main()