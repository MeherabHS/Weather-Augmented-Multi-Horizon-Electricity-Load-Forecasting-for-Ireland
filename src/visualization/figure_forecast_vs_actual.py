# figure_forecast_vs_actual.py

"""
Purpose
-------
Plot actual load versus forecast median and prediction interval
using the saved weather-aware Quantile GBR prediction artifact.

Rationale
---------
The engineered horizon datasets do not contain model forecasts.
The correct visualization source is the saved model prediction file
written during the weather-aware horizon GBR run.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


# Use the final selected model's test predictions because the project conclusion is based on this model.
PREDICTION_FILE = "models/horizon_quantile_gbr_weather/t_plus_24/test_predictions.csv"

# Store output location so report artifacts remain organized.
OUTPUT_DIR = "reports/figures"

# Define the output figure path because the figure should be reproducible and exportable.
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "forecast_vs_actual.png")


def main():
    """
    Build the forecast-versus-actual visualization.
    """

    # Fail explicitly if the prediction artifact is missing because silent fallback would be misleading.
    if not os.path.exists(PREDICTION_FILE):
        raise FileNotFoundError(f"Prediction file not found: {PREDICTION_FILE}")

    # Ensure the figure output directory exists before saving.
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load the saved model predictions because this file contains actuals and quantile forecasts together.
    df = pd.read_csv(PREDICTION_FILE)

    # Validate the required schema because plotting should fail clearly if the artifact structure is wrong.
    required_columns = ["actual_target_mw", "pred_q10", "pred_q50", "pred_q90"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Prediction file is missing required columns: {missing_columns}"
        )

    # Use only the last 200 rows so the figure remains readable and publication-friendly.
    df = df.tail(200).reset_index(drop=True)

    # Create a simple integer x-axis because relative sequence order is sufficient for this figure.
    time = range(len(df))

    # Initialize the figure with a readable aspect ratio for reports and README usage.
    plt.figure(figsize=(10, 5))

    # Plot the actual realized load because this is the benchmark series against which forecasts are judged.
    plt.plot(time, df["actual_target_mw"], label="Actual Load")

    # Plot the median forecast because it is the model's central point estimate.
    plt.plot(time, df["pred_q50"], label="Forecast Median")

    # Plot the prediction interval because probabilistic uncertainty is a key project deliverable.
    plt.fill_between(
        time,
        df["pred_q10"],
        df["pred_q90"],
        alpha=0.3,
        label="Prediction Interval (10–90%)",
    )

    # Label the x-axis because the figure represents a chronological forecast window.
    plt.xlabel("Time Index")

    # Label the y-axis because the target is electricity load in MW.
    plt.ylabel("Load (MW)")

    # Add a precise title so the figure is self-explanatory in reports and GitHub documentation.
    plt.title("Forecast vs Actual Load (t+24, GBR + Weather)")

    # Add legend because the figure contains actuals, median forecast, and interval band.
    plt.legend()

    # Add grid for easier visual comparison of peaks, troughs, and interval width.
    plt.grid(True)

    # Tight layout reduces clipping risk during export.
    plt.tight_layout()

    # Save the figure at high resolution so it remains usable in documentation and presentations.
    plt.savefig(OUTPUT_FILE, dpi=300)

    # Print the output path so the user can verify successful generation immediately.
    print(f"Saved figure: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()