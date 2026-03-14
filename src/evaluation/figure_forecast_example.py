import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "data/horizons_weather/entsoe/ireland_load_t_plus_24.csv"
OUTPUT_DIR = "reports/figures"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "forecast_example.png")


def detect_target_column(df):
    """
    Automatically detect the target column.

    This prevents failures when datasets use horizon-specific
    naming such as load_target_t_plus_24.
    """

    candidates = [c for c in df.columns if c.startswith("load_target_")]

    if not candidates:
        raise ValueError("No target column found in dataset.")

    return candidates[0]


def main():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Dataset not found.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # Detect correct target column
    target_col = detect_target_column(df)

    # Select a small window to visualize
    df = df.tail(200)

    time = range(len(df))

    plt.figure(figsize=(10,5))

    # Actual load
    plt.plot(time, df[target_col], label="Actual Load")

    # Median prediction if available
    if "pred_q50" in df.columns:
        plt.plot(time, df["pred_q50"], label="Forecast Median")

    # Prediction interval
    if "pred_q10" in df.columns and "pred_q90" in df.columns:
        plt.fill_between(
            time,
            df["pred_q10"],
            df["pred_q90"],
            alpha=0.3,
            label="Prediction Interval"
        )

    plt.xlabel("Time")
    plt.ylabel("Load (MW)")
    plt.title("Example Forecast Window")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300)

    print("\nFigure saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()