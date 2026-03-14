# Import json so evaluation metrics can be persisted for later comparison.
import json

# Import logging so the modeling process emits operational diagnostics.
import logging

# Import sys so the script can return explicit process exit codes.
import sys

# Import datetime utilities so run metadata timestamps remain standardized.
from datetime import datetime, timezone

# Import Path so filesystem operations remain deterministic.
from pathlib import Path

# Import pandas so tabular datasets can be loaded and evaluated.
import pandas as pd

# Import numpy because numerical operations are required for evaluation metrics.
import numpy as np


# Create a module-level logger so diagnostics remain consistent.
LOGGER = logging.getLogger("baseline_seasonal_naive")


# Define the directory containing the dataset splits.
SPLIT_DIR = Path("data/splits/entsoe")


# Define where model outputs should be stored.
MODEL_OUTPUT_DIR = Path("models/baselines")


# Configure logging.
def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )


# Ensure output directories exist.
def ensure_directory(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# Load dataset helper.
def load_dataset(path: Path) -> pd.DataFrame:

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Dataset empty: {path}")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    return df


# Seasonal naive forecast.
def seasonal_naive_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame, seasonal_lag: int = 168):

    history = train_df["load_mw"].tolist()

    predictions = []

    for i in range(len(test_df)):

        index = len(history) - seasonal_lag

        if index < 0:
            raise ValueError("Seasonal lag exceeds available history.")

        forecast = history[index]

        predictions.append(forecast)

        history.append(test_df["load_mw"].iloc[i])

    return np.array(predictions)


# Evaluation metrics.
def compute_metrics(actual: np.ndarray, forecast: np.ndarray):

    mae = np.mean(np.abs(actual - forecast))

    rmse = np.sqrt(np.mean((actual - forecast) ** 2))

    return {
        "MAE": float(mae),
        "RMSE": float(rmse)
    }


# Metadata timestamp.
def utc_now():
    return datetime.now(timezone.utc).isoformat()


# Main workflow.
def main():

    configure_logging()

    try:

        ensure_directory(MODEL_OUTPUT_DIR)

        train_path = SPLIT_DIR / "train.csv"
        validation_path = SPLIT_DIR / "validation.csv"
        test_path = SPLIT_DIR / "test.csv"

        train_df = load_dataset(train_path)
        validation_df = load_dataset(validation_path)
        test_df = load_dataset(test_path)

        LOGGER.info("Running seasonal naive baseline.")

        val_pred = seasonal_naive_forecast(train_df, validation_df)

        val_metrics = compute_metrics(validation_df["load_mw"].values, val_pred)

        LOGGER.info("Validation metrics computed.")

        train_plus_val = pd.concat([train_df, validation_df], ignore_index=True)

        test_pred = seasonal_naive_forecast(train_plus_val, test_df)

        test_metrics = compute_metrics(test_df["load_mw"].values, test_pred)

        LOGGER.info("Test metrics computed.")

        metrics = {
            "model": "seasonal_naive_weekly",
            "seasonal_lag_hours": 168,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "generated_at_utc": utc_now()
        }

        output_path = MODEL_OUTPUT_DIR / "seasonal_naive_metrics.json"

        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(json.dumps(metrics, indent=2))

        LOGGER.info("Seasonal naive baseline completed.")

        return 0

    except Exception as exc:

        LOGGER.exception("Seasonal naive model failed: %s", exc)

        return 1


if __name__ == "__main__":
    sys.exit(main())