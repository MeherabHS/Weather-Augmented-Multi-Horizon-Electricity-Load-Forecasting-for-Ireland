# final_model_comparison_report.py

"""
Purpose
-------
Creates the final comparison table between all trained models.

Rationale
---------
The forecasting pipeline already produced evaluation artifacts for each
model family. Recomputing forecasts would introduce unnecessary compute
cost and possible inconsistencies. Instead, this reporting script loads
the saved evaluation summaries and performs deterministic aggregation.

Design Principles
-----------------
- Idempotent: repeated execution produces identical results
- Defensive: validates file existence and schema
- Reproducible: outputs structured artifacts for downstream figures
"""

import json
import os
import logging
import pandas as pd

# ------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger("final_model_comparison")


# ------------------------------------------------------------
# Configuration paths
# ------------------------------------------------------------

MODEL_FILES = {
    "GBR": "models/horizon_quantile_gbr/all_horizon_summary.json",
    "GBR_weather": "models/horizon_quantile_gbr_weather/all_horizon_summary.json",
    "DeepAR_weather": "models/horizon_deepar_weather/all_horizon_summary.json"
}

OUTPUT_DIR = "reports"
CSV_OUTPUT = os.path.join(OUTPUT_DIR, "model_comparison_table.csv")
JSON_OUTPUT = os.path.join(OUTPUT_DIR, "model_comparison_table.json")


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def validate_file(path: str):
    """
    Ensures required artifact exists before processing.

    This prevents silent failures where downstream calculations
    operate on missing or incorrect inputs.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required model summary not found: {path}")


def load_model_summary(model_name: str, path: str) -> pd.DataFrame:
    """
    Loads horizon evaluation metrics from model summary JSON.

    This loader is deliberately defensive because different model pipelines
    may serialize RMSE using different schemas.

    Supported patterns:
    1. entry["test_rmse"]
    2. entry["rmse"]
    3. entry["rmse_test"]
    4. entry["test_error"]
    5. entry["test_metrics"]["point_metrics"]["RMSE"]
    6. entry["validation_metrics"]["point_metrics"]["RMSE"]  # fallback only if test is unavailable
    """

    validate_file(path)

    with open(path, "r") as f:
        data = json.load(f)

    rows = []

    for entry in data["horizons"]:

        # Extract horizon label explicitly because all downstream comparison depends on it.
        horizon_label = entry.get("horizon_label")

        if horizon_label is None:
            raise ValueError(
                f"Horizon label missing in entry for {model_name}: {entry}"
            )

        # Initialize rmse as missing because extraction must be explicit and validated.
        rmse = None

        # ---- Flat schema candidates ----
        for candidate in ["test_rmse", "rmse", "rmse_test", "test_error"]:
            if candidate in entry:
                rmse = entry[candidate]
                break

        # ---- Nested schema: preferred test metric path ----
        if rmse is None:
            if (
                isinstance(entry.get("test_metrics"), dict)
                and isinstance(entry["test_metrics"].get("point_metrics"), dict)
                and "RMSE" in entry["test_metrics"]["point_metrics"]
            ):
                rmse = entry["test_metrics"]["point_metrics"]["RMSE"]

        # ---- Nested fallback path: validation only if test truly absent ----
        if rmse is None:
            if (
                isinstance(entry.get("validation_metrics"), dict)
                and isinstance(entry["validation_metrics"].get("point_metrics"), dict)
                and "RMSE" in entry["validation_metrics"]["point_metrics"]
            ):
                rmse = entry["validation_metrics"]["point_metrics"]["RMSE"]

        # Fail explicitly if no supported RMSE path exists because silent coercion would corrupt the report.
        if rmse is None:
            raise ValueError(
                f"RMSE field not found in horizon entry for {model_name}: {entry}"
            )

        # Append the normalized record because the comparison table requires a consistent schema.
        rows.append({
            "horizon": horizon_label,
            "rmse": float(rmse),
            "model": model_name
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Core aggregation logic
# ------------------------------------------------------------

def build_comparison_table():

    logger.info("Loading model evaluation summaries")

    frames = []

    for model_name, path in MODEL_FILES.items():
        df = load_model_summary(model_name, path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    logger.info("Constructing pivot comparison table")

    pivot = combined.pivot(
        index="horizon",
        columns="model",
        values="rmse"
    )

        # Identify only the model columns because horizon is categorical and should not be included.
    model_columns = [col for col in pivot.columns if col != "horizon"]

    # Enforce numeric conversion because JSON loading may sometimes leave values as strings.
    pivot[model_columns] = pivot[model_columns].apply(pd.to_numeric, errors="coerce")

    # Compute best model per horizon using numeric columns only.
    pivot["best_model"] = pivot[model_columns].idxmin(axis=1)

    # Compute best RMSE value.
    pivot["best_rmse"] = pivot[model_columns].min(axis=1)

    pivot.reset_index(inplace=True)

    return pivot


# ------------------------------------------------------------
# Output persistence
# ------------------------------------------------------------

def save_outputs(df: pd.DataFrame):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df.to_csv(CSV_OUTPUT, index=False)

    df.to_json(JSON_OUTPUT, orient="records", indent=2)

    logger.info(f"Saved CSV report: {CSV_OUTPUT}")
    logger.info(f"Saved JSON report: {JSON_OUTPUT}")


# ------------------------------------------------------------
# Main execution
# ------------------------------------------------------------

def main():

    logger.info("Starting model comparison report generation")

    comparison_df = build_comparison_table()

    save_outputs(comparison_df)

    logger.info("Final comparison table")
    print("\n")
    print(comparison_df.to_string(index=False))
    print("\n")

    logger.info("Model comparison report completed successfully")


if __name__ == "__main__":
    main()