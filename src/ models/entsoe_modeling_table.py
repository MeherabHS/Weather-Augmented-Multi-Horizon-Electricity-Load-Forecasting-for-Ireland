# Import json so run metadata can be persisted for auditability and later review.
import json

# Import logging so the modeling-table build emits operational diagnostics.
import logging

# Import sys so the script can return explicit process exit codes.
import sys

# Import datetime so metadata timestamps remain timezone-aware and standardized.
from datetime import datetime
from datetime import timezone

# Import Path so file-system operations remain portable and deterministic.
from pathlib import Path

# Import typing helpers so interfaces remain explicit and less error-prone.
from typing import Dict

# Import numpy so cyclical encodings can be computed efficiently and deterministically.
import numpy as np

# Import pandas so hourly data can be merged, indexed, and feature-engineered safely.
import pandas as pd


# Create a module-level logger so all build stages report through one consistent channel.
LOGGER = logging.getLogger("entsoe_modeling_table")


# Store the processed input directory because this step consumes Step 3 outputs only.
PROCESSED_DIR = Path("data/processed/entsoe")


# Store the modeling output directory so analysis-ready tables remain separate from intermediate artifacts.
MODELING_DIR = Path("data/modeling/entsoe")


# Configure logging once so terminal output remains structured and readable.
def configure_logging() -> None:
    # Initialize logging with timestamped structured formatting for reproducibility and debugging.
    logging.basicConfig(
        # Use INFO because lifecycle visibility is needed without excessive verbosity.
        level=logging.INFO,
        # Use a structured format so logs remain readable in terminal and saved records.
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# Ensure a directory exists before writing files so failures reflect logic issues, not missing folders.
def ensure_directory(path: Path) -> None:
    # Create the full directory tree idempotently because repeated pipeline runs are expected.
    path.mkdir(parents=True, exist_ok=True)


# Return an ISO-8601 UTC timestamp so metadata remains unambiguous across systems.
def utc_now_iso() -> str:
    # Generate the current UTC timestamp because lineage should be preserved explicitly.
    return datetime.now(timezone.utc).isoformat()


# Read a CSV file because upstream processed artifacts are stored as flat tabular files.
def read_csv(path: Path) -> pd.DataFrame:
    # Load the CSV into a dataframe so merging and feature engineering can be applied.
    dataframe = pd.read_csv(path)
    # Fail explicitly if the dataset is empty because an empty upstream artifact is not usable for forecasting.
    if dataframe.empty:
        # Raise a value error so the operator can inspect the upstream preprocessing result.
        raise ValueError(f"Input dataset is empty: {path}")
    # Return the loaded dataframe so downstream steps can operate on it.
    return dataframe


# Write a JSON file because pipeline metadata should remain machine-readable and reviewable.
def write_json(path: Path, payload: Dict) -> None:
    # Open the file in UTF-8 mode so serialization remains portable.
    with path.open("w", encoding="utf-8") as handle:
        # Write formatted JSON so the metadata is easy to inspect manually.
        json.dump(payload, handle, indent=2)


# Build the canonical 2024 hourly UTC index because the project currently targets calendar year 2024.
def build_expected_hourly_index() -> pd.DatetimeIndex:
    # Create the full leap-year hourly index so the modeling table has an explicit temporal contract.
    return pd.date_range(
        start="2024-01-01 00:00:00+00:00",
        end="2024-12-31 23:00:00+00:00",
        freq="h",
        tz="UTC",
    )


# Prepare the load table because the target variable must be one row per hour with explicit missingness flags.
def prepare_load_table(load_df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy so the source dataframe remains unchanged for debugging and lineage.
    working_df = load_df.copy()
    # Parse timestamps as UTC because all temporal joins must occur in one coherent timezone.
    working_df["timestamp_utc"] = pd.to_datetime(working_df["timestamp_utc"], utc=True)
    # Keep only the columns required for the target table so the artifact remains lean.
    working_df = working_df[["timestamp_utc", "load_mw"]].copy()
    # Sort chronologically because deterministic ordering is necessary for lag and rolling logic later.
    working_df.sort_values("timestamp_utc", inplace=True)
    # Create the full expected 2024 hourly frame so structural gaps become explicit rather than implicit.
    base_df = pd.DataFrame({"timestamp_utc": build_expected_hourly_index()})
    # Left-join the observed load onto the canonical hourly frame so all hours are represented.
    merged_df = base_df.merge(working_df, on="timestamp_utc", how="left")
    # Flag missing target observations because target imputation should not be hidden from later modeling steps.
    merged_df["load_missing_flag"] = merged_df["load_mw"].isna().astype(int)
    # Return the target table because it becomes the spine of the modeling dataset.
    return merged_df


# Extract onshore wind because ENTSO-E psrType B19 represents Wind Onshore and is present in the audit output.
def prepare_wind_table(generation_df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy so the source dataframe remains unchanged for debugging and lineage.
    working_df = generation_df.copy()
    # Parse timestamps as UTC because all source joins must use one consistent chronology.
    working_df["timestamp_utc"] = pd.to_datetime(working_df["timestamp_utc"], utc=True)
    # Filter to B19 only because this dataset’s available wind subtype is onshore wind.
    working_df = working_df.loc[working_df["psr_type"] == "B19"].copy()
    # Keep only the columns needed for the exogenous wind feature so the table remains compact.
    working_df = working_df[["timestamp_utc", "generation_mw"]].copy()
    # Rename the generation column so downstream modeling code remains semantically clear.
    working_df.rename(columns={"generation_mw": "wind_onshore_mw"}, inplace=True)
    # Sort chronologically because later interpolation logic should operate on ordered data.
    working_df.sort_values("timestamp_utc", inplace=True)
    # Group by timestamp and average conservatively in case multiple B19 series overlap for the same hour.
    working_df = working_df.groupby("timestamp_utc", as_index=False)["wind_onshore_mw"].mean()
    # Create the full expected hourly base so wind coverage gaps become explicit.
    base_df = pd.DataFrame({"timestamp_utc": build_expected_hourly_index()})
    # Left-join observed wind onto the canonical hourly frame so all hours are represented.
    merged_df = base_df.merge(working_df, on="timestamp_utc", how="left")
    # Flag raw wind missingness before any gap treatment so auditability is preserved.
    merged_df["wind_missing_flag_raw"] = merged_df["wind_onshore_mw"].isna().astype(int)
    # Apply time interpolation with a short limit because short exogenous gaps can be smoothed conservatively.
    merged_df["wind_onshore_mw"] = merged_df["wind_onshore_mw"].interpolate(
        method="linear",
        limit=3,
        limit_direction="both",
    )
    # Flag remaining wind missingness after conservative interpolation so unresolved gaps stay explicit.
    merged_df["wind_missing_flag_postfill"] = merged_df["wind_onshore_mw"].isna().astype(int)
    # Return the wind exogenous table because it will be merged into the final modeling dataset.
    return merged_df


# Add calendar and cyclical fields because the project brief explicitly requires them in the modeling table.
def add_calendar_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy so the caller’s dataframe remains unchanged.
    result = dataframe.copy()
    # Extract hour-of-day because short-term load has strong intraday seasonality.
    result["hour_of_day"] = result["timestamp_utc"].dt.hour
    # Extract day-of-week because load patterns differ materially across weekdays and weekends.
    result["day_of_week"] = result["timestamp_utc"].dt.dayofweek
    # Extract month because seasonal load structure changes over the year.
    result["month"] = result["timestamp_utc"].dt.month
    # Extract day-of-year because annual position can help later diagnostics and plotting.
    result["day_of_year"] = result["timestamp_utc"].dt.dayofyear
    # Encode hour cyclically so models can represent 23→0 continuity without ordinal distortion.
    result["sin_hour"] = np.sin(2 * np.pi * result["hour_of_day"] / 24)
    # Encode hour cyclically so intraday periodicity remains rotation-consistent.
    result["cos_hour"] = np.cos(2 * np.pi * result["hour_of_day"] / 24)
    # Encode day-of-week cyclically so weekly periodicity remains continuous rather than ordinal.
    result["sin_dow"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
    # Encode day-of-week cyclically so the weekly cycle can be learned without edge discontinuity.
    result["cos_dow"] = np.cos(2 * np.pi * result["day_of_week"] / 7)
    # Encode month cyclically so December→January continuity is preserved.
    result["sin_month"] = np.sin(2 * np.pi * (result["month"] - 1) / 12)
    # Encode month cyclically so annual seasonality can be represented smoothly.
    result["cos_month"] = np.cos(2 * np.pi * (result["month"] - 1) / 12)
    # Return the enriched dataframe because these features are part of the project specification.
    return result


# Add lag and rolling features because the project brief requires them for forecasting models.
def add_load_history_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy so prior pipeline state is preserved for debugging and comparison.
    result = dataframe.copy()
    # Create the 1-hour lag because immediate autocorrelation is a core load signal.
    result["load_lag_1"] = result["load_mw"].shift(1)
    # Create the 24-hour lag because day-ahead recurrence is operationally meaningful.
    result["load_lag_24"] = result["load_mw"].shift(24)
    # Create the 48-hour lag because two-day recurrence can stabilize short-horizon forecasting.
    result["load_lag_48"] = result["load_mw"].shift(48)
    # Create the 168-hour lag because weekly recurrence is important for load forecasting.
    result["load_lag_168"] = result["load_mw"].shift(168)
    # Compute the trailing 24-hour mean because recent level context often improves forecast calibration.
    result["load_rollmean_24"] = result["load_mw"].rolling(window=24, min_periods=24).mean()
    # Compute the trailing 24-hour standard deviation because recent volatility is operationally informative.
    result["load_rollstd_24"] = result["load_mw"].rolling(window=24, min_periods=24).std()
    # Return the dataframe with historical features because they are required by the project brief.
    return result


# Build the final modeling table because load, wind, and temporal features must exist in one row-per-hour dataset.
def build_modeling_table(load_df: pd.DataFrame, wind_df: pd.DataFrame) -> pd.DataFrame:
    # Merge load and wind on the canonical hourly timestamp because the dataset contract is one row per hour.
    merged_df = load_df.merge(wind_df, on="timestamp_utc", how="left")
    # Add calendar and cyclical features because they are required by the project design.
    merged_df = add_calendar_features(merged_df)
    # Add lag and rolling target-history features because the baseline models need them.
    merged_df = add_load_history_features(merged_df)
    # Add a model-eligibility flag so rows with missing target values can be excluded transparently later.
    merged_df["target_available_for_training"] = merged_df["load_mw"].notna().astype(int)
    # Return the completed modeling table because it is the direct output of this step.
    return merged_df


# Build a compact quality summary because the resulting table must be validated before backtesting and training.
def build_quality_summary(modeling_df: pd.DataFrame) -> Dict:
    # Count total rows because the table should span the full 2024 hourly index.
    total_rows = int(len(modeling_df))
    # Count target-missing rows because target gaps must remain explicit for honest modeling.
    target_missing_rows = int(modeling_df["load_mw"].isna().sum())
    # Count wind-missing rows after conservative fill because unresolved exogenous gaps affect usable coverage.
    wind_missing_rows = int(modeling_df["wind_onshore_mw"].isna().sum())
    # Count rows eligible for training because this is the effective supervised sample size.
    trainable_rows = int(modeling_df["target_available_for_training"].sum())
    # Return the summary because it should be stored in metadata for project governance.
    return {
        "row_count": total_rows,
        "target_missing_row_count": target_missing_rows,
        "wind_missing_row_count_after_fill": wind_missing_rows,
        "trainable_row_count": trainable_rows,
        "start_utc": str(modeling_df["timestamp_utc"].min()),
        "end_utc": str(modeling_df["timestamp_utc"].max()),
    }


# Execute the modeling-table build because the project now needs a row-per-hour analysis dataset.
def main() -> int:
    # Configure logging before any file or dataframe work begins.
    configure_logging()
    try:
        # Ensure the modeling output directory exists before writing artifacts.
        ensure_directory(MODELING_DIR)
        # Define the processed load input path because it is the forecasting target source.
        load_input_path = PROCESSED_DIR / "ireland_load_hourly.csv"
        # Define the processed generation input path because it contains the wind subtype source.
        generation_input_path = PROCESSED_DIR / "ireland_generation_per_type_hourly.csv"
        # Load the processed hourly load dataset because the target series is required.
        load_df = read_csv(load_input_path)
        # Load the processed hourly generation dataset because wind extraction depends on it.
        generation_df = read_csv(generation_input_path)
        # Prepare the target load table because target missingness must be preserved explicitly.
        prepared_load_df = prepare_load_table(load_df)
        # Prepare the wind table because B19 onshore wind is the relevant exogenous feature.
        prepared_wind_df = prepare_wind_table(generation_df)
        # Build the final modeling table because all features must live in one hourly frame.
        modeling_df = build_modeling_table(prepared_load_df, prepared_wind_df)
        # Define the modeling-table output path because the next project stages consume this artifact.
        modeling_output_path = MODELING_DIR / "ireland_load_modeling_table.csv"
        # Define the metadata path because this step should preserve its own lineage and quality summary.
        metadata_output_path = MODELING_DIR / "entsoe_modeling_table_metadata.json"
        # Persist the modeling table because downstream backtesting and training need a stable input file.
        modeling_df.to_csv(modeling_output_path, index=False)
        # Build the quality summary because row coverage and missingness must be documented.
        quality_summary = build_quality_summary(modeling_df)
        # Persist metadata so this step remains auditable and reviewable.
        write_json(
            metadata_output_path,
            {
                "generated_at_utc": utc_now_iso(),
                "source_load_input_path": str(load_input_path),
                "source_generation_input_path": str(generation_input_path),
                "wind_feature_source_psr_type": "B19",
                "wind_feature_description": "ENTSO-E Wind Onshore",
                "quality_summary": quality_summary,
                "note": "Target missing values are intentionally preserved for transparent exclusion during model training.",
            },
        )
        # Log successful completion so the operator can distinguish a full run from a partial one.
        LOGGER.info("ENTSO-E modeling table build completed successfully.")
        # Print a concise summary so the operator can validate the output immediately.
        print(
            json.dumps(
                {
                    "modeling_output_path": str(modeling_output_path),
                    "metadata_output_path": str(metadata_output_path),
                    "quality_summary": quality_summary,
                },
                indent=2,
            )
        )
        # Return success because the build completed without exceptions.
        return 0
    except Exception as exc:
        # Log the full exception because silent feature-table failures would compromise the project.
        LOGGER.exception("ENTSO-E modeling table build failed: %s", exc)
        # Return failure so shells and automation can detect the issue.
        return 1


# Execute the workflow only when the file is run directly so imports remain side-effect free.
if __name__ == "__main__":
    # Exit with the workflow status so terminal execution gets an accurate process code.
    sys.exit(main())