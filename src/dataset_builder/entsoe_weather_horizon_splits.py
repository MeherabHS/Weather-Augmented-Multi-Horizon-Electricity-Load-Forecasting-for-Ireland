# Import json so split metadata can be persisted for auditability and later review.
import json

# Import logging so execution diagnostics remain visible and structured.
import logging

# Import sys so the script can return explicit process exit codes.
import sys

# Import datetime utilities for reproducible metadata timestamps.
from datetime import datetime
from datetime import timezone

# Import Path so filesystem operations remain deterministic and portable.
from pathlib import Path

# Import typing helpers so interfaces remain explicit and lower-risk.
from typing import Dict
from typing import List

# Import pandas for dataset manipulation and chronological filtering.
import pandas as pd


# Create a module-level logger so all diagnostics use one consistent channel.
LOGGER = logging.getLogger("entsoe_weather_horizon_splits")


# Store the weather-horizon dataset directory because this step consumes those corrected horizon tables.
INPUT_DIR = Path("data/horizons_weather/entsoe")


# Store the split output directory so weather-aware split artifacts remain separate from prior non-weather splits.
OUTPUT_DIR = Path("data/splits_weather/entsoe")


# Store the fixed chronological boundaries because all horizons should share one time-based split policy.
TRAIN_END = pd.Timestamp("2024-09-30 23:00:00+00:00")

# Store the validation end boundary because the middle window is reserved for model selection.
VAL_END = pd.Timestamp("2024-11-15 23:00:00+00:00")


# Configure logging once so terminal diagnostics remain structured and readable.
def configure_logging() -> None:
    # Initialize logging with timestamped structured formatting for reproducibility and debugging.
    logging.basicConfig(
        # Use INFO because lifecycle visibility is needed without excessive verbosity.
        level=logging.INFO,
        # Use a structured format so logs remain readable in terminal and stored records.
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# Ensure a directory exists before writing files so failures reflect logic issues, not missing folders.
def ensure_directory(path: Path) -> None:
    # Create the full directory tree idempotently because repeated split runs are expected.
    path.mkdir(parents=True, exist_ok=True)


# Return an ISO-8601 UTC timestamp so metadata remains temporally explicit.
def utc_now_iso() -> str:
    # Generate the current UTC timestamp because lineage should be preserved explicitly.
    return datetime.now(timezone.utc).isoformat()


# Return the agreed project horizons because each weather-aware task requires its own split files.
def get_horizon_specifications() -> List[Dict]:
    # Return the horizon labels and lead times so downstream logic remains contract-aligned.
    return [
        {"label": "t_plus_1", "lead_hours": 1},
        {"label": "t_plus_24", "lead_hours": 24},
        {"label": "t_plus_168", "lead_hours": 168},
    ]


# Read one weather-aware horizon CSV because each horizon is now a distinct supervised forecasting task.
def read_dataset(path: Path) -> pd.DataFrame:
    # Load the CSV into a dataframe so chronological splitting can be applied.
    df = pd.read_csv(path)
    # Fail explicitly if the dataset is empty because no valid split can be created from it.
    if df.empty:
        # Raise a value error so the operator can inspect the upstream horizon-dataset build.
        raise ValueError(f"Horizon dataset is empty: {path}")
    # Parse timestamps as UTC because time-based splitting must operate on coherent temporal values.
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    # Sort chronologically because deterministic time slicing requires ordered observations.
    df.sort_values("timestamp_utc", inplace=True)
    # Reset the index so downstream exports remain clean and sequential.
    df.reset_index(drop=True, inplace=True)
    # Return the prepared dataframe because it is now suitable for filtering and splitting.
    return df


# Filter to rows with available future target because supervised learning requires observed horizon outcomes.
def filter_available_targets(df: pd.DataFrame, horizon_label: str) -> pd.DataFrame:
    # Build the horizon-specific availability flag because each horizon has its own future target.
    availability_column = f"target_available_{horizon_label}"
    # Fail explicitly if the required flag is missing because the horizon dataset schema would be invalid.
    if availability_column not in df.columns:
        # Raise a key error so the operator knows the upstream horizon builder did not produce the expected schema.
        raise KeyError(f"Required availability column missing: {availability_column}")
    # Keep only rows with observed future target because training and evaluation require known outcomes.
    result = df.loc[df[availability_column] == 1].copy()
    # Reset the index so exported splits remain clean and standalone.
    result.reset_index(drop=True, inplace=True)
    # Return the filtered dataframe because it is the valid supervised base for this horizon.
    return result


# Apply the fixed chronological split because all horizons should use comparable time-based partitions.
def split_dataset(df: pd.DataFrame):
    # Build the training subset from the earliest chronological observations only.
    train = df[df["timestamp_utc"] <= TRAIN_END].copy()
    # Build the validation subset from the middle chronological window only.
    validation = df[(df["timestamp_utc"] > TRAIN_END) & (df["timestamp_utc"] <= VAL_END)].copy()
    # Build the test subset from the final untouched chronological window only.
    test = df[df["timestamp_utc"] > VAL_END].copy()
    # Return the three subsets because downstream model training depends on this partition structure.
    return train, validation, test


# Build a compact split summary because partition coverage and sample size must be documented explicitly.
def summarize(df: pd.DataFrame, horizon_label: str) -> Dict:
    # Build the horizon-specific target column name because each horizon predicts a different future load.
    target_column = f"load_target_{horizon_label}"
    # Return row count and temporal coverage because split validity should be easy to inspect.
    return {
        "row_count": int(len(df)),
        "start_utc": str(df["timestamp_utc"].min()) if not df.empty else None,
        "end_utc": str(df["timestamp_utc"].max()) if not df.empty else None,
        "target_missing_row_count": int(df[target_column].isna().sum()) if not df.empty else 0,
        "wind_missing_row_count": int(df["wind_onshore_mw"].isna().sum()) if not df.empty else 0,
    }


# Write a JSON file because split metadata should remain machine-readable and auditable.
def write_json(path: Path, payload: Dict) -> None:
    # Open the destination file in UTF-8 mode so serialization remains portable.
    with path.open("w", encoding="utf-8") as handle:
        # Persist formatted JSON so the metadata remains easy to inspect manually.
        json.dump(payload, handle, indent=2)


# Execute the corrected weather split workflow because each forecast horizon requires valid supervised partitions.
def main():
    # Configure logging before any file or dataframe work begins.
    configure_logging()
    try:
        # Ensure the output root exists before writing artifacts.
        ensure_directory(OUTPUT_DIR)
        # Initialize metadata storage because each horizon output should be documented explicitly.
        metadata = []
        # Iterate through the agreed project horizons because each one is a distinct supervised task.
        for horizon_spec in get_horizon_specifications():
            # Extract the horizon label because it controls input selection and schema interpretation.
            label = horizon_spec["label"]
            # Extract the lead hours because the metadata should preserve task semantics explicitly.
            hours = horizon_spec["lead_hours"]
            # Define the input path so the correct weather-aware horizon dataset is loaded.
            input_path = INPUT_DIR / f"ireland_load_{label}.csv"
            # Load the weather-aware horizon dataset because chronological partitioning depends on it.
            df = read_dataset(input_path)
            # Filter out rows with unavailable future target because supervised evaluation requires observed outcomes.
            df = filter_available_targets(df, label)
            # Split the filtered dataset chronologically because time-based validation is mandatory.
            train, validation, test = split_dataset(df)
            # Define the horizon-specific output directory so each task remains clearly isolated.
            horizon_dir = OUTPUT_DIR / label
            # Ensure the horizon-specific directory exists before writing files.
            ensure_directory(horizon_dir)
            # Define the split output paths because each partition should be persisted independently.
            train_path = horizon_dir / "train.csv"
            # Define the validation output path because model selection depends on it.
            val_path = horizon_dir / "validation.csv"
            # Define the test output path because final untouched evaluation depends on it.
            test_path = horizon_dir / "test.csv"
            # Write the training partition because downstream models will fit on this subset.
            train.to_csv(train_path, index=False)
            # Write the validation partition because downstream model selection will use this subset.
            validation.to_csv(val_path, index=False)
            # Write the test partition because downstream final evaluation will use this subset.
            test.to_csv(test_path, index=False)
            # Build a horizon summary because split quality should remain auditable.
            summary = {
                "horizon_label": label,
                "lead_hours": hours,
                "input_path": str(input_path),
                "train_output_path": str(train_path),
                "validation_output_path": str(val_path),
                "test_output_path": str(test_path),
                "summaries": {
                    "train": summarize(train, label),
                    "validation": summarize(validation, label),
                    "test": summarize(test, label),
                },
            }
            # Append the summary so the full run can be reviewed after completion.
            metadata.append(summary)
            # Log completion of the current horizon because progress visibility matters.
            LOGGER.info("Built corrected weather splits for horizon: %s", label)
        # Define the metadata output path because the corrected split step should preserve its own lineage.
        metadata_path = OUTPUT_DIR / "weather_split_metadata.json"
        # Persist the metadata because downstream modeling should reference a durable split contract.
        write_json(
            metadata_path,
            {
                "generated_at_utc": utc_now_iso(),
                "split_policy": "Chronological split only; rows without future target availability removed before splitting.",
                "boundaries": {
                    "train_end_utc": str(TRAIN_END),
                    "validation_end_utc": str(VAL_END),
                },
                "horizons": metadata,
            },
        )
        # Print a concise summary so the operator can validate the correction immediately.
        print(
            json.dumps(
                {
                    "metadata_output_path": str(metadata_path),
                    "horizons": metadata,
                },
                indent=2,
            )
        )
        # Log successful completion so the operator can distinguish a full run from a partial one.
        LOGGER.info("Corrected weather horizon splits completed successfully.")
        # Return success because the workflow completed without exceptions.
        return 0
    except Exception as exc:
        # Log the full exception because silent split failures would compromise the downstream model comparison.
        LOGGER.exception("Corrected weather split build failed: %s", exc)
        # Return failure so terminal execution can detect the issue.
        return 1


# Execute the workflow only when the script is run directly so imports remain side-effect free.
if __name__ == "__main__":
    # Exit with the workflow status so the shell receives an accurate process code.
    sys.exit(main())