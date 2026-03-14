# Import json so split metadata can be persisted for auditability and later review.
import json

# Import logging so the split workflow emits operational diagnostics.
import logging

# Import sys so the script can return explicit process exit codes.
import sys

# Import datetime so metadata timestamps remain timezone-aware and standardized.
from datetime import datetime
from datetime import timezone

# Import Path so filesystem operations remain portable and deterministic.
from pathlib import Path

# Import typing helpers so interfaces remain explicit and lower-risk.
from typing import Dict
from typing import List

# Import pandas so horizon datasets can be filtered into chronological partitions.
import pandas as pd


# Create a module-level logger so all diagnostics use one consistent channel.
LOGGER = logging.getLogger("entsoe_horizon_splits")


# Store the horizon dataset directory because this step consumes the corrected supervised datasets.
HORIZON_DIR = Path("data/horizons/entsoe")


# Store the split output directory so horizon-specific partitions remain organized and separated.
SPLIT_DIR = Path("data/splits_horizon/entsoe")


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


# Read one horizon CSV because each horizon is now a distinct supervised forecasting task.
def read_horizon_dataset(path: Path) -> pd.DataFrame:
    # Load the CSV into a dataframe so chronological splitting can be applied.
    dataframe = pd.read_csv(path)
    # Fail explicitly if the dataset is empty because no valid split can be created from it.
    if dataframe.empty:
        # Raise a value error so the operator can inspect the upstream horizon-dataset build.
        raise ValueError(f"Horizon dataset is empty: {path}")
    # Parse timestamps as UTC because time-based splitting must operate on coherent temporal values.
    dataframe["timestamp_utc"] = pd.to_datetime(dataframe["timestamp_utc"], utc=True)
    # Sort chronologically because deterministic time slicing requires ordered observations.
    dataframe.sort_values("timestamp_utc", inplace=True)
    # Reset the index so downstream exports remain clean and sequential.
    dataframe.reset_index(drop=True, inplace=True)
    # Return the prepared dataframe because it is now suitable for horizon-specific splitting.
    return dataframe


# Write a JSON file because split metadata should remain machine-readable and auditable.
def write_json(path: Path, payload: Dict) -> None:
    # Open the destination file in UTF-8 mode so serialization remains portable.
    with path.open("w", encoding="utf-8") as handle:
        # Persist formatted JSON so the metadata remains easy to inspect manually.
        json.dump(payload, handle, indent=2)


# Return the agreed project horizons because splits must be generated for each task explicitly.
def get_horizon_specifications() -> List[Dict]:
    # Return the horizon labels and lead times so downstream logic remains contract-aligned.
    return [
        {"label": "t_plus_1", "lead_hours": 1},
        {"label": "t_plus_24", "lead_hours": 24},
        {"label": "t_plus_168", "lead_hours": 168},
    ]


# Remove rows without future target availability because supervised evaluation requires observed future outcomes.
def filter_available_target_rows(dataframe: pd.DataFrame, horizon_label: str) -> pd.DataFrame:
    # Build the target-availability column name because it is horizon-specific by design.
    target_flag_column = f"target_available_{horizon_label}"
    # Keep only rows where the future target exists because model training cannot use undefined outcomes.
    result = dataframe.loc[dataframe[target_flag_column] == 1].copy()
    # Reset the index so split exports remain clean and standalone.
    result.reset_index(drop=True, inplace=True)
    # Return the filtered dataframe because it is the valid supervised base for this horizon.
    return result


# Apply the fixed chronological split because all horizons must use comparable time-based partitions.
def build_time_splits(dataframe: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    # Define the train end boundary because the first nine months remain the main learning window.
    train_end = pd.Timestamp("2024-09-30 23:00:00+00:00")
    # Define the validation start boundary because validation should immediately follow train.
    validation_start = pd.Timestamp("2024-10-01 00:00:00+00:00")
    # Define the validation end boundary because this window supports model selection.
    validation_end = pd.Timestamp("2024-11-15 23:00:00+00:00")
    # Define the test start boundary because the final segment must remain untouched during selection.
    test_start = pd.Timestamp("2024-11-16 00:00:00+00:00")
    # Build the training partition using only the earliest chronological observations.
    train_df = dataframe.loc[dataframe["timestamp_utc"] <= train_end].copy()
    # Build the validation partition using the middle chronological window only.
    validation_df = dataframe.loc[
        (dataframe["timestamp_utc"] >= validation_start) & (dataframe["timestamp_utc"] <= validation_end)
    ].copy()
    # Build the test partition using the final untouched chronological window only.
    test_df = dataframe.loc[dataframe["timestamp_utc"] >= test_start].copy()
    # Return all three partitions because downstream model training depends on this structure.
    return {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }


# Build a compact split summary because partition coverage and sample size must be documented explicitly.
def summarize_split(dataframe: pd.DataFrame, horizon_label: str) -> Dict:
    # Build the horizon-specific target column name because each horizon uses a distinct future target.
    target_column = f"load_target_{horizon_label}"
    # Return the split size and time coverage so partition validity can be checked quickly.
    return {
        "row_count": int(len(dataframe)),
        "start_utc": str(dataframe["timestamp_utc"].min()) if not dataframe.empty else None,
        "end_utc": str(dataframe["timestamp_utc"].max()) if not dataframe.empty else None,
        "target_missing_row_count": int(dataframe[target_column].isna().sum()) if not dataframe.empty else 0,
        "wind_missing_row_count": int(dataframe["wind_onshore_mw"].isna().sum()) if not dataframe.empty else 0,
    }


# Execute the horizon split workflow because each forecast task now requires its own chronological partitions.
def main() -> int:
    # Configure logging before any file or dataframe work begins.
    configure_logging()
    try:
        # Ensure the root split directory exists before writing artifacts.
        ensure_directory(SPLIT_DIR)
        # Initialize metadata storage because each horizon output should be documented explicitly.
        all_horizon_metadata = []
        # Iterate through the agreed project horizons because each one is a distinct supervised task.
        for horizon_spec in get_horizon_specifications():
            # Extract the horizon label because it controls file names and target-column selection.
            horizon_label = horizon_spec["label"]
            # Define the input path so the correct horizon dataset is loaded.
            input_path = HORIZON_DIR / f"ireland_load_{horizon_label}.csv"
            # Define the output subdirectory so horizon-specific split files remain isolated and clear.
            horizon_output_dir = SPLIT_DIR / horizon_label
            # Ensure the horizon-specific output directory exists before writing files.
            ensure_directory(horizon_output_dir)
            # Load the horizon dataset because chronological partitioning depends on it.
            horizon_df = read_horizon_dataset(input_path)
            # Remove rows without future targets because supervised evaluation requires observed future load.
            horizon_df = filter_available_target_rows(horizon_df, horizon_label=horizon_label)
            # Build the train, validation, and test partitions for this horizon.
            split_frames = build_time_splits(horizon_df)
            # Define output paths because each partition should be persisted independently.
            train_output_path = horizon_output_dir / "train.csv"
            # Define the validation output path because model selection depends on it.
            validation_output_path = horizon_output_dir / "validation.csv"
            # Define the test output path because final evaluation depends on it.
            test_output_path = horizon_output_dir / "test.csv"
            # Write the training partition because downstream models will fit on this subset.
            split_frames["train"].to_csv(train_output_path, index=False)
            # Write the validation partition because downstream model selection will use this subset.
            split_frames["validation"].to_csv(validation_output_path, index=False)
            # Write the test partition because downstream final evaluation will use this subset.
            split_frames["test"].to_csv(test_output_path, index=False)
            # Build the metadata summary for this horizon because split quality should remain auditable.
            horizon_metadata = {
                "horizon_label": horizon_label,
                "lead_hours": horizon_spec["lead_hours"],
                "input_path": str(input_path),
                "train_output_path": str(train_output_path),
                "validation_output_path": str(validation_output_path),
                "test_output_path": str(test_output_path),
                "summaries": {
                    "train": summarize_split(split_frames["train"], horizon_label=horizon_label),
                    "validation": summarize_split(split_frames["validation"], horizon_label=horizon_label),
                    "test": summarize_split(split_frames["test"], horizon_label=horizon_label),
                },
            }
            # Append the summary so the full run can be reviewed after completion.
            all_horizon_metadata.append(horizon_metadata)
            # Log completion of the current horizon because progress visibility matters.
            LOGGER.info("Built chronological splits for horizon: %s", horizon_label)
        # Define the metadata output path because this corrective step should preserve its own lineage.
        metadata_output_path = SPLIT_DIR / "horizon_split_metadata.json"
        # Persist the metadata because downstream modeling should reference a durable split contract.
        write_json(
            metadata_output_path,
            {
                "generated_at_utc": utc_now_iso(),
                "split_policy": "Chronological split only; no random sampling.",
                "boundaries": {
                    "train_end_utc": "2024-09-30 23:00:00+00:00",
                    "validation_start_utc": "2024-10-01 00:00:00+00:00",
                    "validation_end_utc": "2024-11-15 23:00:00+00:00",
                    "test_start_utc": "2024-11-16 00:00:00+00:00",
                },
                "note": "These splits are built from explicit t+1, t+24, and t+168 supervised horizon datasets.",
                "horizons": all_horizon_metadata,
            },
        )
        # Log successful completion so the operator can distinguish a full run from a partial one.
        LOGGER.info("Horizon split build completed successfully.")
        # Print a concise summary so the operator can validate the correction immediately.
        print(
            json.dumps(
                {
                    "metadata_output_path": str(metadata_output_path),
                    "horizons": all_horizon_metadata,
                },
                indent=2,
            )
        )
        # Return success because the workflow completed without exceptions.
        return 0
    except Exception as exc:
        # Log the full exception because silent split failures would compromise project validity.
        LOGGER.exception("Horizon split build failed: %s", exc)
        # Return failure so terminal execution can detect the issue.
        return 1


# Execute the workflow only when the script is run directly so imports remain side-effect free.
if __name__ == "__main__":
    # Exit with the workflow status so the shell receives an accurate process code.
    sys.exit(main())