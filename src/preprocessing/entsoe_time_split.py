# Import json so split metadata can be persisted for reproducibility and later review.
import json

# Import logging so the split process emits operational diagnostics.
import logging

# Import sys so the script can return explicit process exit codes.
import sys

# Import datetime so metadata timestamps remain timezone-aware and standardized.
from datetime import datetime
from datetime import timezone

# Import Path so file-system operations remain portable and deterministic.
from pathlib import Path

# Import typing helpers so interfaces remain explicit and lower-risk.
from typing import Dict

# Import pandas so the modeling table can be filtered into chronological partitions.
import pandas as pd


# Create a module-level logger so all split-stage diagnostics use one consistent channel.
LOGGER = logging.getLogger("entsoe_time_split")


# Store the modeling directory because this step consumes the completed modeling table artifact.
MODELING_DIR = Path("data/modeling/entsoe")


# Store the split output directory so partitioned datasets remain separate from the master modeling table.
SPLIT_DIR = Path("data/splits/entsoe")


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


# Return an ISO-8601 UTC timestamp so metadata remains temporally explicit across runs.
def utc_now_iso() -> str:
    # Generate the current UTC timestamp because split lineage should be preserved.
    return datetime.now(timezone.utc).isoformat()


# Read a CSV file because the modeling table is persisted as a flat tabular artifact.
def read_csv(path: Path) -> pd.DataFrame:
    # Load the CSV into a dataframe so chronological filtering can be applied.
    dataframe = pd.read_csv(path)
    # Fail explicitly if the dataset is empty because no valid split can be produced from an empty table.
    if dataframe.empty:
        # Raise a value error so the operator can inspect the upstream modeling-table build.
        raise ValueError(f"Input dataset is empty: {path}")
    # Return the loaded dataframe so downstream split logic can operate on it.
    return dataframe


# Write a JSON file because split metadata must remain machine-readable and auditable.
def write_json(path: Path, payload: Dict) -> None:
    # Open the destination file in UTF-8 mode so serialization remains portable.
    with path.open("w", encoding="utf-8") as handle:
        # Persist formatted JSON so the metadata is easy to inspect manually.
        json.dump(payload, handle, indent=2)


# Parse timestamps and sort the modeling table because chronological integrity is mandatory for time-series splits.
def prepare_modeling_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy so the source dataframe remains unchanged for debugging and lineage.
    result = dataframe.copy()
    # Parse timestamps as UTC because all split boundaries must be evaluated in one coherent timezone.
    result["timestamp_utc"] = pd.to_datetime(result["timestamp_utc"], utc=True)
    # Sort chronologically because time-based splitting must operate on ordered observations.
    result.sort_values("timestamp_utc", inplace=True)
    # Reset the index so exported split files remain clean and sequential.
    result.reset_index(drop=True, inplace=True)
    # Return the prepared dataframe because it is now ready for deterministic slicing.
    return result


# Remove rows with missing target values because baseline supervised models cannot train on undefined outcomes.
def filter_trainable_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    # Keep only rows where the target is available because training, validation, and testing require observed load.
    result = dataframe.loc[dataframe["target_available_for_training"] == 1].copy()
    # Reset the index so each split exports as a clean standalone dataset.
    result.reset_index(drop=True, inplace=True)
    # Return the filtered dataframe because it forms the basis of the supervised partitions.
    return result


# Apply the fixed chronological split because the project requires time-based partitioning only.
def build_time_splits(dataframe: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    # Define the train end boundary because the first nine months provide the main learning window.
    train_end = pd.Timestamp("2024-09-30 23:00:00+00:00")
    # Define the validation start boundary because validation should immediately follow train chronologically.
    validation_start = pd.Timestamp("2024-10-01 00:00:00+00:00")
    # Define the validation end boundary because the validation window should be long enough for model comparison.
    validation_end = pd.Timestamp("2024-11-15 23:00:00+00:00")
    # Define the test start boundary because the final segment should remain untouched during tuning.
    test_start = pd.Timestamp("2024-11-16 00:00:00+00:00")
    # Build the training partition using only the earliest observations.
    train_df = dataframe.loc[dataframe["timestamp_utc"] <= train_end].copy()
    # Build the validation partition using the middle chronological window only.
    validation_df = dataframe.loc[
        (dataframe["timestamp_utc"] >= validation_start) & (dataframe["timestamp_utc"] <= validation_end)
    ].copy()
    # Build the test partition using only the final untouched chronological window.
    test_df = dataframe.loc[dataframe["timestamp_utc"] >= test_start].copy()
    # Return all three partitions because downstream baseline modeling depends on this structure.
    return {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }


# Build a compact split summary because partition coverage and sample size must be documented explicitly.
def summarize_split(dataframe: pd.DataFrame) -> Dict:
    # Return the split size and time coverage so partition validity can be checked quickly.
    return {
        "row_count": int(len(dataframe)),
        "start_utc": str(dataframe["timestamp_utc"].min()) if not dataframe.empty else None,
        "end_utc": str(dataframe["timestamp_utc"].max()) if not dataframe.empty else None,
        "target_missing_row_count": int(dataframe["load_mw"].isna().sum()) if not dataframe.empty else 0,
        "wind_missing_row_count": int(dataframe["wind_onshore_mw"].isna().sum()) if not dataframe.empty else 0,
    }


# Execute the split workflow because baseline model training requires fixed chronological partitions.
def main() -> int:
    # Configure logging before any file or dataframe work begins.
    configure_logging()
    try:
        # Ensure the split output directory exists before writing artifacts.
        ensure_directory(SPLIT_DIR)
        # Define the modeling-table input path because this step consumes the completed row-level feature table.
        modeling_input_path = MODELING_DIR / "ireland_load_modeling_table.csv"
        # Load the modeling table because chronological partitioning depends on the integrated feature dataset.
        modeling_df = read_csv(modeling_input_path)
        # Prepare the modeling table because timestamp parsing and sorting are required before splitting.
        modeling_df = prepare_modeling_table(modeling_df)
        # Remove rows without target values because supervised evaluation requires observed outcomes.
        trainable_df = filter_trainable_rows(modeling_df)
        # Build the chronological train, validation, and test partitions.
        split_frames = build_time_splits(trainable_df)
        # Define the split output file paths because each partition should be persisted independently.
        train_output_path = SPLIT_DIR / "train.csv"
        # Define the validation output path because model selection depends on a dedicated validation set.
        validation_output_path = SPLIT_DIR / "validation.csv"
        # Define the test output path because final evaluation must remain isolated.
        test_output_path = SPLIT_DIR / "test.csv"
        # Define the metadata output path because split lineage and coverage should be documented.
        metadata_output_path = SPLIT_DIR / "split_metadata.json"
        # Write the training partition because baseline models will fit on this subset.
        split_frames["train"].to_csv(train_output_path, index=False)
        # Write the validation partition because model tuning and comparison will use this subset.
        split_frames["validation"].to_csv(validation_output_path, index=False)
        # Write the test partition because final untouched evaluation will use this subset.
        split_frames["test"].to_csv(test_output_path, index=False)
        # Build a structured metadata payload because partition quality and boundaries must be reviewable.
        metadata_payload = {
            "generated_at_utc": utc_now_iso(),
            "source_modeling_input_path": str(modeling_input_path),
            "split_policy": "Chronological split only; no random sampling.",
            "boundaries": {
                "train_end_utc": "2024-09-30 23:00:00+00:00",
                "validation_start_utc": "2024-10-01 00:00:00+00:00",
                "validation_end_utc": "2024-11-15 23:00:00+00:00",
                "test_start_utc": "2024-11-16 00:00:00+00:00",
            },
            "summaries": {
                "train": summarize_split(split_frames["train"]),
                "validation": summarize_split(split_frames["validation"]),
                "test": summarize_split(split_frames["test"]),
            },
            "note": "Rows with missing target values were excluded before splitting to preserve supervised evaluation integrity.",
        }
        # Persist the metadata because split provenance and coverage should remain auditable.
        write_json(metadata_output_path, metadata_payload)
        # Log successful completion so the operator can distinguish a full run from a partial one.
        LOGGER.info("ENTSO-E chronological split completed successfully.")
        # Print a concise summary so the operator can validate the split immediately.
        print(
            json.dumps(
                {
                    "train_output_path": str(train_output_path),
                    "validation_output_path": str(validation_output_path),
                    "test_output_path": str(test_output_path),
                    "metadata_output_path": str(metadata_output_path),
                    "summaries": metadata_payload["summaries"],
                },
                indent=2,
            )
        )
        # Return success because the split step completed without exceptions.
        return 0
    except Exception as exc:
        # Log the full exception because silent split failures would compromise downstream evaluation.
        LOGGER.exception("ENTSO-E chronological split failed: %s", exc)
        # Return failure so shells and automation can detect the issue.
        return 1


# Execute the workflow only when the script is run directly so imports remain side-effect free.
if __name__ == "__main__":
    # Exit with the workflow status so terminal execution receives an accurate process code.
    sys.exit(main())