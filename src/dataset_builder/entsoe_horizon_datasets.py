# Import json so dataset-build metadata can be persisted for auditability and later review.
import json

# Import logging so the horizon dataset build emits operational diagnostics.
import logging

# Import sys so the script can return explicit process exit codes.
import sys

# Import datetime so metadata timestamps remain timezone-aware and standardized.
from datetime import datetime
from datetime import timezone

# Import Path so filesystem operations remain portable and deterministic.
from pathlib import Path

# Import typing helpers so interfaces remain explicit and safer to maintain.
from typing import Dict
from typing import List

# Import pandas so the modeling table can be transformed into horizon-specific supervised datasets.
import pandas as pd


# Create a module-level logger so all diagnostics use one consistent channel.
LOGGER = logging.getLogger("entsoe_horizon_datasets")


# Store the modeling directory because this step consumes the master modeling table artifact.
MODELING_DIR = Path("data/modeling/entsoe")


# Store the horizon output directory so horizon-specific datasets remain separate from the master table.
HORIZON_DIR = Path("data/horizons/entsoe")


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
    # Create the full directory tree idempotently because repeated pipeline runs are expected.
    path.mkdir(parents=True, exist_ok=True)


# Return an ISO-8601 UTC timestamp so metadata remains temporally explicit.
def utc_now_iso() -> str:
    # Generate the current UTC timestamp because lineage should be preserved explicitly.
    return datetime.now(timezone.utc).isoformat()


# Read a CSV file because the master modeling table is stored as a flat tabular artifact.
def read_csv(path: Path) -> pd.DataFrame:
    # Load the CSV into a dataframe so horizon-specific targets can be constructed.
    dataframe = pd.read_csv(path)
    # Fail explicitly if the dataset is empty because no valid horizon datasets can be built from it.
    if dataframe.empty:
        # Raise a value error so the operator can inspect the upstream modeling-table build.
        raise ValueError(f"Input dataset is empty: {path}")
    # Parse timestamps as UTC because all horizon calculations depend on coherent temporal ordering.
    dataframe["timestamp_utc"] = pd.to_datetime(dataframe["timestamp_utc"], utc=True)
    # Sort chronologically because target shifting must operate on a time-ordered table.
    dataframe.sort_values("timestamp_utc", inplace=True)
    # Reset the index so downstream row operations remain deterministic.
    dataframe.reset_index(drop=True, inplace=True)
    # Return the prepared dataframe because it is now suitable for target shifting.
    return dataframe


# Write a JSON file because build metadata should remain machine-readable and auditable.
def write_json(path: Path, payload: Dict) -> None:
    # Open the destination file in UTF-8 mode so serialization remains portable.
    with path.open("w", encoding="utf-8") as handle:
        # Persist formatted JSON so the metadata remains readable and reviewable.
        json.dump(payload, handle, indent=2)


# Define the forecast horizons because the project contract explicitly requires these three tasks.
def get_horizon_specifications() -> List[Dict]:
    # Return the agreed project horizons so all downstream datasets remain contract-aligned.
    return [
        {"label": "t_plus_1", "lead_hours": 1},
        {"label": "t_plus_24", "lead_hours": 24},
        {"label": "t_plus_168", "lead_hours": 168},
    ]


# Build one horizon-specific supervised dataset because each lead time is a distinct forecasting task.
def build_horizon_dataset(dataframe: pd.DataFrame, lead_hours: int, label: str) -> pd.DataFrame:
    # Work on a copy so the source modeling table remains unchanged for debugging and lineage.
    result = dataframe.copy()
    # Shift the load target backward so each row's predictors map to the future load at the required horizon.
    result[f"load_target_{label}"] = result["load_mw"].shift(-lead_hours)
    # Shift the future timestamp backward so each row records the exact target timestamp it is forecasting.
    result[f"target_timestamp_{label}"] = result["timestamp_utc"].shift(-lead_hours)
    # Create a target-availability flag because forecast rows without future outcomes must be excluded later.
    result[f"target_available_{label}"] = result[f"load_target_{label}"].notna().astype(int)
    # Return the horizon-specific dataset because it is now a proper supervised forecasting table for that lead.
    return result


# Summarize one horizon dataset because row counts and target availability should be documented explicitly.
def summarize_horizon_dataset(dataframe: pd.DataFrame, label: str) -> Dict:
    # Build the target column name once so the same contract is used consistently.
    target_column = f"load_target_{label}"
    # Build the target-availability flag name once so the same contract is used consistently.
    target_flag_column = f"target_available_{label}"
    # Return a compact summary because build outputs should be reviewable without opening the full CSV.
    return {
        "row_count": int(len(dataframe)),
        "available_target_rows": int(dataframe[target_flag_column].sum()),
        "missing_target_rows": int(dataframe[target_column].isna().sum()),
        "start_utc": str(dataframe["timestamp_utc"].min()),
        "end_utc": str(dataframe["timestamp_utc"].max()),
    }


# Execute the horizon dataset build because the project now needs task-correct supervised datasets.
def main() -> int:
    # Configure logging before any file or dataframe work begins.
    configure_logging()
    try:
        # Ensure the horizon output directory exists before writing artifacts.
        ensure_directory(HORIZON_DIR)
        # Define the master modeling-table path because it is the input to all horizon builds.
        modeling_input_path = MODELING_DIR / "ireland_load_modeling_table.csv"
        # Load the master modeling table because horizon-specific target construction depends on it.
        modeling_df = read_csv(modeling_input_path)
        # Initialize metadata storage because each horizon output should be documented explicitly.
        horizon_metadata = []
        # Iterate through the agreed project horizons because each one is a distinct forecasting task.
        for horizon_spec in get_horizon_specifications():
            # Extract the horizon label because it becomes part of output file naming and column naming.
            label = horizon_spec["label"]
            # Extract the lead time because target shifting depends on the number of forecast hours.
            lead_hours = horizon_spec["lead_hours"]
            # Build the horizon-specific supervised dataset because this is the core corrective step.
            horizon_df = build_horizon_dataset(modeling_df, lead_hours=lead_hours, label=label)
            # Define the output path so each horizon task is persisted independently.
            output_path = HORIZON_DIR / f"ireland_load_{label}.csv"
            # Write the dataset because downstream splitting and modeling must operate on explicit forecast tasks.
            horizon_df.to_csv(output_path, index=False)
            # Summarize the output because availability and missingness should remain auditable.
            summary = summarize_horizon_dataset(horizon_df, label=label)
            # Append metadata so the full build can be reviewed after completion.
            horizon_metadata.append(
                {
                    "horizon_label": label,
                    "lead_hours": lead_hours,
                    "output_path": str(output_path),
                    "summary": summary,
                }
            )
            # Log completion of the current horizon because progress visibility matters in iterative pipelines.
            LOGGER.info("Built horizon dataset: %s", output_path)
        # Define the metadata output path because the corrective step should preserve its own lineage.
        metadata_output_path = HORIZON_DIR / "horizon_dataset_metadata.json"
        # Persist the metadata because downstream steps should reference a durable horizon contract.
        write_json(
            metadata_output_path,
            {
                "generated_at_utc": utc_now_iso(),
                "source_modeling_input_path": str(modeling_input_path),
                "note": "These datasets correct the earlier task-framing drift by explicitly encoding t+1, t+24, and t+168 forecast targets.",
                "horizons": horizon_metadata,
            },
        )
        # Log successful completion so the operator can distinguish a full run from a partial one.
        LOGGER.info("Horizon dataset build completed successfully.")
        # Print a concise summary so the operator can validate the correction immediately.
        print(
            json.dumps(
                {
                    "metadata_output_path": str(metadata_output_path),
                    "horizons": horizon_metadata,
                },
                indent=2,
            )
        )
        # Return success because the corrective workflow completed without exceptions.
        return 0
    except Exception as exc:
        # Log the full exception because silent task-framing failures would compromise project validity.
        LOGGER.exception("Horizon dataset build failed: %s", exc)
        # Return failure so terminal execution can detect the issue.
        return 1


# Execute the workflow only when the script is run directly so imports remain side-effect free.
if __name__ == "__main__":
    # Exit with the workflow status so the shell receives an accurate process code.
    sys.exit(main())