# Import json so the audit summary can be persisted as structured metadata.
import json

# Import logging so the audit process emits operational diagnostics.
import logging

# Import sys so the script can return explicit process exit codes.
import sys

# Import datetime utilities so expected time ranges can be created deterministically.
from datetime import datetime
from datetime import timezone

# Import Path so file-system operations remain portable and deterministic.
from pathlib import Path

# Import typing helpers so interfaces remain explicit and low-risk.
from typing import Dict
from typing import List

# Import pandas so hourly coverage and gap structure can be audited efficiently.
import pandas as pd


# Create a module-level logger so every audit stage emits traceable diagnostics.
LOGGER = logging.getLogger("entsoe_quality_audit")


# Store the processed directory because this audit operates on Step 3 outputs only.
PROCESSED_DIR = Path("data/processed/entsoe")


# Store the audit output directory so diagnostics remain separate from modeling assets.
AUDIT_DIR = Path("data/audit/entsoe")


# Configure logging once so terminal output remains structured and readable.
def configure_logging() -> None:
    # Initialize logging with timestamped structured formatting for reproducibility.
    logging.basicConfig(
        # Use INFO because audit lifecycle events should be visible without excessive verbosity.
        level=logging.INFO,
        # Use a structured format so logs remain readable in terminal and logs.
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# Ensure a directory exists before writing files so failures reflect logic issues, not missing folders.
def ensure_directory(path: Path) -> None:
    # Create the directory tree idempotently because repeated audit runs are expected.
    path.mkdir(parents=True, exist_ok=True)


# Return an ISO UTC timestamp so audit metadata remains temporally explicit.
def utc_now_iso() -> str:
    # Generate the current UTC timestamp because audit lineage must be preserved.
    return datetime.now(timezone.utc).isoformat()


# Read a CSV file because the processed artifacts are stored as flat tabular files.
def read_csv(path: Path) -> pd.DataFrame:
    # Load the CSV into a dataframe so audit logic can inspect coverage and null structure.
    dataframe = pd.read_csv(path)
    # Fail explicitly if the dataset is empty because there would be nothing meaningful to audit.
    if dataframe.empty:
        # Raise a value error so the operator can inspect the upstream preprocessing failure.
        raise ValueError(f"Processed dataset is empty: {path}")
    # Return the loaded dataframe so downstream audit logic can operate on it.
    return dataframe


# Build the expected 2024 hourly UTC index because the project currently targets calendar year 2024.
def build_expected_hourly_index() -> pd.DatetimeIndex:
    # Create the full leap-year hourly index so observed coverage can be compared against expectation.
    return pd.date_range(
        start="2024-01-01 00:00:00+00:00",
        end="2024-12-31 23:00:00+00:00",
        freq="h",
        tz="UTC",
    )


# Audit the hourly load series because the target variable must be continuous and well-understood before modeling.
def audit_load_hourly(load_df: pd.DataFrame) -> Dict:
    # Work on a copy so the caller's dataframe remains unchanged.
    working_df = load_df.copy()
    # Parse timestamps as timezone-aware UTC values because coverage checks require consistent chronology.
    working_df["timestamp_utc"] = pd.to_datetime(working_df["timestamp_utc"], utc=True)
    # Sort by timestamp so missing-range analysis operates deterministically.
    working_df.sort_values("timestamp_utc", inplace=True)
    # Create the expected full-hour index for the project year.
    expected_index = build_expected_hourly_index()
    # Build the observed timestamp index from the processed dataset.
    observed_index = pd.DatetimeIndex(working_df["timestamp_utc"])
    # Compute missing timestamps relative to the expected complete calendar-year series.
    missing_timestamps = expected_index.difference(observed_index)
    # Compute timestamps that exist but whose load values are null because both structural and value gaps matter.
    null_value_timestamps = working_df.loc[working_df["load_mw"].isna(), "timestamp_utc"].tolist()
    # Compute duplicate timestamp count because duplicate targets would distort modeling later.
    duplicate_count = int(working_df["timestamp_utc"].duplicated().sum())
    # Return a structured audit summary because the result must be persisted and reviewed.
    return {
        "expected_hour_count": int(len(expected_index)),
        "observed_hour_count": int(len(working_df)),
        "missing_timestamp_count": int(len(missing_timestamps)),
        "missing_timestamps": [str(timestamp) for timestamp in missing_timestamps],
        "null_value_count": int(working_df["load_mw"].isna().sum()),
        "null_value_timestamps": [str(timestamp) for timestamp in null_value_timestamps],
        "duplicate_timestamp_count": duplicate_count,
        "observed_start_utc": str(working_df["timestamp_utc"].min()),
        "observed_end_utc": str(working_df["timestamp_utc"].max()),
    }


# Audit the generation subtype coverage because wind extraction depends on actual returned psr_type values.
def audit_generation_hourly(generation_df: pd.DataFrame) -> Dict:
    # Work on a copy so the original dataframe remains unchanged.
    working_df = generation_df.copy()
    # Parse timestamps as UTC because subtype coverage is still time-dependent.
    working_df["timestamp_utc"] = pd.to_datetime(working_df["timestamp_utc"], utc=True)
    # Sort rows so later inspection exports remain deterministic.
    working_df.sort_values(["psr_type", "timestamp_utc"], inplace=True)
    # Aggregate subtype statistics because the next preprocessing step needs subtype-level visibility.
    subtype_summary = (
        working_df.groupby("psr_type", dropna=False)
        .agg(
            row_count=("generation_mw", "size"),
            missing_values=("generation_mw", lambda series: int(series.isna().sum())),
            min_generation_mw=("generation_mw", "min"),
            max_generation_mw=("generation_mw", "max"),
        )
        .reset_index()
    )
    # Return both a compact summary and the subtype inventory because wind mapping depends on inspection.
    return {
        "observed_start_utc": str(working_df["timestamp_utc"].min()),
        "observed_end_utc": str(working_df["timestamp_utc"].max()),
        "row_count": int(len(working_df)),
        "unique_psr_types": sorted(working_df["psr_type"].dropna().unique().tolist()),
        "subtype_summary_records": subtype_summary.to_dict(orient="records"),
    }


# Write JSON because the audit summary should be preserved for pipeline governance.
def write_json(path: Path, payload: Dict) -> None:
    # Open the destination file in UTF-8 mode so serialization remains portable.
    with path.open("w", encoding="utf-8") as handle:
        # Write formatted JSON so the audit report stays machine-readable and reviewable.
        json.dump(payload, handle, indent=2)


# Execute the audit workflow because Step 3 outputs must be validated before feature engineering.
def main() -> int:
    # Configure logging before any I/O or dataframe processing begins.
    configure_logging()
    try:
        # Ensure the audit directory exists before writing any artifacts.
        ensure_directory(AUDIT_DIR)
        # Define the processed load path because it is the target series requiring quality validation.
        load_path = PROCESSED_DIR / "ireland_load_hourly.csv"
        # Define the processed generation path because subtype inspection is required before wind extraction.
        generation_path = PROCESSED_DIR / "ireland_generation_per_type_hourly.csv"
        # Load the processed hourly load dataset so coverage and null structure can be assessed.
        load_df = read_csv(load_path)
        # Load the processed hourly generation dataset so subtype inventory can be assessed.
        generation_df = read_csv(generation_path)
        # Audit the load series because forecasting should not proceed on an unvalidated target timeline.
        load_audit = audit_load_hourly(load_df)
        # Audit the generation series because wind selection depends on actual subtype evidence.
        generation_audit = audit_generation_hourly(generation_df)
        # Write missing-load timestamps to CSV so manual inspection remains simple.
        missing_load_output = AUDIT_DIR / "load_missing_timestamps.csv"
        # Convert missing timestamp strings into a dataframe for easy inspection in Excel or pandas.
        pd.DataFrame({"timestamp_utc": load_audit["missing_timestamps"]}).to_csv(missing_load_output, index=False)
        # Write generation subtype summary to CSV so subtype mapping can be reviewed directly.
        generation_subtype_output = AUDIT_DIR / "generation_psr_type_summary.csv"
        # Persist the subtype summary because wind mapping depends on inspecting actual subtype characteristics.
        pd.DataFrame(generation_audit["subtype_summary_records"]).to_csv(generation_subtype_output, index=False)
        # Build the main audit metadata payload so all findings are stored together.
        audit_payload = {
            "generated_at_utc": utc_now_iso(),
            "load_audit": load_audit,
            "generation_audit": generation_audit,
            "artifacts": {
                "missing_load_timestamps_csv": str(missing_load_output),
                "generation_psr_type_summary_csv": str(generation_subtype_output),
            },
            "note": "Do not impute or extract wind until missing-load structure and psr_type semantics are reviewed.",
        }
        # Define the audit JSON output path because the summary should persist beyond terminal output.
        audit_output_path = AUDIT_DIR / "entsoe_quality_audit.json"
        # Persist the audit summary because it is a decision artifact for the next project step.
        write_json(audit_output_path, audit_payload)
        # Log successful completion so the operator knows the audit stage finished cleanly.
        LOGGER.info("ENTSO-E quality audit completed successfully.")
        # Print a concise summary so the operator can validate the critical findings immediately.
        print(
            json.dumps(
                {
                    "audit_output_path": str(audit_output_path),
                    "missing_load_timestamp_count": load_audit["missing_timestamp_count"],
                    "null_load_value_count": load_audit["null_value_count"],
                    "unique_psr_types": generation_audit["unique_psr_types"],
                    "generation_subtype_summary_csv": str(generation_subtype_output),
                },
                indent=2,
            )
        )
        # Return success because the audit step completed without exceptions.
        return 0
    except Exception as exc:
        # Log the full exception because silent audit failures would weaken pipeline governance.
        LOGGER.exception("ENTSO-E quality audit failed: %s", exc)
        # Return failure so terminal invocation can detect the issue.
        return 1


# Execute the audit workflow only when the script is run directly so imports remain side-effect free.
if __name__ == "__main__":
    # Exit with the workflow status so the shell receives an accurate process code.
    sys.exit(main())