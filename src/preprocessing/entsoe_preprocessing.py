# Import json so input metadata can be read and output metadata can be written for reproducibility.
import json

# Import logging so preprocessing operations emit traceable diagnostics.
import logging

# Import re so token redaction can be applied deterministically to stored URLs.
import re

# Import sys so the script can return explicit process exit codes.
import sys

# Import datetime so timestamps can be parsed and normalized safely.
from datetime import datetime
from datetime import timezone

# Import Path so file-system paths remain portable and deterministic.
from pathlib import Path

# Import typing helpers so interfaces are explicit and less error-prone.
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

# Import pandas so interval data can be reconstructed and resampled cleanly.
import pandas as pd


# Create a module-level logger so preprocessing diagnostics are emitted consistently.
LOGGER = logging.getLogger("entsoe_preprocessing")


# Store the expected raw directory so the preprocessing step reads only Step 2 artifacts.
RAW_DIR = Path("data/raw/entsoe")


# Store the processed directory so analysis-ready outputs remain separated from raw artifacts.
PROCESSED_DIR = Path("data/processed/entsoe")


# Configure logging once so terminal diagnostics remain structured and readable.
def configure_logging() -> None:
    # Initialize root logging with timestamped structured output for debugging and auditability.
    logging.basicConfig(
        # Use INFO because lifecycle events and quality checks should be visible without excessive noise.
        level=logging.INFO,
        # Use a structured format so logs are readable in terminal and saved records.
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# Ensure a directory exists before writing files so failures reflect logic or data issues, not missing folders.
def ensure_directory(path: Path) -> None:
    # Create the full directory tree idempotently because repeated preprocessing runs are expected.
    path.mkdir(parents=True, exist_ok=True)


# Return an ISO-8601 UTC timestamp so metadata remains unambiguous across systems.
def utc_now_iso() -> str:
    # Generate the current UTC time because pipeline lineage must be timestamped consistently.
    return datetime.now(timezone.utc).isoformat()


# Read a JSON file because metadata and audit artifacts are stored in JSON format.
def read_json(path: Path) -> Dict:
    # Open the JSON file in UTF-8 mode so content remains portable across environments.
    with path.open("r", encoding="utf-8") as handle:
        # Parse and return the JSON object because downstream steps require structured metadata access.
        return json.load(handle)


# Write a JSON file because processed-run metadata must be preserved for reproducibility.
def write_json(path: Path, payload: Dict) -> None:
    # Open the destination path in UTF-8 mode so serialization remains portable.
    with path.open("w", encoding="utf-8") as handle:
        # Persist formatted JSON so the metadata remains machine-readable and human-reviewable.
        json.dump(payload, handle, indent=2)


# Identify the latest Step 2 metadata file so preprocessing uses the most recent raw ingestion artifacts.
def find_latest_run_metadata(raw_dir: Path) -> Path:
    # Collect all run metadata files because Step 2 names them consistently by timestamp token.
    candidates = sorted(raw_dir.glob("entsoe_run_metadata_*.json"))
    # Fail explicitly when no metadata exists because preprocessing requires a completed ingestion run.
    if not candidates:
        # Raise a file error so the operator knows Step 2 artifacts are missing.
        raise FileNotFoundError("No ENTSO-E run metadata file found in data/raw/entsoe.")
    # Return the most recent metadata file because that is the natural default for the next pipeline step.
    return candidates[-1]


# Redact a security token from a URL because raw query strings should not expose credentials in metadata.
def redact_security_token(url: Optional[str]) -> Optional[str]:
    # Return early when the input is empty because there is nothing to sanitize.
    if url is None:
        # Preserve the null-like state because fabrication would weaken metadata integrity.
        return None
    # Replace the token value while preserving the rest of the URL for provenance.
    return re.sub(r"(securityToken=)[^&]+", r"\1REDACTED", url)


# Parse an ENTSO-E resolution token into pandas-compatible minute length.
def resolution_to_minutes(resolution: str) -> int:
    # Fail when resolution is missing because timestamp reconstruction depends on interval length.
    if not resolution:
        # Raise a precise error because silent fallback would corrupt time alignment.
        raise ValueError("Missing resolution value; cannot reconstruct timestamps.")
    # Map common ISO-8601 duration tokens used by ENTSO-E to minute counts.
    resolution_map = {
        "PT15M": 15,
        "PT30M": 30,
        "PT60M": 60,
        "PT1H": 60,
    }
    # Fail clearly on unhandled resolutions because the pipeline must not invent interval semantics.
    if resolution not in resolution_map:
        # Raise an error so unsupported interval widths are explicitly investigated.
        raise ValueError(f"Unsupported ENTSO-E resolution encountered: {resolution}")
    # Return the mapped minute count so timestamp offsets can be computed deterministically.
    return resolution_map[resolution]


# Parse an ENTSO-E UTC timestamp string into a timezone-aware pandas timestamp.
def parse_period_start(period_start: str) -> pd.Timestamp:
    # Fail when the period start is missing because all point timestamps derive from this anchor.
    if not period_start:
        # Raise a hard error because there is no valid fallback for a missing anchor time.
        raise ValueError("Missing period_start value; cannot reconstruct timestamps.")
    # Parse the timestamp as UTC because ENTSO-E payloads use UTC semantics in the XML time intervals.
    return pd.Timestamp(period_start, tz="UTC")


# Reconstruct the true timestamp for each row from period start, resolution, and position.
def reconstruct_timestamp(period_start: str, resolution: str, position: int) -> pd.Timestamp:
    # Parse the interval anchor time because point positions are defined relative to it.
    start_timestamp = parse_period_start(period_start)
    # Convert the ENTSO-E interval resolution to minutes so arithmetic remains explicit and auditable.
    resolution_minutes = resolution_to_minutes(resolution)
    # Fail when position is missing because the point offset cannot otherwise be reconstructed.
    if pd.isna(position):
        # Raise a precise error because silent imputation would corrupt chronology.
        raise ValueError("Missing position value; cannot reconstruct timestamps.")
    # Convert the one-based ENTSO-E position into a zero-based offset because position 1 represents the first interval.
    zero_based_offset = int(position) - 1
    # Return the final timestamp by adding the appropriate interval offset to the period start.
    return start_timestamp + pd.Timedelta(minutes=zero_based_offset * resolution_minutes)


# Load one Step 2 CSV artifact because preprocessing operates on normalized raw CSV rather than XML.
def load_raw_dataset(csv_path: Path) -> pd.DataFrame:
    # Read the CSV into a dataframe so timestamp reconstruction and QC can be applied.
    dataframe = pd.read_csv(csv_path)
    # Fail explicitly if the dataframe is empty because no valid downstream time series can be produced.
    if dataframe.empty:
        # Raise a value error so the operator can inspect the raw query result.
        raise ValueError(f"Raw dataset is empty: {csv_path}")
    # Return the loaded dataframe so preprocessing can continue.
    return dataframe


# Add reconstructed timestamps to the raw dataframe because Step 3 requires a real chronological index.
def add_reconstructed_timestamps(dataframe: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy so the input dataframe remains unchanged for safer debugging and reuse.
    result = dataframe.copy()
    # Reconstruct the timestamp for every row because the raw CSV stores only relative time information.
    result["timestamp_utc"] = result.apply(
        # Apply row-wise because timestamp reconstruction depends on three columns per row.
        lambda row: reconstruct_timestamp(
            period_start=row["period_start"],
            resolution=row["resolution"],
            position=row["position"],
        ),
        axis=1,
    )
    # Return the dataframe with the new timestamp column so later steps can aggregate by time.
    return result


# Standardize load data to one hourly series because the forecast target must be one row per hour.
def build_hourly_load_table(load_df: pd.DataFrame) -> pd.DataFrame:
    # Reconstruct timestamps because the raw dataset is position-based rather than clock-based.
    working_df = add_reconstructed_timestamps(load_df)
    # Keep only the columns required for the target series so the processed artifact remains lean.
    working_df = working_df[["timestamp_utc", "quantity"]].copy()
    # Rename the quantity column to an explicit business name so downstream modeling code remains readable.
    working_df.rename(columns={"quantity": "load_mw"}, inplace=True)
    # Sort chronologically because resampling and duplicate handling require deterministic ordering.
    working_df.sort_values("timestamp_utc", inplace=True)
    # Group by timestamp and average duplicate entries conservatively because multiple overlapping series may exist.
    working_df = working_df.groupby("timestamp_utc", as_index=False)["load_mw"].mean()
    # Set the timestamp as the index because hourly resampling requires a datetime index.
    working_df.set_index("timestamp_utc", inplace=True)
    # Resample to hourly means so quarter-hour intervals are standardized to the project frequency.
    hourly_df = working_df.resample("1H").mean()
    # Restore timestamp as a normal column because flat CSV export is needed for downstream steps.
    hourly_df.reset_index(inplace=True)
    # Return the hourly load table because it is the processed target series for the project.
    return hourly_df


# Standardize generation-per-type data to hourly form while preserving subtype information for later feature selection.
def build_hourly_generation_table(generation_df: pd.DataFrame) -> pd.DataFrame:
    # Reconstruct timestamps because generation rows are also indexed by relative interval positions.
    working_df = add_reconstructed_timestamps(generation_df)
    # Keep only the fields required for subtype-aware aggregation so the processed table stays focused.
    working_df = working_df[["timestamp_utc", "psr_type", "business_type", "quantity"]].copy()
    # Rename quantity to generation_mw so the column meaning remains explicit in later joins.
    working_df.rename(columns={"quantity": "generation_mw"}, inplace=True)
    # Fill missing subtype values with a stable sentinel because missing labels should remain explicit, not fabricated.
    working_df["psr_type"] = working_df["psr_type"].fillna("UNKNOWN")
    # Sort chronologically because grouped resampling should operate on ordered timestamps.
    working_df.sort_values("timestamp_utc", inplace=True)
    # Group by timestamp and subtype because generation-per-type is inherently multi-series.
    grouped_df = (
        working_df.groupby(["timestamp_utc", "psr_type", "business_type"], as_index=False)["generation_mw"]
        .mean()
    )
    # Set timestamp as the index so subtype-specific hourly resampling becomes deterministic.
    grouped_df.set_index("timestamp_utc", inplace=True)
    # Resample to hourly mean within each subtype because project frequency is hourly.
    hourly_df = (
        grouped_df.groupby(["psr_type", "business_type"])
        .resample("1H")
        .mean(numeric_only=True)
        .reset_index()
    )
    # Return the hourly generation table because wind filtering can be done safely after subtype inspection.
    return hourly_df


# Produce a compact data-quality summary because Step 3 should verify chronology and coverage before modeling.
def build_quality_summary(load_hourly_df: pd.DataFrame, generation_hourly_df: pd.DataFrame) -> Dict:
    # Compute the observed load start timestamp so coverage can be validated against the query window.
    load_start = load_hourly_df["timestamp_utc"].min()
    # Compute the observed load end timestamp so coverage can be validated against the query window.
    load_end = load_hourly_df["timestamp_utc"].max()
    # Compute the total load rows because this is the primary target series cardinality.
    load_rows = int(len(load_hourly_df))
    # Compute missing load values because target completeness directly affects forecasting validity.
    load_missing = int(load_hourly_df["load_mw"].isna().sum())
    # Compute generation start timestamp so exogenous feature coverage can be validated.
    generation_start = generation_hourly_df["timestamp_utc"].min()
    # Compute generation end timestamp so exogenous feature coverage can be validated.
    generation_end = generation_hourly_df["timestamp_utc"].max()
    # Compute total generation rows because the processed subtype table may be large and needs explicit tracking.
    generation_rows = int(len(generation_hourly_df))
    # Count unique generation subtype labels because wind extraction depends on inspecting actual returned classes.
    generation_unique_psr_types = sorted(generation_hourly_df["psr_type"].dropna().unique().tolist())
    # Return the assembled quality summary because it will be stored as processed-run metadata.
    return {
        "load_hourly_start_utc": str(load_start) if pd.notna(load_start) else None,
        "load_hourly_end_utc": str(load_end) if pd.notna(load_end) else None,
        "load_hourly_row_count": load_rows,
        "load_hourly_missing_values": load_missing,
        "generation_hourly_start_utc": str(generation_start) if pd.notna(generation_start) else None,
        "generation_hourly_end_utc": str(generation_end) if pd.notna(generation_end) else None,
        "generation_hourly_row_count": generation_rows,
        "generation_unique_psr_types": generation_unique_psr_types,
    }


# Resolve the latest dataset paths from the Step 2 metadata because file names are timestamped by run token.
def extract_dataset_paths(metadata: Dict) -> Tuple[Path, Path]:
    # Initialize placeholders so both required datasets can be identified explicitly.
    load_csv_path = None
    # Initialize the generation path placeholder because Step 3 requires both raw datasets.
    generation_csv_path = None
    # Iterate through dataset entries because Step 2 stores each artifact path in the metadata file.
    for dataset_entry in metadata.get("datasets", []):
        # Capture the load CSV path when the load dataset entry is encountered.
        if dataset_entry.get("dataset_name") == "ireland_load":
            load_csv_path = Path(dataset_entry["normalized_csv_path"])
        # Capture the generation CSV path when the generation dataset entry is encountered.
        if dataset_entry.get("dataset_name") == "ireland_generation_per_type":
            generation_csv_path = Path(dataset_entry["normalized_csv_path"])
    # Fail explicitly when either required input is missing because preprocessing depends on both artifacts.
    if load_csv_path is None or generation_csv_path is None:
        # Raise an error so the operator knows the Step 2 metadata is incomplete or corrupted.
        raise FileNotFoundError("Could not resolve both Ireland load and generation CSV paths from Step 2 metadata.")
    # Return both resolved paths so preprocessing can load the correct raw inputs.
    return load_csv_path, generation_csv_path


# Execute Step 3 preprocessing because the project now needs true hourly chronological tables.
def main() -> int:
    # Configure logging before any file or dataframe operations begin.
    configure_logging()
    try:
        # Ensure the processed output directory exists before any artifacts are written.
        ensure_directory(PROCESSED_DIR)
        # Locate the most recent Step 2 run metadata so the latest ingestion artifacts are processed.
        latest_metadata_path = find_latest_run_metadata(RAW_DIR)
        # Read the metadata file because it contains the raw CSV paths produced in Step 2.
        raw_metadata = read_json(latest_metadata_path)
        # Resolve the raw dataset CSV paths from the Step 2 metadata structure.
        load_csv_path, generation_csv_path = extract_dataset_paths(raw_metadata)
        # Log the resolved paths so the operator can confirm which raw artifacts are being processed.
        LOGGER.info("Using load CSV: %s", load_csv_path)
        # Log the generation CSV path so both raw inputs are visible in the run log.
        LOGGER.info("Using generation CSV: %s", generation_csv_path)
        # Load the raw load dataset because Step 3 starts from the normalized raw CSV artifact.
        load_raw_df = load_raw_dataset(load_csv_path)
        # Load the raw generation dataset because exogenous preprocessing also begins from the raw CSV artifact.
        generation_raw_df = load_raw_dataset(generation_csv_path)
        # Build the hourly load table because the project target must be standardized to hourly frequency.
        load_hourly_df = build_hourly_load_table(load_raw_df)
        # Build the hourly generation-per-type table because subtype-aware exogenous features are needed later.
        generation_hourly_df = build_hourly_generation_table(generation_raw_df)
        # Create output paths for the processed load artifact because processed and raw assets must remain separate.
        load_output_path = PROCESSED_DIR / "ireland_load_hourly.csv"
        # Create output paths for the processed generation artifact because later wind inspection depends on it.
        generation_output_path = PROCESSED_DIR / "ireland_generation_per_type_hourly.csv"
        # Create the processed metadata path so lineage and QC results are persisted.
        metadata_output_path = PROCESSED_DIR / "entsoe_preprocessing_metadata.json"
        # Write the processed hourly load table because it is the main target series for forecasting.
        load_hourly_df.to_csv(load_output_path, index=False)
        # Write the processed hourly generation table because it is the base exogenous series table.
        generation_hourly_df.to_csv(generation_output_path, index=False)
        # Build a quality summary because Step 3 must verify the resulting coverage and subtype inventory.
        quality_summary = build_quality_summary(load_hourly_df, generation_hourly_df)
        # Persist preprocessing metadata with redacted source URLs because secrets must not leak into audit files.
        write_json(
            metadata_output_path,
            {
                "generated_at_utc": utc_now_iso(),
                "source_step2_metadata_path": str(latest_metadata_path),
                "source_load_csv_path": str(load_csv_path),
                "source_generation_csv_path": str(generation_csv_path),
                "redacted_datasets": [
                    {
                        "dataset_name": dataset_entry.get("dataset_name"),
                        "document_type": dataset_entry.get("document_type"),
                        "resolved_url": redact_security_token(dataset_entry.get("resolved_url")),
                    }
                    for dataset_entry in raw_metadata.get("datasets", [])
                ],
                "quality_summary": quality_summary,
                "note": "Wind subtype selection is intentionally deferred until actual psr_type labels are inspected.",
            },
        )
        # Log successful completion so the operator can distinguish a completed preprocessing run from a partial one.
        LOGGER.info("ENTSO-E preprocessing completed successfully.")
        # Print a concise summary so the operator can validate row counts and output locations immediately.
        print(
            json.dumps(
                {
                    "load_output_path": str(load_output_path),
                    "generation_output_path": str(generation_output_path),
                    "metadata_output_path": str(metadata_output_path),
                    "quality_summary": quality_summary,
                },
                indent=2,
            )
        )
        # Return success because Step 3 preprocessing completed without exceptions.
        return 0
    except Exception as exc:
        # Log the full exception because silent preprocessing failures would compromise model validity.
        LOGGER.exception("ENTSO-E preprocessing failed: %s", exc)
        # Return a non-zero status so shells and automation can detect the failure.
        return 1


# Execute the preprocessing workflow only when the script is run directly so imports remain side-effect free.
if __name__ == "__main__":
    # Exit with the returned status code so terminal invocation receives an accurate result.
    sys.exit(main())