# Import json so run metadata can be persisted for auditability and later review.
import json

# Import logging so the merge workflow emits operational diagnostics.
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
from typing import List

# Import pandas so hourly datasets can be merged and quality-checked safely.
import pandas as pd


# Create a module-level logger so all diagnostics use one consistent channel.
LOGGER = logging.getLogger("entsoe_weather_augmented_modeling_table")


# Store the existing modeling-table directory because the current ENTSO-E master table already exists there.
MODELING_DIR = Path("data/modeling/entsoe")


# Store the processed weather directory because the normalized NASA POWER file lives there.
WEATHER_DIR = Path("data/processed/weather")


# Store the output directory so the weather-augmented artifact remains clearly separated from the prior version.
OUTPUT_DIR = Path("data/modeling_weather/entsoe")


# Configure logging once so terminal diagnostics remain structured and readable.
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


# Return an ISO-8601 UTC timestamp so metadata remains temporally explicit.
def utc_now_iso() -> str:
    # Generate the current UTC timestamp because lineage should be preserved explicitly.
    return datetime.now(timezone.utc).isoformat()


# Read a CSV file because all upstream artifacts are persisted as flat tabular files.
def read_csv(path: Path) -> pd.DataFrame:
    # Load the CSV into a dataframe so merge and quality-check operations can be applied.
    dataframe = pd.read_csv(path)
    # Fail explicitly if the dataset is empty because no valid merge can be built from it.
    if dataframe.empty:
        # Raise a value error so the operator can inspect the upstream artifact.
        raise ValueError(f"Input dataset is empty: {path}")
    # Return the loaded dataframe because downstream steps require structured tabular access.
    return dataframe


# Write a JSON file because build metadata should remain machine-readable and auditable.
def write_json(path: Path, payload: Dict) -> None:
    # Open the destination file in UTF-8 mode so serialization remains portable.
    with path.open("w", encoding="utf-8") as handle:
        # Persist formatted JSON so the metadata remains readable and reviewable.
        json.dump(payload, handle, indent=2)


# Parse and standardize timestamps because all merges must occur on one coherent temporal index.
def standardize_timestamp_column(dataframe: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    # Work on a copy so the source dataframe remains unchanged for debugging and lineage.
    result = dataframe.copy()
    # Parse timestamps as UTC because the project contract uses a unified hourly UTC chronology.
    result[timestamp_column] = pd.to_datetime(result[timestamp_column], utc=True)
    # Sort chronologically because deterministic merge behavior depends on ordered observations.
    result.sort_values(timestamp_column, inplace=True)
    # Reset the index so downstream row operations remain clean and predictable.
    result.reset_index(drop=True, inplace=True)
    # Return the standardized dataframe because it is now suitable for merge operations.
    return result


# Return the expected weather feature columns because the model table should keep only operationally relevant inputs.
def get_weather_feature_columns() -> List[str]:
    # Return the normalized weather columns created during NASA POWER ingestion.
    return [
        "temp_2m_c",
        "rel_humidity_2m_pct",
        "wind_speed_10m_ms",
        "wind_direction_10m_deg",
        "surface_pressure_kpa",
        "precipitation_mm_hr",
        "allsky_surface_solar_downward_wm2",
    ]


# Add weather missingness flags because explicit data-quality traceability is required for modeling governance.
def add_weather_missing_flags(dataframe: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy so the caller's dataframe remains unchanged.
    result = dataframe.copy()
    # Iterate through weather features because each should preserve its own missingness signal.
    for column_name in get_weather_feature_columns():
        # Add a raw missingness flag so later modeling or audit steps can inspect source completeness explicitly.
        result[f"{column_name}_missing_flag_raw"] = result[column_name].isna().astype(int)
    # Return the enriched dataframe because explicit missingness flags improve auditability.
    return result


# Apply conservative weather filling because short exogenous gaps can be interpolated without hiding target gaps.
def fill_weather_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy so pre-fill values remain available in the caller if needed.
    result = dataframe.copy()
    # Set timestamp as index because time interpolation requires an ordered datetime index.
    result.set_index("timestamp_utc", inplace=True)
    # Iterate through weather features because each numeric weather column should be gap-treated consistently.
    for column_name in get_weather_feature_columns():
        # Apply time interpolation because weather varies continuously and short hourly gaps can be filled conservatively.
        result[column_name] = result[column_name].interpolate(
            method="time",
            limit=6,
            limit_direction="both",
        )
    # Restore timestamp as a normal column because the downstream artifact should remain a flat CSV table.
    result.reset_index(inplace=True)
    # Return the filled dataframe because it is ready for post-fill missingness checks.
    return result


# Add post-fill weather missingness flags because unresolved exogenous gaps should remain explicit after treatment.
def add_weather_postfill_flags(dataframe: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy so the caller's dataframe remains unchanged.
    result = dataframe.copy()
    # Iterate through weather features because unresolved missingness should be tracked per variable.
    for column_name in get_weather_feature_columns():
        # Add a post-fill missingness flag so remaining exogenous gaps remain visible to later model code.
        result[f"{column_name}_missing_flag_postfill"] = result[column_name].isna().astype(int)
    # Return the enriched dataframe because it preserves post-treatment gap visibility.
    return result


# Merge weather into the current modeling table because this is the core improvement step.
def merge_modeling_with_weather(modeling_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    # Keep only timestamp and weather features because the weather file contains no other modeling-critical columns.
    weather_subset = weather_df[["timestamp_utc"] + get_weather_feature_columns()].copy()
    # Left-join weather onto the existing modeling table because the ENTSO-E table remains the canonical spine.
    merged_df = modeling_df.merge(weather_subset, on="timestamp_utc", how="left")
    # Add raw weather missingness flags because merge completeness should be auditable.
    merged_df = add_weather_missing_flags(merged_df)
    # Apply conservative weather filling because short exogenous gaps should not unnecessarily shrink training data.
    merged_df = fill_weather_features(merged_df)
    # Add post-fill weather missingness flags because unresolved gaps should remain explicit for later modeling.
    merged_df = add_weather_postfill_flags(merged_df)
    # Return the weather-augmented modeling table because it is the new master artifact for downstream tasks.
    return merged_df


# Build a compact quality summary because the merged table should be validated before horizon regeneration.
def build_quality_summary(dataframe: pd.DataFrame) -> Dict:
    # Count total rows because the augmented master table should preserve full hourly coverage.
    row_count = int(len(dataframe))
    # Capture temporal coverage because chronology should remain unchanged after the merge.
    start_utc = str(dataframe["timestamp_utc"].min())
    # Capture the end timestamp for the same reason.
    end_utc = str(dataframe["timestamp_utc"].max())
    # Count missing targets because the merge should not alter target availability.
    target_missing_row_count = int(dataframe["load_mw"].isna().sum())
    # Count unresolved wind gaps because the prior exogenous feature should remain visible.
    wind_missing_row_count = int(dataframe["wind_onshore_mw"].isna().sum())
    # Initialize weather quality storage because each covariate should be audited individually.
    weather_quality = {}
    # Iterate through weather features because each requires explicit missingness accounting.
    for column_name in get_weather_feature_columns():
        # Record raw and post-fill missingness because improvement should be transparent rather than assumed.
        weather_quality[column_name] = {
            "raw_missing_count": int(dataframe[f"{column_name}_missing_flag_raw"].sum()),
            "postfill_missing_count": int(dataframe[f"{column_name}_missing_flag_postfill"].sum()),
        }
    # Return the summary because it becomes part of the build metadata.
    return {
        "row_count": row_count,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "target_missing_row_count": target_missing_row_count,
        "wind_missing_row_count": wind_missing_row_count,
        "weather_quality": weather_quality,
    }


# Execute the weather-augmentation workflow because the next model comparison should use richer covariates.
def main() -> int:
    # Configure logging before any file or dataframe work begins.
    configure_logging()
    try:
        # Ensure the output directory exists before writing artifacts.
        ensure_directory(OUTPUT_DIR)
        # Define the existing master modeling-table path because this is the current canonical ENTSO-E feature table.
        modeling_input_path = MODELING_DIR / "ireland_load_modeling_table.csv"
        # Define the processed weather input path because this is the normalized NASA POWER artifact.
        weather_input_path = WEATHER_DIR / "nasa_power_weather_hourly.csv"
        # Load the current master modeling table because weather will be merged into it.
        modeling_df = read_csv(modeling_input_path)
        # Load the processed weather table because it is the new exogenous improvement layer.
        weather_df = read_csv(weather_input_path)
        # Standardize timestamps in the modeling table because deterministic merge behavior depends on coherent chronology.
        modeling_df = standardize_timestamp_column(modeling_df, "timestamp_utc")
        # Standardize timestamps in the weather table for the same reason.
        weather_df = standardize_timestamp_column(weather_df, "timestamp_utc")
        # Merge weather into the modeling table because this is the core objective of the step.
        augmented_df = merge_modeling_with_weather(modeling_df, weather_df)
        # Define the output CSV path because downstream horizon regeneration should consume this new master table.
        output_csv_path = OUTPUT_DIR / "ireland_load_modeling_table_with_weather.csv"
        # Define the metadata path because the new master artifact should preserve its own lineage and QC summary.
        metadata_output_path = OUTPUT_DIR / "entsoe_weather_augmented_modeling_table_metadata.json"
        # Persist the weather-augmented modeling table because downstream steps must consume a stable artifact.
        augmented_df.to_csv(output_csv_path, index=False)
        # Build the quality summary because the merged table must be validated before further modeling.
        quality_summary = build_quality_summary(augmented_df)
        # Persist metadata because the merge lineage and covariate completeness must remain auditable.
        write_json(
            metadata_output_path,
            {
                "generated_at_utc": utc_now_iso(),
                "source_modeling_input_path": str(modeling_input_path),
                "source_weather_input_path": str(weather_input_path),
                "weather_feature_columns": get_weather_feature_columns(),
                "quality_summary": quality_summary,
                "note": "This artifact extends the prior ENTSO-E modeling table with NASA POWER hourly weather covariates.",
            },
        )
        # Log successful completion so the operator can distinguish a full run from a partial one.
        LOGGER.info("Weather-augmented modeling table build completed successfully.")
        # Print a concise summary so the operator can validate the output immediately.
        print(
            json.dumps(
                {
                    "output_csv_path": str(output_csv_path),
                    "metadata_output_path": str(metadata_output_path),
                    "quality_summary": quality_summary,
                },
                indent=2,
            )
        )
        # Return success because the workflow completed without exceptions.
        return 0
    except Exception as exc:
        # Log the full exception because silent merge failures would compromise downstream comparisons.
        LOGGER.exception("Weather-augmented modeling table build failed: %s", exc)
        # Return failure so terminal execution can detect the issue.
        return 1


# Execute the workflow only when the script is run directly so imports remain side-effect free.
if __name__ == "__main__":
    # Exit with the workflow status so the shell receives an accurate process code.
    sys.exit(main())