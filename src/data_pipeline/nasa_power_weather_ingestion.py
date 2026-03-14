# Import json so raw metadata and API payloads can be persisted for auditability.
import json

# Import logging so the ingestion workflow emits traceable operational diagnostics.
import logging

# Import sys so the script can return explicit process exit codes.
import sys

# Import datetime utilities so file timestamps remain timezone-aware and reproducible.
from datetime import datetime
from datetime import timezone

# Import Path so file-system operations remain portable and deterministic.
from pathlib import Path

# Import typing helpers so interfaces remain explicit and safer to maintain.
from typing import Dict
from typing import List

# Import pandas so hourly weather data can be normalized into tabular form.
import pandas as pd

# Import requests so the NASA POWER API can be queried over HTTPS.
import requests


# Create a module-level logger so all diagnostics use one consistent channel.
LOGGER = logging.getLogger("nasa_power_weather_ingestion")


# Store the official NASA POWER hourly point endpoint so the request target remains centralized and auditable.
BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"


# Store the raw weather output directory so source artifacts remain separated from processed outputs.
RAW_DIR = Path("data/raw/weather")


# Store the processed weather output directory so normalized files remain separated from raw payloads.
PROCESSED_DIR = Path("data/processed/weather")


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
    # Create the full directory tree idempotently because repeated ingestion runs are expected.
    path.mkdir(parents=True, exist_ok=True)


# Return a filesystem-safe UTC token so run artifacts can be grouped deterministically.
def utc_now_token() -> str:
    # Generate a compact UTC timestamp because output filenames should remain sortable and collision-resistant.
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# Return an ISO-8601 UTC timestamp so metadata remains temporally explicit.
def utc_now_iso() -> str:
    # Generate the current UTC timestamp because lineage should be preserved explicitly.
    return datetime.now(timezone.utc).isoformat()


# Define the weather variable set because the project improvement step should remain targeted and parsimonious.
def get_weather_parameters() -> List[str]:
    # Return a compact set of meteorological variables relevant to electricity-demand forecasting.
    return [
        "T2M",
        "RH2M",
        "WS10M",
        "WD10M",
        "PS",
        "PRECTOTCORR",
        "ALLSKY_SFC_SW_DWN",
    ]


# Build the NASA POWER query parameters because the API expects point-query parameters in the request string.
def build_query_params() -> Dict[str, str]:
    # Return a bounded 2024 Dublin-area point request so the weather layer aligns with the current load study window.
    return {
        "parameters": ",".join(get_weather_parameters()),
        "community": "RE",
        "longitude": "-6.2603",
        "latitude": "53.3498",
        "start": "20240101",
        "end": "20241231",
        "format": "JSON",
        "time-standard": "UTC",
    }


# Execute the NASA POWER request because the project now needs weather covariates from an official source.
def fetch_weather_payload() -> Dict:
    # Build the query parameters once so the request remains explicit and reproducible.
    params = build_query_params()
    # Log the request boundary so acquisition activity is visible in the terminal.
    LOGGER.info("Requesting NASA POWER hourly weather data for Ireland proxy point.")
    # Execute the GET request with a bounded timeout because ingestion should not block indefinitely.
    response = requests.get(BASE_URL, params=params, timeout=60)
    # Raise immediately on HTTP errors because silent partial ingestion is unacceptable.
    response.raise_for_status()
    # Return both parsed payload and resolved URL because both are needed for normalization and metadata.
    return {
        "payload": response.json(),
        "resolved_url": response.url,
    }


# Normalize the nested NASA POWER parameter dictionary into a flat hourly dataframe.
def normalize_weather_payload(payload: Dict) -> pd.DataFrame:
    # Extract the parameter block because NASA POWER organizes hourly values by variable name.
    parameter_block = payload["properties"]["parameter"]
    # Capture parameter names so the normalized table preserves semantic identity.
    parameter_names = list(parameter_block.keys())
    # Select the first parameter as the canonical source of hour keys because all variables should align on time.
    first_parameter_name = parameter_names[0]
    # Extract the hour keys exactly as returned because ingestion should not invent timestamp semantics.
    hour_keys = list(parameter_block[first_parameter_name].keys())
    # Initialize row storage because the final weather table must be row-oriented.
    records = []
    # Iterate through each hour key because the normalized dataset should have one row per hour.
    for hour_key in hour_keys:
        # Start the row with the raw NASA hour key so original temporal encoding is preserved.
        row = {"nasa_hour_key": hour_key}
        # Populate each requested weather variable for the current hour.
        for parameter_name in parameter_names:
            # Read the parameter value by hour key because the response is nested by variable.
            row[parameter_name] = parameter_block[parameter_name].get(hour_key)
        # Append the completed row so the dataframe can be built after the loop.
        records.append(row)
    # Convert the accumulated rows into a dataframe because CSV export requires tabular structure.
    dataframe = pd.DataFrame(records)
    # Parse the raw hour key into a proper UTC timestamp because the next merge step requires aligned chronology.
    dataframe["timestamp_utc"] = pd.to_datetime(dataframe["nasa_hour_key"], format="%Y%m%d%H", utc=True)
    # Sort chronologically because later merging with load data requires ordered timestamps.
    dataframe.sort_values("timestamp_utc", inplace=True)
    # Reset the index so the exported CSV remains clean and sequential.
    dataframe.reset_index(drop=True, inplace=True)
    # Rename variables to explicit business-friendly feature names so downstream modeling code remains readable.
    dataframe.rename(
        columns={
            "T2M": "temp_2m_c",
            "RH2M": "rel_humidity_2m_pct",
            "WS10M": "wind_speed_10m_ms",
            "WD10M": "wind_direction_10m_deg",
            "PS": "surface_pressure_kpa",
            "PRECTOTCORR": "precipitation_mm_hr",
            "ALLSKY_SFC_SW_DWN": "allsky_surface_solar_downward_wm2",
        },
        inplace=True,
    )
    # Create a stable precipitation column even if the API returns a different schema than expected.
    if "precipitation_mm_hr" not in dataframe.columns:
        # Add the column explicitly as missing so downstream merge logic stays schema-stable and auditable.
        dataframe["precipitation_mm_hr"] = pd.NA
    # Return the normalized dataframe because it becomes the processed weather table for later merging.
    return dataframe


# Write JSON because raw payloads and metadata should remain machine-readable and auditable.
def write_json(path: Path, payload: Dict) -> None:
    # Open the destination file in UTF-8 mode so serialization remains portable.
    with path.open("w", encoding="utf-8") as handle:
        # Persist formatted JSON so the output remains readable and reviewable.
        json.dump(payload, handle, indent=2)


# Execute the full weather ingestion workflow because feature-space improvement starts with official weather data.
def main() -> int:
    # Configure logging before any file or network work begins.
    configure_logging()
    try:
        # Ensure raw and processed weather directories exist before writing artifacts.
        ensure_directory(RAW_DIR)
        # Ensure the processed directory exists for normalized outputs.
        ensure_directory(PROCESSED_DIR)
        # Create a run token so all artifacts from this execution remain grouped deterministically.
        run_token = utc_now_token()
        # Fetch the NASA POWER payload because the project now needs hourly weather covariates.
        weather_response = fetch_weather_payload()
        # Extract the parsed payload because it will be written raw and normalized.
        payload = weather_response["payload"]
        # Normalize the payload into one row per hour because the next pipeline step requires tabular weather data.
        weather_df = normalize_weather_payload(payload)
        # Define the raw payload output path because source preservation is mandatory.
        raw_payload_path = RAW_DIR / f"nasa_power_weather_{run_token}.json"
        # Define the processed CSV output path because later merging should use flat files.
        processed_csv_path = PROCESSED_DIR / "nasa_power_weather_hourly.csv"
        # Define the metadata output path because acquisition lineage should remain auditable.
        metadata_path = PROCESSED_DIR / "nasa_power_weather_metadata.json"
        # Persist the raw payload because raw source snapshots should always be kept before transformation.
        write_json(raw_payload_path, payload)
        # Persist the normalized weather table because the next merge step requires a stable processed artifact.
        weather_df.to_csv(processed_csv_path, index=False)
        # Build metadata because source URL, parameters, and row counts should be documented explicitly.
        metadata = {
            "generated_at_utc": utc_now_iso(),
            "source_system": "NASA POWER",
            "resolved_url": weather_response["resolved_url"],
            "raw_payload_path": str(raw_payload_path),
            "processed_csv_path": str(processed_csv_path),
            "row_count": int(len(weather_df)),
            "start_utc": str(weather_df["timestamp_utc"].min()),
            "end_utc": str(weather_df["timestamp_utc"].max()),
            "parameters": get_weather_parameters(),
            "available_columns": weather_df.columns.tolist(),
            "note": "Single-point Dublin proxy weather ingestion for later merge with Irish load forecasting table.",
        }
        # Persist metadata because the weather layer should be reproducible and reviewable.
        write_json(metadata_path, metadata)
        # Log successful completion so the operator can distinguish a full run from a partial one.
        LOGGER.info("NASA POWER weather ingestion completed successfully.")
        # Print a concise summary so the operator can validate the result immediately.
        print(
            json.dumps(
                {
                    "raw_payload_path": str(raw_payload_path),
                    "processed_csv_path": str(processed_csv_path),
                    "metadata_path": str(metadata_path),
                    "row_count": int(len(weather_df)),
                    "start_utc": str(weather_df["timestamp_utc"].min()),
                    "end_utc": str(weather_df["timestamp_utc"].max()),
                    "available_columns": weather_df.columns.tolist(),
                },
                indent=2,
            )
        )
        # Return success because the workflow completed without exceptions.
        return 0
    except Exception as exc:
        # Log the full exception because silent ingestion failures would compromise downstream modeling.
        LOGGER.exception("NASA POWER weather ingestion failed: %s", exc)
        # Return failure so terminal execution can detect the issue.
        return 1


# Execute the workflow only when the script is run directly so imports remain side-effect free.
if __name__ == "__main__":
    # Exit with the workflow status so the shell receives an accurate process code.
    sys.exit(main())