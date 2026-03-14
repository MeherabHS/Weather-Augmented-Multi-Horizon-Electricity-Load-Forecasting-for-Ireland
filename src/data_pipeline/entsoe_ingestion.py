# Import json so run metadata can be persisted in a machine-readable audit format.
import json

# Import logging so the ingestion process emits traceable operational diagnostics.
import logging

# Import os so the API token can be read securely from environment variables.
import os

# Import sys so the script can return explicit process exit codes.
import sys

# Import dataclass so request configuration remains structured without over-engineering.
from dataclasses import dataclass

# Import datetime utilities so query windows and file timestamps remain timezone-aware.
from datetime import datetime
from datetime import timezone

# Import Path so file-system operations remain portable and deterministic.
from pathlib import Path

# Import typing helpers so interfaces are explicit and safer to maintain.
from typing import Dict
from typing import List
from typing import Optional

# Import pandas so parsed time-series records can be materialized as CSV tables.
import pandas as pd

# Import requests so the ENTSO-E API can be queried over HTTPS.
import requests

# Import ElementTree so XML responses can be parsed without adding unnecessary dependencies.
import xml.etree.ElementTree as ET


# Create a module-level logger so all functions write to a consistent diagnostic channel.
LOGGER = logging.getLogger("entsoe_ingestion")


# Store the official ENTSO-E API endpoint so the request target is centralized and auditable.
BASE_URL = "https://web-api.tp.entsoe.eu/api"


# Store the Ireland bidding-zone EIC code because the project scope is fixed to Ireland.
IRELAND_DOMAIN = "10YIE-1001A00010"


# Store the default date window because Step 2 should produce a bounded first raw snapshot.
DEFAULT_PERIOD_START = "202401010000"


# Store the default end timestamp because the first ingestion pass targets calendar year 2024.
DEFAULT_PERIOD_END = "202412312300"


# Define a lightweight request configuration because the script only needs a small stable contract.
@dataclass
class QueryConfig:
    # Store the ENTSO-E document type so each call remains semantically explicit.
    document_type: str
    # Store the ENTSO-E process type because the project needs realized historical data.
    process_type: str
    # Store the area parameter name because load and generation do not use the same domain field.
    area_param_name: str
    # Store the area code so the request can target the Irish bidding zone.
    area_code: str
    # Store the period start in ENTSO-E format so the query remains API-compliant.
    period_start: str
    # Store the period end in ENTSO-E format so the query remains API-compliant.
    period_end: str
    # Store a dataset label so output artifacts are self-describing.
    dataset_name: str


# Configure logging once so terminal output remains uniform across runs.
def configure_logging() -> None:
    # Initialize root logging with a structured timestamped format for reproducibility and debugging.
    logging.basicConfig(
        # Use INFO because ingestion lifecycle visibility is needed without excessive verbosity.
        level=logging.INFO,
        # Use a structured format so logs remain readable in both terminal and saved outputs.
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# Ensure a directory exists before writing into it so failures reflect source issues, not local path issues.
def ensure_directory(path: Path) -> None:
    # Create the full directory tree idempotently because repeated runs are normal in data acquisition.
    path.mkdir(parents=True, exist_ok=True)


# Create a UTC timestamp string so output files can be grouped by run consistently.
def utc_now_token() -> str:
    # Return a filesystem-safe UTC timestamp because output artifact naming must be deterministic.
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# Create an ISO-8601 UTC timestamp so metadata remains unambiguous across environments.
def utc_now_iso() -> str:
    # Return an ISO timestamp because it is both machine-readable and audit-friendly.
    return datetime.now(timezone.utc).isoformat()


# Read the ENTSO-E token from the environment because secrets should not be hardcoded into source files.
def get_api_token() -> str:
    # Read the environment variable that will hold the user's ENTSO-E token.
    token = os.getenv("ENTSOE_API_TOKEN")
    # Fail explicitly if the token is missing because silent fallback would be operationally unsafe.
    if not token:
        # Raise a clear error so the operator knows exactly how to fix the invocation.
        raise ValueError(
            "Missing ENTSOE_API_TOKEN environment variable. "
            "Set the token in your shell before running the script."
        )
    # Return the validated token because the script can now authenticate requests.
    return token


# Build the query parameter dictionary because the API expects query-string based requests.
def build_query_params(token: str, config: QueryConfig) -> Dict[str, str]:
    # Start with the parameters common to both load and generation requests.
    params = {
        "securityToken": token,
        "documentType": config.document_type,
        "processType": config.process_type,
        "periodStart": config.period_start,
        "periodEnd": config.period_end,
    }
    # Add the area parameter using the API-specific field name required by the dataset type.
    params[config.area_param_name] = config.area_code
    # Return the full parameter dictionary so the caller can execute the request.
    return params


# Execute one ENTSO-E API request and return the raw XML payload plus the resolved request URL.
def fetch_entsoe_xml(token: str, config: QueryConfig, timeout_seconds: int = 120) -> Dict[str, str]:
    # Build the query string parameters from the provided request configuration.
    params = build_query_params(token, config)
    # Log the request boundary so the operator can see which dataset is being pulled.
    LOGGER.info("Requesting ENTSO-E dataset '%s' with documentType=%s", config.dataset_name, config.document_type)
    # Perform the HTTPS GET request because the ENTSO-E API is query-string driven.
    response = requests.get(BASE_URL, params=params, timeout=timeout_seconds)
    # Raise immediately on HTTP errors because partial ingestion must not pass silently.
    response.raise_for_status()
    # Return both content and resolved URL because both are useful for metadata logging.
    return {
        "xml_text": response.text,
        "resolved_url": response.url,
    }


# Strip XML namespaces from parsed tags because ENTSO-E payloads are namespace-qualified.
def strip_namespace(tag: str) -> str:
    # Remove the namespace prefix when present so downstream matching remains simple and robust.
    return tag.split("}", 1)[-1] if "}" in tag else tag


# Find the first direct child element by local tag name because namespace prefixes can vary.
def find_child_text(parent: ET.Element, child_name: str) -> Optional[str]:
    # Iterate direct children so lookup remains resilient to namespace-qualified tags.
    for child in list(parent):
        # Compare on local name only because XML namespace URIs are not stable for manual string matching.
        if strip_namespace(child.tag) == child_name:
            # Return normalized text when available so downstream parsing remains predictable.
            return child.text.strip() if child.text is not None else None
    # Return None when the child does not exist because missing fields should not be fabricated.
    return None


# Extract all TimeSeries elements regardless of namespace because ENTSO-E embeds them in XML documents.
def find_timeseries_elements(root: ET.Element) -> List[ET.Element]:
    # Initialize storage so all matching time-series blocks can be returned to the caller.
    timeseries_elements = []
    # Walk the full tree because TimeSeries can appear under nested containers.
    for element in root.iter():
        # Keep only TimeSeries nodes because those contain the actual chronological records.
        if strip_namespace(element.tag) == "TimeSeries":
            # Append the matching element so it can be parsed later into tabular rows.
            timeseries_elements.append(element)
    # Return all collected time-series blocks for parsing.
    return timeseries_elements


# Parse ENTSO-E Period blocks into row-wise records while preserving series-level metadata.
def parse_timeseries_rows(xml_text: str, dataset_name: str) -> pd.DataFrame:
    # Parse the XML document into an element tree because the API response format is XML.
    root = ET.fromstring(xml_text)
    # Find all TimeSeries blocks because each one may represent a separate subtype or business slice.
    timeseries_list = find_timeseries_elements(root)
    # Initialize row storage because the final output must be a flat dataframe.
    records: List[Dict] = []
    # Iterate through each TimeSeries because metadata can differ across series.
    for series_index, timeseries in enumerate(timeseries_list, start=1):
        # Read the business type because it is useful for later validation and subtype inspection.
        business_type = find_child_text(timeseries, "businessType")
        # Read the production subtype when present because generation-per-type depends on this information.
        psr_type = None
        # Iterate through descendants so nested MktPSRType blocks can be captured safely.
        for descendant in timeseries.iter():
            # Detect the production subtype tag independent of namespace formatting.
            if strip_namespace(descendant.tag) == "psrType":
                # Persist the subtype value for every point in the same series.
                psr_type = descendant.text.strip() if descendant.text is not None else None
        # Iterate through descendants again so Period blocks can be processed.
        for period in timeseries.iter():
            # Skip non-Period nodes because the chronological points live inside Period elements.
            if strip_namespace(period.tag) != "Period":
                continue
            # Read the period start string because point positions are relative to this timestamp.
            period_start_text = None
            # Read the resolution string because timestamp reconstruction depends on interval length.
            resolution_text = None
            # Inspect direct children of Period so start time and resolution can be extracted.
            for period_child in list(period):
                # Capture the nested timeInterval block because it carries the period start and end.
                if strip_namespace(period_child.tag) == "timeInterval":
                    # Inspect the timeInterval children to extract the start timestamp.
                    for interval_child in list(period_child):
                        # Persist only the start timestamp because points are indexed forward from it.
                        if strip_namespace(interval_child.tag) == "start":
                            period_start_text = interval_child.text.strip() if interval_child.text is not None else None
                # Capture the resolution because ENTSO-E point positions require interval interpretation.
                if strip_namespace(period_child.tag) == "resolution":
                    # Store the interval token exactly as returned so no unverified conversion is invented.
                    resolution_text = period_child.text.strip() if period_child.text is not None else None
            # Process each Point inside the Period because each point corresponds to one interval observation.
            for point in period.iter():
                # Skip all elements except Point because only those contain position and quantity.
                if strip_namespace(point.tag) != "Point":
                    continue
                # Read the relative position because ENTSO-E indexes points from the period start.
                position_text = find_child_text(point, "position")
                # Read the quantity because it is the actual numerical value of the observation.
                quantity_text = find_child_text(point, "quantity")
                # Append a flat row while preserving metadata needed for later timestamp reconstruction.
                records.append(
                    {
                        "dataset_name": dataset_name,
                        "series_index": series_index,
                        "business_type": business_type,
                        "psr_type": psr_type,
                        "period_start": period_start_text,
                        "resolution": resolution_text,
                        "position": int(position_text) if position_text is not None else None,
                        "quantity": float(quantity_text) if quantity_text is not None else None,
                    }
                )
    # Convert the collected records into a dataframe because downstream storage and QC expect tabular data.
    dataframe = pd.DataFrame(records)
    # Return the dataframe even if empty because explicit emptiness is safer than silent coercion.
    return dataframe


# Persist raw XML exactly as received because raw source snapshots are mandatory for provenance.
def write_raw_xml(output_path: Path, xml_text: str) -> None:
    # Open the destination in UTF-8 text mode because the payload is XML text.
    with output_path.open("w", encoding="utf-8") as handle:
        # Write the XML without transformation so the raw response remains intact.
        handle.write(xml_text)


# Persist JSON metadata because each run must record source lineage and acquisition context.
def write_metadata(output_path: Path, payload: Dict) -> None:
    # Open the target path in UTF-8 text mode so serialization remains portable.
    with output_path.open("w", encoding="utf-8") as handle:
        # Write readable JSON so the audit trail can be manually inspected later.
        json.dump(payload, handle, indent=2)


# Build the two request configurations needed for Step 2 because the project needs load and generation raw data.
def build_default_query_configs() -> List[QueryConfig]:
    # Return the load request and the generation-per-type request as a fixed ordered list.
    return [
        QueryConfig(
            document_type="A65",
            process_type="A16",
            area_param_name="outBiddingZone_Domain",
            area_code=IRELAND_DOMAIN,
            period_start=DEFAULT_PERIOD_START,
            period_end=DEFAULT_PERIOD_END,
            dataset_name="ireland_load",
        ),
        QueryConfig(
            document_type="A75",
            process_type="A16",
            area_param_name="in_Domain",
            area_code=IRELAND_DOMAIN,
            period_start=DEFAULT_PERIOD_START,
            period_end=DEFAULT_PERIOD_END,
            dataset_name="ireland_generation_per_type",
        ),
    ]


# Execute the full Step 2 ingestion run because the project needs one controlled entry point.
def main() -> int:
    # Configure logging before any network or file work begins.
    configure_logging()
    try:
        # Read the token from the environment because secrets must remain outside source control.
        token = get_api_token()
        # Define the output directory used by the ENTSO-E ingestion step.
        output_dir = Path("data/raw/entsoe")
        # Ensure the output directory exists before writing any artifacts.
        ensure_directory(output_dir)
        # Build the fixed query set for the Irish load forecasting project.
        query_configs = build_default_query_configs()
        # Create a run timestamp so all artifacts from the same execution are grouped deterministically.
        run_token = utc_now_token()
        # Initialize a metadata list so per-dataset lineage can be written after the loop.
        metadata_entries: List[Dict] = []
        # Iterate through each configured dataset because Step 2 ingests both load and generation raw data.
        for query_config in query_configs:
            # Fetch the raw XML payload and resolved request URL for the current dataset.
            response_payload = fetch_entsoe_xml(token, query_config)
            # Build the raw XML output path so source fidelity is preserved.
            raw_xml_path = output_dir / f"{query_config.dataset_name}_{run_token}.xml"
            # Build the normalized CSV output path so later preprocessing can operate on flat files.
            csv_output_path = output_dir / f"{query_config.dataset_name}_{run_token}.csv"
            # Write the raw XML snapshot before any parsing so provenance remains intact.
            write_raw_xml(raw_xml_path, response_payload["xml_text"])
            # Parse the XML into a flat dataframe while preserving series-level metadata.
            dataframe = parse_timeseries_rows(response_payload["xml_text"], query_config.dataset_name)
            # Write the normalized dataframe to CSV because downstream feature engineering will consume flat files.
            dataframe.to_csv(csv_output_path, index=False)
            # Append dataset metadata so the full run can be audited after completion.
            metadata_entries.append(
                {
                    "dataset_name": query_config.dataset_name,
                    "document_type": query_config.document_type,
                    "process_type": query_config.process_type,
                    "area_param_name": query_config.area_param_name,
                    "area_code": query_config.area_code,
                    "period_start": query_config.period_start,
                    "period_end": query_config.period_end,
                    "downloaded_at_utc": utc_now_iso(),
                    "resolved_url": response_payload["resolved_url"],
                    "raw_xml_path": str(raw_xml_path),
                    "normalized_csv_path": str(csv_output_path),
                    "row_count": int(len(dataframe)),
                    "note": "Timestamps are not yet reconstructed from period_start, resolution, and position; that is Step 3.",
                }
            )
            # Log completion of the current dataset so operational progress remains visible.
            LOGGER.info("Saved dataset '%s' to %s", query_config.dataset_name, csv_output_path)
        # Build the run-level metadata path so both dataset requests are documented together.
        metadata_path = output_dir / f"entsoe_run_metadata_{run_token}.json"
        # Persist the run metadata so lineage remains queryable and reviewable.
        write_metadata(metadata_path, {"run_generated_at_utc": utc_now_iso(), "datasets": metadata_entries})
        # Log successful completion so the operator can distinguish a full run from a partial one.
        LOGGER.info("ENTSO-E ingestion completed successfully.")
        # Print a concise terminal summary so the operator can validate the run immediately.
        print(json.dumps({"metadata_path": str(metadata_path), "datasets": metadata_entries}, indent=2))
        # Return success because the ingestion run completed without exceptions.
        return 0
    except Exception as exc:
        # Log the full exception because silent ingestion failures are unacceptable in research workflows.
        LOGGER.exception("ENTSO-E ingestion failed: %s", exc)
        # Return failure so shells and automation can detect the broken run.
        return 1


# Execute the main workflow only when the file is run directly so imports remain side-effect free.
if __name__ == "__main__":
    # Exit with the returned status code so terminal execution gets an accurate result.
    sys.exit(main())