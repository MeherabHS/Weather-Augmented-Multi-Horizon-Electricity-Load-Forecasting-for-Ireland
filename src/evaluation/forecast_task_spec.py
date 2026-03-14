# Import json to serialize the forecast task specification into a durable machine-readable artifact.
import json

# Import logging to preserve an execution trail for project governance and reproducibility.
import logging

# Import sys to return explicit process status codes for automation and validation.
import sys

# Import dataclass to keep the forecast specification structured and low-complexity.
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field

# Import datetime to timestamp the specification artifact for provenance control.
from datetime import datetime
from datetime import timezone

# Import Path to write the specification into a deterministic repository location.
from pathlib import Path

# Import typing helpers to make schema intent explicit and reduce ambiguity.
from typing import Dict
from typing import List


# Create a logger so specification generation remains auditable like later ingestion and modeling steps.
LOGGER = logging.getLogger("forecast_task_spec")


# Configure logging once so execution records are readable and consistent across environments.
def configure_logging() -> None:
    # Initialize logging with timestamped formatting because this project requires traceable artifacts.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# Create a compact horizon model because forecast targets must be explicitly governed, not implied.
@dataclass
class ForecastHorizon:
    # Store the horizon label to keep reports and downstream code semantically readable.
    label: str
    # Store the lead time in hours because this project operates on hourly aligned forecasting tasks.
    lead_hours: int
    # Store the operational interpretation so the task remains understandable in README and reports.
    business_meaning: str


# Create the core project specification schema because downstream scripts should inherit one source of truth.
@dataclass
class ForecastTaskSpecification:
    # Store the project name to bind the artifact to the correct research objective.
    project_name: str
    # Store the geographic scope so future benchmarking does not drift beyond Ireland unintentionally.
    geography: str
    # Store the target variable so all later transformations remain contractually aligned.
    target_variable: str
    # Store the target unit because model outputs and plots need a consistent quantitative scale.
    target_unit: str
    # Store the temporal resolution because ingestion and feature windows depend on a fixed cadence.
    temporal_resolution: str
    # Store the probabilistic requirement because this project is not a point-forecast-only exercise.
    probabilistic_output: Dict[str, object]
    # Store the forecast horizons because they define the actual supervised learning tasks.
    forecast_horizons: List[ForecastHorizon]
    # Store the exogenous candidate inputs so the modeling table design remains within project scope.
    candidate_covariates: List[str]
    # Store the split policy to prevent invalid random sampling in a time-series context.
    split_policy: Dict[str, str]
    # Store the evaluation metrics so model comparison remains pre-registered and consistent.
    evaluation_metrics: Dict[str, List[str]]
    # Store governance notes because this project explicitly requires zero-hallucination discipline.
    governance_rules: List[str]
    # Store creation timestamp for provenance and auditability.
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# Ensure the output directory exists because project artifacts should be generated idempotently.
def ensure_directory(path: Path) -> None:
    # Create missing directories without failing on repeated runs.
    path.mkdir(parents=True, exist_ok=True)


# Build the canonical Step 1 project specification directly from the agreed project brief.
def build_forecast_specification() -> ForecastTaskSpecification:
    # Return the formalized project contract so every later file can reference the same definitions.
    return ForecastTaskSpecification(
        project_name="Project A: Probabilistic Energy Load Forecaster for Ireland",
        geography="Ireland",
        target_variable="Electricity system load",
        target_unit="MW or GW depending on official source field; exact source unit must be verified during ingestion",
        temporal_resolution="Hourly",
        probabilistic_output={
            "forecast_type": "Probabilistic",
            "required_outputs": [
                "Point forecast using median or mean",
                "80% prediction interval",
                "90% prediction interval",
                "Quantile forecasts for operational evaluation",
            ],
            "note": "If source unit or target naming differs across EirGrid and fallback sources, preserve raw labels and standardize later.",
        },
        forecast_horizons=[
            ForecastHorizon(
                label="t_plus_1h",
                lead_hours=1,
                business_meaning="Ultra-short-term operational forecasting",
            ),
            ForecastHorizon(
                label="t_plus_24h",
                lead_hours=24,
                business_meaning="Day-ahead operational planning",
            ),
            ForecastHorizon(
                label="t_plus_168h",
                lead_hours=168,
                business_meaning="Week-ahead planning horizon",
            ),
        ],
        candidate_covariates=[
            "Historical electricity load",
            "Wind generation",
            "Temperature",
            "Humidity",
            "Wind speed",
            "Wind direction",
            "Surface pressure",
            "Precipitation",
            "Calendar variables",
            "Cyclical encodings",
        ],
        split_policy={
            "train_validation_test": "Time-based split only",
            "random_split": "Prohibited",
            "backtesting": "Rolling-origin evaluation required",
        },
        evaluation_metrics={
            "point_forecast_metrics": [
                "RMSE",
                "MAE",
            ],
            "probabilistic_metrics": [
                "Pinball loss",
                "80% interval empirical coverage",
                "90% interval empirical coverage",
            ],
        },
        governance_rules=[
            "Use official or clearly documented sources only",
            "Keep immutable raw data snapshots before cleaning",
            "Do not invent missing metadata",
            "Log transformations in preprocessing scripts",
            "Report failed models honestly",
            "Do not claim causality from forecast features without separate testing",
        ],
    )


# Convert nested dataclasses into a JSON-serializable structure because Horizon objects are not directly serializable by default.
def specification_to_dict(specification: ForecastTaskSpecification) -> Dict:
    # Transform the dataclass tree into plain Python primitives for stable JSON persistence.
    return asdict(specification)


# Persist the project specification because Step 1 should produce a durable artifact, not just chat agreement.
def write_specification(specification: ForecastTaskSpecification, output_path: Path) -> None:
    # Ensure the parent directory exists so file writing does not fail on missing folders.
    ensure_directory(output_path.parent)
    # Open the target file in UTF-8 mode for portability and deterministic encoding.
    with output_path.open("w", encoding="utf-8") as handle:
        # Write the structured specification with indentation so it remains both machine- and human-readable.
        json.dump(specification_to_dict(specification), handle, indent=2)


# Run the Step 1 formalization workflow so the project can proceed with a locked forecasting contract.
def main() -> int:
    # Configure logging before creating any artifact so execution is traceable.
    configure_logging()
    try:
        # Build the canonical specification from the agreed project design.
        specification = build_forecast_specification()
        # Define a deterministic artifact path so the team knows where the contract lives.
        output_path = Path("artifacts") / "forecast_task_spec.json"
        # Persist the specification to disk for downstream consumption.
        write_specification(specification, output_path)
        # Emit an operational log because artifact creation is the completion criterion for Step 1.
        LOGGER.info("Forecast task specification written to %s", output_path)
        # Print the generated specification so the operator can inspect it immediately.
        print(json.dumps(specification_to_dict(specification), indent=2))
        # Return success because the artifact was produced without error.
        return 0
    except Exception as exc:
        # Log the full exception so project governance can diagnose why Step 1 failed.
        LOGGER.exception("Failed to generate forecast task specification: %s", exc)
        # Return non-zero because later steps should not proceed on a failed specification stage.
        return 1


# Execute the script only when run directly so imports remain side-effect free.
if __name__ == "__main__":
    # Exit with an explicit process code for shell and CI compatibility.
    sys.exit(main())