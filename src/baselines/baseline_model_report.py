# Import json so baseline metric files can be read and the consolidated summary can be written.
import json

# Import logging so report generation emits traceable diagnostics.
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

# Import pandas so comparison tables can be assembled and exported cleanly.
import pandas as pd


# Create a module-level logger so all report-generation diagnostics use one consistent channel.
LOGGER = logging.getLogger("baseline_model_report")


# Store the baseline model output directory because prior scripts wrote their artifacts there.
BASELINE_DIR = Path("models/baselines")


# Store the report output directory so consolidated evaluation artifacts remain separate from raw model files.
REPORT_DIR = Path("reports/baselines")


# Configure logging once so terminal diagnostics remain structured and readable.
def configure_logging() -> None:
    # Initialize logging with timestamped structured formatting for reproducibility and debugging.
    logging.basicConfig(
        # Use INFO because lifecycle visibility is needed without excessive verbosity.
        level=logging.INFO,
        # Use a structured format so logs remain readable in terminal and saved logs.
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# Ensure a directory exists before writing files so failures reflect logic issues, not missing folders.
def ensure_directory(path: Path) -> None:
    # Create the full directory tree idempotently because repeated report runs are expected.
    path.mkdir(parents=True, exist_ok=True)


# Return an ISO-8601 UTC timestamp so report metadata remains temporally explicit.
def utc_now_iso() -> str:
    # Generate the current UTC timestamp because report lineage should be preserved explicitly.
    return datetime.now(timezone.utc).isoformat()


# Read a JSON file because prior baseline scripts persisted their metrics as JSON artifacts.
def read_json(path: Path) -> Dict:
    # Open the JSON file in UTF-8 mode so deserialization remains portable.
    with path.open("r", encoding="utf-8") as handle:
        # Parse and return the JSON payload because downstream reporting needs structured access.
        return json.load(handle)


# Write a JSON file because the consolidated report metadata should remain machine-readable.
def write_json(path: Path, payload: Dict) -> None:
    # Open the destination file in UTF-8 mode so serialization remains portable.
    with path.open("w", encoding="utf-8") as handle:
        # Persist formatted JSON so the report metadata remains readable and auditable.
        json.dump(payload, handle, indent=2)


# Safely extract a nested value because baseline outputs are not perfectly schema-identical.
def get_nested(payload: Dict, keys: List[str], default=None):
    # Start from the root payload because nested extraction must traverse predictably.
    current = payload
    # Iterate through the requested key path because each level may or may not exist.
    for key in keys:
        # Return the default immediately when the current object is not a dictionary.
        if not isinstance(current, dict):
            return default
        # Return the default when the requested key is absent.
        if key not in current:
            return default
        # Advance to the next nesting level because the path still exists.
        current = current[key]
    # Return the extracted value because the full path resolved successfully.
    return current


# Convert the seasonal naive output into the common comparison schema because all models should be compared uniformly.
def normalize_seasonal_naive(payload: Dict) -> Dict:
    # Return a normalized record so the report table can compare heterogeneous baseline outputs consistently.
    return {
        "model_name": payload.get("model", "seasonal_naive_weekly"),
        "model_family": "SeasonalNaive",
        "validation_mae": get_nested(payload, ["validation_metrics", "MAE"]),
        "validation_rmse": get_nested(payload, ["validation_metrics", "RMSE"]),
        "test_mae": get_nested(payload, ["test_metrics", "MAE"]),
        "test_rmse": get_nested(payload, ["test_metrics", "RMSE"]),
        "validation_pinball_q10": None,
        "validation_pinball_q50": None,
        "validation_pinball_q90": None,
        "validation_coverage_80": None,
        "test_pinball_q10": None,
        "test_pinball_q50": None,
        "test_pinball_q90": None,
        "test_coverage_80": None,
        "primary_selection_metric": get_nested(payload, ["test_metrics", "RMSE"]),
        "selection_metric_name": "test_rmse",
    }


# Convert the SARIMAX output into the common comparison schema because all models should be compared uniformly.
def normalize_sarimax(payload: Dict) -> Dict:
    # Return a normalized record so the report table can compare heterogeneous baseline outputs consistently.
    return {
        "model_name": payload.get("selected_model_name", "sarimax"),
        "model_family": payload.get("model_family", "SARIMAX"),
        "validation_mae": get_nested(payload, ["validation_metrics", "MAE"]),
        "validation_rmse": get_nested(payload, ["validation_metrics", "RMSE"]),
        "test_mae": get_nested(payload, ["test_metrics", "MAE"]),
        "test_rmse": get_nested(payload, ["test_metrics", "RMSE"]),
        "validation_pinball_q10": None,
        "validation_pinball_q50": None,
        "validation_pinball_q90": None,
        "validation_coverage_80": None,
        "test_pinball_q10": None,
        "test_pinball_q50": None,
        "test_pinball_q90": None,
        "test_coverage_80": None,
        "primary_selection_metric": get_nested(payload, ["test_metrics", "RMSE"]),
        "selection_metric_name": "test_rmse",
    }


# Convert the quantile GBR output into the common comparison schema because it contains additional probabilistic metrics.
def normalize_quantile_gbr(payload: Dict) -> Dict:
    # Return a normalized record so the report table can compare heterogeneous baseline outputs consistently.
    return {
        "model_name": payload.get("selected_model_name", "quantile_gbr"),
        "model_family": payload.get("model_family", "GradientBoostingRegressor_Quantile"),
        "validation_mae": get_nested(payload, ["validation_metrics", "point_metrics", "MAE"]),
        "validation_rmse": get_nested(payload, ["validation_metrics", "point_metrics", "RMSE"]),
        "test_mae": get_nested(payload, ["test_metrics", "point_metrics", "MAE"]),
        "test_rmse": get_nested(payload, ["test_metrics", "point_metrics", "RMSE"]),
        "validation_pinball_q10": get_nested(payload, ["validation_metrics", "pinball_metrics", "pinball_q10"]),
        "validation_pinball_q50": get_nested(payload, ["validation_metrics", "pinball_metrics", "pinball_q50"]),
        "validation_pinball_q90": get_nested(payload, ["validation_metrics", "pinball_metrics", "pinball_q90"]),
        "validation_coverage_80": get_nested(payload, ["validation_metrics", "coverage_80_interval"]),
        "test_pinball_q10": get_nested(payload, ["test_metrics", "pinball_metrics", "pinball_q10"]),
        "test_pinball_q50": get_nested(payload, ["test_metrics", "pinball_metrics", "pinball_q50"]),
        "test_pinball_q90": get_nested(payload, ["test_metrics", "pinball_metrics", "pinball_q90"]),
        "test_coverage_80": get_nested(payload, ["test_metrics", "coverage_80_interval"]),
        "primary_selection_metric": get_nested(payload, ["test_metrics", "point_metrics", "RMSE"]),
        "selection_metric_name": "test_rmse",
    }


# Load and normalize all baseline outputs because reporting should fail early if any required artifact is missing.
def load_baseline_results() -> List[Dict]:
    # Define the required artifact paths because baseline comparison depends on prior model outputs.
    seasonal_path = BASELINE_DIR / "seasonal_naive_metrics.json"
    # Define the SARIMAX summary path because that model output must also be consolidated.
    sarimax_path = BASELINE_DIR / "sarimax_baseline_summary.json"
    # Define the quantile GBR summary path because it is central to the project objective.
    quantile_path = BASELINE_DIR / "quantile_gbr_summary.json"
    # Fail clearly when a required baseline artifact is missing because comparison would be incomplete.
    for required_path in [seasonal_path, sarimax_path, quantile_path]:
        if not required_path.exists():
            raise FileNotFoundError(f"Required baseline artifact not found: {required_path}")
    # Read the raw payloads so they can be normalized into a common comparison schema.
    seasonal_payload = read_json(seasonal_path)
    # Read the SARIMAX payload for the same reason.
    sarimax_payload = read_json(sarimax_path)
    # Read the quantile payload for the same reason.
    quantile_payload = read_json(quantile_path)
    # Return the normalized baseline records because the comparison table requires consistent row structure.
    return [
        normalize_seasonal_naive(seasonal_payload),
        normalize_sarimax(sarimax_payload),
        normalize_quantile_gbr(quantile_payload),
    ]


# Determine the best point-forecast baseline because the project should state this explicitly and honestly.
def identify_best_point_model(comparison_df: pd.DataFrame) -> Dict:
    # Rank by test RMSE because the central forecast comparison should prioritize out-of-sample accuracy.
    ranked_df = comparison_df.sort_values("test_rmse", ascending=True).reset_index(drop=True)
    # Select the top-ranked row because it is the current best point-forecast baseline.
    best_row = ranked_df.iloc[0]
    # Return a compact summary because the consolidated report should state the winner explicitly.
    return {
        "model_name": best_row["model_name"],
        "model_family": best_row["model_family"],
        "test_rmse": float(best_row["test_rmse"]),
        "test_mae": float(best_row["test_mae"]),
    }


# Determine the best probabilistic baseline because interval forecasting is the project's core requirement.
def identify_best_probabilistic_model(comparison_df: pd.DataFrame) -> Dict:
    # Filter to rows that actually contain probabilistic metrics because deterministic baselines cannot compete here.
    probabilistic_df = comparison_df.loc[comparison_df["test_pinball_q50"].notna()].copy()
    # Fail explicitly when no probabilistic model exists because the report should not fabricate one.
    if probabilistic_df.empty:
        raise ValueError("No probabilistic model metrics found in baseline comparison data.")
    # Rank by test q50 pinball because that is the central quantile scoring rule for the probabilistic model.
    ranked_df = probabilistic_df.sort_values("test_pinball_q50", ascending=True).reset_index(drop=True)
    # Select the top-ranked row because it is the current best probabilistic baseline.
    best_row = ranked_df.iloc[0]
    # Return a compact summary because the consolidated report should state the winner explicitly.
    return {
        "model_name": best_row["model_name"],
        "model_family": best_row["model_family"],
        "test_pinball_q50": float(best_row["test_pinball_q50"]),
        "test_coverage_80": float(best_row["test_coverage_80"]),
    }


# Add a qualitative interpretation because the report should be publication-ready rather than raw metrics only.
def build_interpretation(best_point: Dict, best_probabilistic: Dict) -> Dict:
    # Assess the 80% interval calibration because under- or over-coverage should be stated honestly.
    coverage_80 = best_probabilistic["test_coverage_80"]
    # Classify calibration conservatively because 80% nominal coverage is the target reference point.
    if coverage_80 is None:
        calibration_label = "inconclusive"
    elif coverage_80 < 0.75:
        calibration_label = "under-covered"
    elif coverage_80 > 0.85:
        calibration_label = "over-covered"
    else:
        calibration_label = "reasonably calibrated"
    # Return the narrative summary because later README/report writing should use explicit findings.
    return {
        "best_point_forecast_model": best_point["model_name"],
        "best_probabilistic_model": best_probabilistic["model_name"],
        "interval_calibration_assessment": calibration_label,
        "summary_statement": (
            f"The strongest current point-forecast baseline is {best_point['model_name']}, "
            f"and the strongest probabilistic baseline is {best_probabilistic['model_name']}. "
            f"The observed 80% interval coverage is {coverage_80:.4f}, which is {calibration_label} relative to the nominal target."
        ),
    }


# Execute the consolidated report workflow because baseline modeling is now complete and should be summarized formally.
def main() -> int:
    # Configure logging before any file or dataframe work begins.
    configure_logging()
    try:
        # Ensure the report output directory exists before writing artifacts.
        ensure_directory(REPORT_DIR)
        # Load and normalize the baseline outputs because comparison requires a common schema.
        normalized_results = load_baseline_results()
        # Build the comparison dataframe because tabular reporting is the most useful artifact at this stage.
        comparison_df = pd.DataFrame(normalized_results)
        # Rank by point-forecast test RMSE so the most competitive overall baseline appears first.
        comparison_df.sort_values("test_rmse", ascending=True, inplace=True)
        # Reset the index so the exported table remains clean and sequential.
        comparison_df.reset_index(drop=True, inplace=True)
        # Identify the best point-forecast model because this must be stated explicitly.
        best_point_model = identify_best_point_model(comparison_df)
        # Identify the best probabilistic model because this is central to the project objective.
        best_probabilistic_model = identify_best_probabilistic_model(comparison_df)
        # Build the narrative interpretation because the report should explain the results, not just list them.
        interpretation = build_interpretation(best_point_model, best_probabilistic_model)
        # Define output paths because both CSV and JSON artifacts are useful downstream.
        comparison_csv_path = REPORT_DIR / "baseline_model_comparison.csv"
        # Define the JSON summary path because the consolidated findings should remain machine-readable.
        summary_json_path = REPORT_DIR / "baseline_model_summary.json"
        # Export the comparison table to CSV because it is convenient for README/report insertion.
        comparison_df.to_csv(comparison_csv_path, index=False)
        # Build the summary payload because the project now needs one authoritative baseline comparison artifact.
        summary_payload = {
            "generated_at_utc": utc_now_iso(),
            "best_point_model": best_point_model,
            "best_probabilistic_model": best_probabilistic_model,
            "interpretation": interpretation,
            "comparison_csv_path": str(comparison_csv_path),
        }
        # Persist the summary JSON because the result should remain portable across later project steps.
        write_json(summary_json_path, summary_payload)
        # Log successful completion so the operator can distinguish a full run from a partial one.
        LOGGER.info("Baseline model report generated successfully.")
        # Print a concise terminal summary so the key outcome is visible immediately.
        print(
            json.dumps(
                {
                    "comparison_csv_path": str(comparison_csv_path),
                    "summary_json_path": str(summary_json_path),
                    "best_point_model": best_point_model,
                    "best_probabilistic_model": best_probabilistic_model,
                    "interpretation": interpretation,
                },
                indent=2,
            )
        )
        # Return success because the report generation completed without exceptions.
        return 0
    except Exception as exc:
        # Log the full exception because silent reporting failures would weaken the project audit trail.
        LOGGER.exception("Baseline model report generation failed: %s", exc)
        # Return failure so the shell can detect the issue.
        return 1


# Execute the workflow only when the script is run directly so imports remain side-effect free.
if __name__ == "__main__":
    # Exit with the workflow status so terminal execution receives an accurate process code.
    sys.exit(main())