# Import json so model metrics and selected configuration can be persisted for later comparison.
import json

# Import logging so the training and evaluation process emits traceable diagnostics.
import logging

# Import sys so the script can return explicit process exit codes to the shell.
import sys

# Import datetime utilities so output metadata remains timezone-aware and reproducible.
from datetime import datetime
from datetime import timezone

# Import Path so file-system operations remain portable and deterministic.
from pathlib import Path

# Import typing helpers so model-spec interfaces remain explicit and safer to maintain.
from typing import Dict
from typing import List
from typing import Tuple

# Import numpy so numerical metric calculations remain stable and vectorized.
import numpy as np

# Import pandas so the modeling table can be filtered into chronological windows.
import pandas as pd

# Import statsmodels SARIMAX because this project step requires a seasonal ARIMA baseline model.
from statsmodels.tsa.statespace.sarimax import SARIMAX


# Create a module-level logger so all diagnostics use one consistent channel.
LOGGER = logging.getLogger("baseline_sarimax")


# Store the modeling directory because this script should read the full hourly modeling table.
MODELING_DIR = Path("data/modeling/entsoe")


# Store the output directory so baseline model artifacts remain separated from raw and processed data.
MODEL_OUTPUT_DIR = Path("models/baselines")


# Configure logging once so terminal diagnostics remain structured and readable.
def configure_logging() -> None:
    # Initialize logging with timestamped structured formatting for reproducibility and debugging.
    logging.basicConfig(
        # Use INFO because lifecycle visibility is needed without excessive verbosity.
        level=logging.INFO,
        # Use a structured format so logs remain readable in terminal and saved records.
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# Ensure an output directory exists before writing files so failures reflect logic issues, not missing folders.
def ensure_directory(path: Path) -> None:
    # Create the full directory tree idempotently because repeated model runs are expected.
    path.mkdir(parents=True, exist_ok=True)


# Return an ISO-8601 UTC timestamp so metadata remains temporally explicit.
def utc_now_iso() -> str:
    # Generate the current UTC timestamp because model-run lineage should be preserved.
    return datetime.now(timezone.utc).isoformat()


# Read the modeling table because SARIMAX must consume the full hourly target series rather than compressed splits.
def read_modeling_table(path: Path) -> pd.DataFrame:
    # Load the CSV into a dataframe so chronological filtering and modeling can be applied.
    dataframe = pd.read_csv(path)
    # Fail explicitly if the dataset is empty because no model can be fitted on an empty table.
    if dataframe.empty:
        # Raise a value error so the operator can inspect the upstream modeling-table build.
        raise ValueError(f"Modeling table is empty: {path}")
    # Parse timestamps as UTC because all split boundaries and forecasts must use one coherent timezone.
    dataframe["timestamp_utc"] = pd.to_datetime(dataframe["timestamp_utc"], utc=True)
    # Sort chronologically because state-space models must operate on ordered time series.
    dataframe.sort_values("timestamp_utc", inplace=True)
    # Reset the index so downstream slicing remains deterministic.
    dataframe.reset_index(drop=True, inplace=True)
    # Return the prepared modeling table because it is now suitable for chronological slicing.
    return dataframe


# Slice the full modeling table into fixed chronological windows while preserving all hourly rows.
def build_windows(dataframe: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    # Define the train end boundary because the first nine months form the fitting window.
    train_end = pd.Timestamp("2024-09-30 23:00:00+00:00")
    # Define the validation start boundary because validation must immediately follow train.
    validation_start = pd.Timestamp("2024-10-01 00:00:00+00:00")
    # Define the validation end boundary because this window is used for model selection.
    validation_end = pd.Timestamp("2024-11-15 23:00:00+00:00")
    # Define the test start boundary because final evaluation must remain untouched during selection.
    test_start = pd.Timestamp("2024-11-16 00:00:00+00:00")
    # Create the training window while preserving missing target timestamps in-place.
    train_df = dataframe.loc[dataframe["timestamp_utc"] <= train_end].copy()
    # Create the validation window while preserving hourly continuity.
    validation_df = dataframe.loc[
        (dataframe["timestamp_utc"] >= validation_start) & (dataframe["timestamp_utc"] <= validation_end)
    ].copy()
    # Create the test window while preserving hourly continuity.
    test_df = dataframe.loc[dataframe["timestamp_utc"] >= test_start].copy()
    # Return the three chronological windows because they define the model-selection workflow.
    return {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }


# Build a small defensible SARIMAX search space because a student baseline should remain tractable and honest.
def get_candidate_specs() -> List[Dict]:
    # Return a compact set of daily-seasonal candidate specifications to balance rigor and runtime.
    return [
        {
            "name": "sarimax_101_101_24",
            "order": (1, 0, 1),
            "seasonal_order": (1, 0, 1, 24),
            "trend": "c",
        },
        {
            "name": "sarimax_111_101_24",
            "order": (1, 1, 1),
            "seasonal_order": (1, 0, 1, 24),
            "trend": "c",
        },
        {
            "name": "sarimax_201_101_24",
            "order": (2, 0, 1),
            "seasonal_order": (1, 0, 1, 24),
            "trend": "c",
        },
    ]


# Fit one SARIMAX model because each candidate must be estimated and compared on the validation window.
def fit_sarimax_model(train_series: pd.Series, spec: Dict):
    # Instantiate the SARIMAX model without simple differencing so forecasts remain on the original load scale.
    model = SARIMAX(
        # Use the univariate load series because exogenous wind still contains training-window gaps.
        endog=train_series,
        # Apply the candidate non-seasonal order because it defines the ARIMA core dynamics.
        order=spec["order"],
        # Apply the candidate seasonal order because daily hourly seasonality is a practical first SARIMAX baseline.
        seasonal_order=spec["seasonal_order"],
        # Include a constant trend because electricity demand usually has a non-zero level.
        trend=spec["trend"],
        # Relax stationarity enforcement because strict constraints can cause avoidable optimization failures in baseline search.
        enforce_stationarity=False,
        # Relax invertibility enforcement for the same reason, prioritizing fit tractability in the first baseline pass.
        enforce_invertibility=False,
    )
    # Fit by maximum likelihood while suppressing optimizer chatter in the terminal.
    results = model.fit(disp=False, maxiter=200)
    # Return the fitted results object because it will be used for forecasting and diagnostics.
    return results


# Compute standard point-forecast metrics because this baseline is still judged against actual load values.
def compute_metrics(actual: np.ndarray, forecast: np.ndarray) -> Dict[str, float]:
    # Compute mean absolute error because it is robust and directly interpretable in MW terms.
    mae = np.mean(np.abs(actual - forecast))
    # Compute root mean squared error because it penalizes larger misses more strongly.
    rmse = np.sqrt(np.mean((actual - forecast) ** 2))
    # Return both metrics as plain floats so they serialize cleanly to JSON.
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
    }


# Evaluate forecasts only where actual targets exist because missing target values should not distort metrics.
def evaluate_forecast(actual_series: pd.Series, forecast_series: pd.Series) -> Dict[str, float]:
    # Build a mask of observed actual values because evaluation must exclude undefined outcomes.
    valid_mask = actual_series.notna()
    # Compute metrics on the aligned observed subset only.
    return compute_metrics(
        actual=actual_series.loc[valid_mask].to_numpy(),
        forecast=forecast_series.loc[valid_mask].to_numpy(),
    )


# Convert model forecasts into a timestamp-aligned dataframe because later review should be inspection-friendly.
def build_forecast_frame(window_df: pd.DataFrame, forecast_values: np.ndarray, model_name: str) -> pd.DataFrame:
    # Create a dataframe with timestamps and actuals so model output can be inspected chronologically.
    result = window_df[["timestamp_utc", "load_mw"]].copy()
    # Attach the model identifier so forecast files remain self-describing when multiple baselines are compared.
    result["model_name"] = model_name
    # Attach the forecast vector so error analysis can be performed later without re-running the model.
    result["forecast_load_mw"] = forecast_values
    # Return the aligned forecast frame because it is a useful downstream artifact.
    return result


# Run the validation search because the best SARIMAX specification should be chosen on the validation window only.
def select_best_model(train_df: pd.DataFrame, validation_df: pd.DataFrame) -> Tuple[Dict, Dict, pd.DataFrame]:
    # Build the candidate specification list because baseline search should remain explicit and reviewable.
    candidate_specs = get_candidate_specs()
    # Initialize storage for candidate outcomes so each attempted model can be reported honestly.
    candidate_results: List[Dict] = []
    # Keep track of the current best specification so test evaluation can refit it later.
    best_spec = None
    # Keep track of the best validation score so model selection remains deterministic.
    best_rmse = np.inf
    # Keep track of the best forecast frame so it can be saved without recomputation.
    best_validation_forecast_frame = None
    # Extract the training target series while preserving missing hourly timestamps in-place.
    train_series = train_df.set_index("timestamp_utc")["load_mw"].asfreq("h")
    # Iterate through each candidate because baseline comparison requires empirical validation.
    for spec in candidate_specs:
        try:
            # Log the candidate under evaluation so runtime progress is transparent.
            LOGGER.info("Fitting validation candidate: %s", spec["name"])
            # Fit the current SARIMAX specification on the training window only.
            fitted_model = fit_sarimax_model(train_series, spec)
            # Forecast exactly the number of hours in the validation window so chronology remains aligned.
            validation_forecast = fitted_model.forecast(steps=len(validation_df))
            # Convert forecasts to a NumPy array because alignment will be handled explicitly by the window frame.
            validation_forecast_values = np.asarray(validation_forecast)
            # Build a timestamp-aligned forecast frame so validation output remains inspectable.
            validation_forecast_frame = build_forecast_frame(
                window_df=validation_df,
                forecast_values=validation_forecast_values,
                model_name=spec["name"],
            )
            # Compute validation metrics only on timestamps with observed target values.
            validation_metrics = evaluate_forecast(
                actual_series=validation_forecast_frame["load_mw"],
                forecast_series=validation_forecast_frame["forecast_load_mw"],
            )
            # Persist the candidate result so all tried models are reported honestly.
            candidate_results.append(
                {
                    "name": spec["name"],
                    "order": spec["order"],
                    "seasonal_order": spec["seasonal_order"],
                    "trend": spec["trend"],
                    "validation_metrics": validation_metrics,
                    "aic": float(fitted_model.aic),
                }
            )
            # Update the incumbent best model when the current candidate improves validation RMSE.
            if validation_metrics["RMSE"] < best_rmse:
                # Store the improved RMSE because later candidates must beat it to become selected.
                best_rmse = validation_metrics["RMSE"]
                # Store the winning specification because it will be refit on train+validation later.
                best_spec = spec
                # Store the winning validation forecast frame so it can be saved directly.
                best_validation_forecast_frame = validation_forecast_frame
        except Exception as exc:
            # Record the candidate failure explicitly because failed models should be reported, not hidden.
            candidate_results.append(
                {
                    "name": spec["name"],
                    "order": spec["order"],
                    "seasonal_order": spec["seasonal_order"],
                    "trend": spec["trend"],
                    "error": str(exc),
                }
            )
            # Log the failure so the operator can see which candidate broke and why.
            LOGGER.warning("Candidate failed: %s | %s", spec["name"], exc)
    # Fail explicitly when no candidate fitted successfully because downstream test evaluation would be invalid.
    if best_spec is None or best_validation_forecast_frame is None:
        # Raise an error so the operator knows the SARIMAX search did not produce a usable model.
        raise RuntimeError("No SARIMAX candidate fitted successfully on the validation search.")
    # Return the winning specification, the full candidate report, and the winning validation forecasts.
    return best_spec, {"candidates": candidate_results}, best_validation_forecast_frame


# Refit the selected specification on train+validation and evaluate on the untouched test window.
def run_test_evaluation(train_df: pd.DataFrame, validation_df: pd.DataFrame, test_df: pd.DataFrame, best_spec: Dict):
    # Concatenate train and validation because the final baseline should use all pre-test information.
    train_plus_validation_df = pd.concat([train_df, validation_df], ignore_index=True)
    # Build the final fitting series on the full pre-test hourly window.
    train_plus_validation_series = train_plus_validation_df.set_index("timestamp_utc")["load_mw"].asfreq("h")
    # Fit the selected SARIMAX specification on the combined pre-test window.
    fitted_model = fit_sarimax_model(train_plus_validation_series, best_spec)
    # Forecast exactly the number of hours in the test window so chronology remains aligned.
    test_forecast = fitted_model.forecast(steps=len(test_df))
    # Convert forecasts to a NumPy array because the output frame will handle timestamp alignment.
    test_forecast_values = np.asarray(test_forecast)
    # Build the timestamp-aligned test forecast frame so the final evaluation can be inspected later.
    test_forecast_frame = build_forecast_frame(
        window_df=test_df,
        forecast_values=test_forecast_values,
        model_name=best_spec["name"],
    )
    # Compute test metrics only where actual load exists because missing targets should not distort evaluation.
    test_metrics = evaluate_forecast(
        actual_series=test_forecast_frame["load_mw"],
        forecast_series=test_forecast_frame["forecast_load_mw"],
    )
    # Return the test metrics and aligned forecast frame because both are needed as output artifacts.
    return test_metrics, test_forecast_frame


# Execute the full SARIMAX baseline workflow because this is the next required project step.
def main() -> int:
    # Configure logging before any file or model work begins.
    configure_logging()
    try:
        # Ensure the model output directory exists before writing artifacts.
        ensure_directory(MODEL_OUTPUT_DIR)
        # Define the modeling-table path because SARIMAX must consume the full hourly target series.
        modeling_table_path = MODELING_DIR / "ireland_load_modeling_table.csv"
        # Load the full modeling table because compressed split files are not appropriate for SARIMAX.
        modeling_df = read_modeling_table(modeling_table_path)
        # Build the chronological windows while preserving missing timestamps in-place.
        windows = build_windows(modeling_df)
        # Run the validation search because candidate selection must happen before test evaluation.
        best_spec, candidate_report, validation_forecast_frame = select_best_model(
            train_df=windows["train"],
            validation_df=windows["validation"],
        )
        # Compute validation metrics for the selected model because they belong in the final summary.
        selected_validation_metrics = evaluate_forecast(
            actual_series=validation_forecast_frame["load_mw"],
            forecast_series=validation_forecast_frame["forecast_load_mw"],
        )
        # Run the final untouched test evaluation by refitting on train plus validation.
        test_metrics, test_forecast_frame = run_test_evaluation(
            train_df=windows["train"],
            validation_df=windows["validation"],
            test_df=windows["test"],
            best_spec=best_spec,
        )
        # Define output paths for the summary and forecast artifacts because results should be reviewable without rerunning.
        summary_output_path = MODEL_OUTPUT_DIR / "sarimax_baseline_summary.json"
        # Define the candidate-report path because failed and losing models should still be documented.
        candidate_output_path = MODEL_OUTPUT_DIR / "sarimax_candidate_report.json"
        # Define the validation forecast path because baseline errors should be inspectable chronologically.
        validation_forecast_output_path = MODEL_OUTPUT_DIR / "sarimax_validation_forecast.csv"
        # Define the test forecast path because final baseline behavior should be inspectable chronologically.
        test_forecast_output_path = MODEL_OUTPUT_DIR / "sarimax_test_forecast.csv"
        # Build the summary payload because the selected model and its metrics should be preserved compactly.
        summary_payload = {
            "generated_at_utc": utc_now_iso(),
            "model_family": "SARIMAX",
            "selected_model_name": best_spec["name"],
            "selected_order": best_spec["order"],
            "selected_seasonal_order": best_spec["seasonal_order"],
            "selected_trend": best_spec["trend"],
            "uses_exog": False,
            "validation_metrics": selected_validation_metrics,
            "test_metrics": test_metrics,
            "note": "Univariate SARIMAX baseline used intentionally because training-window wind exogenous values still contain missing observations.",
        }
        # Persist the compact summary because it will be compared against other baselines.
        with summary_output_path.open("w", encoding="utf-8") as handle:
            # Write formatted JSON so the result remains both machine-readable and easy to inspect.
            json.dump(summary_payload, handle, indent=2)
        # Persist the full candidate report because failed and losing candidates should remain visible.
        with candidate_output_path.open("w", encoding="utf-8") as handle:
            # Write formatted JSON so the search process remains auditable.
            json.dump(candidate_report, handle, indent=2)
        # Write the validation forecasts because chronological forecast inspection is useful for diagnostics.
        validation_forecast_frame.to_csv(validation_forecast_output_path, index=False)
        # Write the test forecasts because final untouched forecast inspection is useful for diagnostics.
        test_forecast_frame.to_csv(test_forecast_output_path, index=False)
        # Log successful completion so the operator can distinguish a full run from a partial one.
        LOGGER.info("SARIMAX baseline completed successfully.")
        # Print the compact summary so the operator can inspect the selected model immediately.
        print(json.dumps(summary_payload, indent=2))
        # Return success because the workflow completed without exceptions.
        return 0
    except Exception as exc:
        # Log the full exception because silent model failures would compromise baseline comparison.
        LOGGER.exception("SARIMAX baseline failed: %s", exc)
        # Return failure so terminal execution can detect the issue.
        return 1


# Execute the workflow only when the file is run directly so imports remain side-effect free.
if __name__ == "__main__":
    # Exit with the workflow status so the shell receives an accurate process code.
    sys.exit(main())