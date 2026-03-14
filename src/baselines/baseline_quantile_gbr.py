# Import json so model metrics and configuration can be persisted for later comparison.
import json

# Import logging so the modeling process emits operational diagnostics.
import logging

# Import sys so the script can return explicit process exit codes.
import sys

# Import datetime utilities so run metadata remains timezone-aware and reproducible.
from datetime import datetime
from datetime import timezone

# Import Path so file-system operations remain portable and deterministic.
from pathlib import Path

# Import typing helpers so interfaces remain explicit and safer to maintain.
from typing import Dict
from typing import List
from typing import Tuple

# Import numpy so numerical calculations remain efficient and stable.
import numpy as np

# Import pandas so split datasets can be loaded and transformed safely.
import pandas as pd

# Import GradientBoostingRegressor because this baseline requires quantile regression capability.
from sklearn.ensemble import GradientBoostingRegressor


# Create a module-level logger so all diagnostics use one consistent channel.
LOGGER = logging.getLogger("baseline_quantile_gbr")


# Store the split directory because this model can safely consume the row-wise split datasets.
SPLIT_DIR = Path("data/splits/entsoe")


# Store the output directory so baseline artifacts remain organized and reviewable.
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


# Read one split CSV because the quantile baseline consumes the prepared chronological partitions.
def read_split(path: Path) -> pd.DataFrame:
    # Load the CSV into a dataframe so feature selection and modeling can be applied.
    dataframe = pd.read_csv(path)
    # Fail explicitly if the dataset is empty because no valid model can be trained on an empty split.
    if dataframe.empty:
        # Raise a value error so the operator can inspect the upstream split step.
        raise ValueError(f"Split dataset is empty: {path}")
    # Parse timestamps as UTC because downstream prediction exports should preserve chronology.
    dataframe["timestamp_utc"] = pd.to_datetime(dataframe["timestamp_utc"], utc=True)
    # Sort chronologically so exported prediction files remain easy to inspect.
    dataframe.sort_values("timestamp_utc", inplace=True)
    # Reset the index so downstream processing remains deterministic.
    dataframe.reset_index(drop=True, inplace=True)
    # Return the prepared dataframe because it is now ready for feature engineering.
    return dataframe


# Define the feature set because the quantile baseline should use the modeling-table features already built.
def get_feature_columns() -> List[str]:
    # Return a compact but relevant feature list aligned with the project design.
    return [
        "hour_of_day",
        "day_of_week",
        "month",
        "day_of_year",
        "sin_hour",
        "cos_hour",
        "sin_dow",
        "cos_dow",
        "sin_month",
        "cos_month",
        "load_lag_1",
        "load_lag_24",
        "load_lag_48",
        "load_lag_168",
        "load_rollmean_24",
        "load_rollstd_24",
        "wind_onshore_mw",
        "wind_missing_flag_postfill",
    ]


# Apply deterministic feature preparation because tree models still require complete numeric inputs.
def prepare_features(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Copy each split so the original loaded frames remain unchanged for debugging and reuse.
    train_result = train_df.copy()
    # Copy the validation split for the same reason.
    validation_result = validation_df.copy()
    # Copy the test split for the same reason.
    test_result = test_df.copy()
    # Define the selected feature columns once so the same contract is used across all splits.
    feature_columns = get_feature_columns()
    # Compute the training median wind value because feature imputation should be learned from train only.
    train_wind_median = train_result["wind_onshore_mw"].median()
    # Fill missing wind values in the training split because gradient boosting cannot consume NaN features.
    train_result["wind_onshore_mw"] = train_result["wind_onshore_mw"].fillna(train_wind_median)
    # Fill missing wind values in validation using the training-derived median to prevent data leakage.
    validation_result["wind_onshore_mw"] = validation_result["wind_onshore_mw"].fillna(train_wind_median)
    # Fill missing wind values in test using the training-derived median to prevent data leakage.
    test_result["wind_onshore_mw"] = test_result["wind_onshore_mw"].fillna(train_wind_median)
    # Drop rows with missing lag or rolling features in training because those rows are unusable for supervised fitting.
    train_result = train_result.dropna(subset=feature_columns + ["load_mw"]).copy()
    # Drop unusable rows in validation for fair comparable evaluation.
    validation_result = validation_result.dropna(subset=feature_columns + ["load_mw"]).copy()
    # Drop unusable rows in test for fair comparable evaluation.
    test_result = test_result.dropna(subset=feature_columns + ["load_mw"]).copy()
    # Return the cleaned split frames because they are now suitable for supervised learning.
    return train_result, validation_result, test_result


# Build feature matrices and targets because scikit-learn expects explicit X and y inputs.
def build_xy(dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Select the agreed feature columns so the model sees only engineered predictors.
    X = dataframe[get_feature_columns()].copy()
    # Select the target load series because this is the supervised prediction variable.
    y = dataframe["load_mw"].copy()
    # Return the feature matrix and target vector because the model API requires both.
    return X, y


# Compute point-forecast metrics because the q50 forecast should still be judged as a central estimate.
def compute_point_metrics(actual: np.ndarray, forecast: np.ndarray) -> Dict[str, float]:
    # Compute mean absolute error because it is directly interpretable in MW units.
    mae = np.mean(np.abs(actual - forecast))
    # Compute root mean squared error because it penalizes large misses more strongly.
    rmse = np.sqrt(np.mean((actual - forecast) ** 2))
    # Return the metrics as floats so they serialize cleanly to JSON.
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
    }


# Compute pinball loss because quantile forecast quality must be evaluated directly.
def compute_pinball_loss(actual: np.ndarray, forecast: np.ndarray, alpha: float) -> float:
    # Compute residuals because pinball loss depends on forecast error direction and quantile level.
    residuals = actual - forecast
    # Apply the standard pinball formulation because probabilistic accuracy is central to this project.
    loss = np.maximum(alpha * residuals, (alpha - 1) * residuals)
    # Return the mean loss as a float so it serializes cleanly to JSON.
    return float(np.mean(loss))


# Compute interval coverage because the forecast interval must be checked for calibration realism.
def compute_interval_coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    # Identify whether each actual observation falls inside the predicted interval.
    covered = (actual >= lower) & (actual <= upper)
    # Return empirical coverage as a proportion because this is the required probabilistic metric.
    return float(np.mean(covered))


# Create a compact hyperparameter search space because a student baseline should remain tractable.
def get_candidate_configs() -> List[Dict]:
    # Return a small defensible configuration set to balance model quality and runtime.
    return [
        {
            "name": "gbr_small",
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 3,
            "min_samples_leaf": 10,
            "subsample": 1.0,
        },
        {
            "name": "gbr_medium",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 3,
            "min_samples_leaf": 10,
            "subsample": 1.0,
        },
        {
            "name": "gbr_deeper",
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 4,
            "min_samples_leaf": 10,
            "subsample": 1.0,
        },
    ]


# Fit one quantile or median gradient boosting model because each alpha requires its own estimator.
def fit_gbr_model(X_train: pd.DataFrame, y_train: pd.Series, config: Dict, loss: str, alpha: float = 0.5):
    # Instantiate the regressor with the supplied configuration because candidate selection is validation-driven.
    model = GradientBoostingRegressor(
        # Use quantile or squared_error according to the current task.
        loss=loss,
        # Use the quantile level only when the loss is quantile-based.
        alpha=alpha,
        # Use the candidate tree count because it controls ensemble capacity.
        n_estimators=config["n_estimators"],
        # Use the candidate learning rate because it controls boosting step size.
        learning_rate=config["learning_rate"],
        # Use the candidate depth because it controls interaction complexity.
        max_depth=config["max_depth"],
        # Use the candidate minimum leaf size because it regularizes the fitted trees.
        min_samples_leaf=config["min_samples_leaf"],
        # Use the candidate subsample because it controls stochasticity and regularization.
        subsample=config["subsample"],
        # Fix the random state so results remain reproducible across runs.
        random_state=42,
    )
    # Fit the model on the training data because the estimator must learn the relationship from historical rows.
    model.fit(X_train, y_train)
    # Return the fitted model because it will be used for validation or test prediction.
    return model


# Evaluate one candidate configuration on the validation set because model selection must happen before test use.
def evaluate_candidate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    config: Dict,
) -> Dict:
    # Log the candidate under evaluation so runtime progress remains visible.
    LOGGER.info("Evaluating quantile GBR candidate: %s", config["name"])
    # Fit the lower-quantile model because interval construction requires q10.
    model_q10 = fit_gbr_model(X_train, y_train, config=config, loss="quantile", alpha=0.10)
    # Fit the median model because point forecasting and central tendency require q50.
    model_q50 = fit_gbr_model(X_train, y_train, config=config, loss="quantile", alpha=0.50)
    # Fit the upper-quantile model because interval construction requires q90.
    model_q90 = fit_gbr_model(X_train, y_train, config=config, loss="quantile", alpha=0.90)
    # Generate q10 predictions on validation because probabilistic evaluation needs the lower bound.
    pred_q10 = model_q10.predict(X_validation)
    # Generate q50 predictions on validation because point metrics will be computed on the median.
    pred_q50 = model_q50.predict(X_validation)
    # Generate q90 predictions on validation because probabilistic evaluation needs the upper bound.
    pred_q90 = model_q90.predict(X_validation)
    # Compute point metrics on q50 because the median acts as the central forecast estimate.
    point_metrics = compute_point_metrics(y_validation.to_numpy(), pred_q50)
    # Compute pinball losses because quantile accuracy must be measured directly.
    pinball_metrics = {
        "pinball_q10": compute_pinball_loss(y_validation.to_numpy(), pred_q10, 0.10),
        "pinball_q50": compute_pinball_loss(y_validation.to_numpy(), pred_q50, 0.50),
        "pinball_q90": compute_pinball_loss(y_validation.to_numpy(), pred_q90, 0.90),
    }
    # Compute empirical coverage of the q10-q90 interval because calibration is a core project output.
    coverage_80 = compute_interval_coverage(
        y_validation.to_numpy(),
        pred_q10,
        pred_q90,
    )
    # Return the full validation result because candidate comparison must consider both point and probabilistic quality.
    return {
        "config": config,
        "point_metrics": point_metrics,
        "pinball_metrics": pinball_metrics,
        "coverage_80_interval": coverage_80,
        "pred_q10": pred_q10,
        "pred_q50": pred_q50,
        "pred_q90": pred_q90,
        "models": {
            "q10": model_q10,
            "q50": model_q50,
            "q90": model_q90,
        },
    }


# Select the best candidate using validation pinball loss at q50 because the central forecast should remain strong.
def select_best_candidate(candidate_results: List[Dict]) -> Dict:
    # Sort candidates by validation median pinball loss because lower is better and directly tied to q50 quality.
    ordered_results = sorted(
        candidate_results,
        key=lambda result: result["pinball_metrics"]["pinball_q50"],
    )
    # Return the best validation candidate because it will be refit on train plus validation for test evaluation.
    return ordered_results[0]


# Build a timestamp-aligned prediction dataframe because forecast inspection should remain easy after the run.
def build_prediction_frame(
    dataframe: pd.DataFrame,
    pred_q10: np.ndarray,
    pred_q50: np.ndarray,
    pred_q90: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    # Create a compact frame with timestamps and actuals so chronological diagnostics remain possible.
    result = dataframe[["timestamp_utc", "load_mw"]].copy()
    # Attach the model identifier so exported predictions remain self-describing.
    result["model_name"] = model_name
    # Attach q10 because lower-bound interval analysis depends on it.
    result["pred_q10"] = pred_q10
    # Attach q50 because central point analysis depends on it.
    result["pred_q50"] = pred_q50
    # Attach q90 because upper-bound interval analysis depends on it.
    result["pred_q90"] = pred_q90
    # Return the aligned prediction frame because it is a useful diagnostic artifact.
    return result


# Refit the selected configuration on train plus validation because final test evaluation should use all pre-test data.
def run_test_evaluation(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    best_config: Dict,
) -> Dict:
    # Concatenate train and validation because the final model should use all information before test time.
    train_plus_validation_df = pd.concat([train_df, validation_df], ignore_index=True)
    # Build feature matrix and target vector from the combined pre-test dataset.
    X_train_plus_validation, y_train_plus_validation = build_xy(train_plus_validation_df)
    # Build the test feature matrix and target vector because untouched evaluation must occur on the held-out set.
    X_test, y_test = build_xy(test_df)
    # Fit q10 on the full pre-test data because interval evaluation requires the lower quantile.
    model_q10 = fit_gbr_model(X_train_plus_validation, y_train_plus_validation, best_config, loss="quantile", alpha=0.10)
    # Fit q50 on the full pre-test data because point evaluation requires the median forecast.
    model_q50 = fit_gbr_model(X_train_plus_validation, y_train_plus_validation, best_config, loss="quantile", alpha=0.50)
    # Fit q90 on the full pre-test data because interval evaluation requires the upper quantile.
    model_q90 = fit_gbr_model(X_train_plus_validation, y_train_plus_validation, best_config, loss="quantile", alpha=0.90)
    # Generate q10 test predictions because interval construction depends on the lower bound.
    pred_q10 = model_q10.predict(X_test)
    # Generate q50 test predictions because point and central probabilistic evaluation depend on the median.
    pred_q50 = model_q50.predict(X_test)
    # Generate q90 test predictions because interval construction depends on the upper bound.
    pred_q90 = model_q90.predict(X_test)
    # Compute point metrics on q50 because the median forecast is the central estimate.
    point_metrics = compute_point_metrics(y_test.to_numpy(), pred_q50)
    # Compute pinball losses because probabilistic test evaluation requires quantile-specific scoring.
    pinball_metrics = {
        "pinball_q10": compute_pinball_loss(y_test.to_numpy(), pred_q10, 0.10),
        "pinball_q50": compute_pinball_loss(y_test.to_numpy(), pred_q50, 0.50),
        "pinball_q90": compute_pinball_loss(y_test.to_numpy(), pred_q90, 0.90),
    }
    # Compute empirical interval coverage because calibration realism is a key deliverable.
    coverage_80 = compute_interval_coverage(y_test.to_numpy(), pred_q10, pred_q90)
    # Build a timestamp-aligned prediction frame because later review should not require re-running the model.
    prediction_frame = build_prediction_frame(
        dataframe=test_df,
        pred_q10=pred_q10,
        pred_q50=pred_q50,
        pred_q90=pred_q90,
        model_name=best_config["name"],
    )
    # Return the evaluation summary and prediction frame because both are required outputs.
    return {
        "point_metrics": point_metrics,
        "pinball_metrics": pinball_metrics,
        "coverage_80_interval": coverage_80,
        "prediction_frame": prediction_frame,
    }


# Execute the full quantile baseline workflow because this is the next required modeling step.
def main() -> int:
    # Configure logging before any file or model work begins.
    configure_logging()
    try:
        # Ensure the output directory exists before writing artifacts.
        ensure_directory(MODEL_OUTPUT_DIR)
        # Define the split file paths because this model consumes the row-wise supervised partitions.
        train_path = SPLIT_DIR / "train.csv"
        # Define the validation path because model selection must happen chronologically.
        validation_path = SPLIT_DIR / "validation.csv"
        # Define the test path because final evaluation must remain untouched during selection.
        test_path = SPLIT_DIR / "test.csv"
        # Load the three chronological splits because the baseline requires them explicitly.
        train_df = read_split(train_path)
        # Load validation because the best candidate must be selected before final test evaluation.
        validation_df = read_split(validation_path)
        # Load test because final evaluation occurs only after validation-based selection.
        test_df = read_split(test_path)
        # Prepare features deterministically because the tree model requires complete numeric inputs.
        train_df, validation_df, test_df = prepare_features(train_df, validation_df, test_df)
        # Build train matrices because candidate fitting requires explicit X and y inputs.
        X_train, y_train = build_xy(train_df)
        # Build validation matrices because candidate evaluation requires explicit X and y inputs.
        X_validation, y_validation = build_xy(validation_df)
        # Build the candidate configuration list because baseline selection should remain explicit and bounded.
        candidate_configs = get_candidate_configs()
        # Initialize storage for candidate results because all tried configurations should be reviewable.
        candidate_results = []
        # Evaluate each candidate because model selection must be evidence-based.
        for config in candidate_configs:
            # Run the validation evaluation for the current candidate.
            candidate_result = evaluate_candidate(X_train, y_train, X_validation, y_validation, config)
            # Append the result so it can participate in selection and later reporting.
            candidate_results.append(candidate_result)
        # Select the best candidate because only one configuration should advance to untouched test evaluation.
        best_candidate = select_best_candidate(candidate_results)
        # Build a validation prediction frame for the selected candidate because diagnostics should be exported.
        validation_prediction_frame = build_prediction_frame(
            dataframe=validation_df,
            pred_q10=best_candidate["pred_q10"],
            pred_q50=best_candidate["pred_q50"],
            pred_q90=best_candidate["pred_q90"],
            model_name=best_candidate["config"]["name"],
        )
        # Run untouched test evaluation using the selected configuration refit on train plus validation.
        test_result = run_test_evaluation(
            train_df=train_df,
            validation_df=validation_df,
            test_df=test_df,
            best_config=best_candidate["config"],
        )
        # Define summary and artifact output paths because results should remain reviewable without rerunning.
        summary_output_path = MODEL_OUTPUT_DIR / "quantile_gbr_summary.json"
        # Define the candidate report path because all tried configurations should remain visible.
        candidate_output_path = MODEL_OUTPUT_DIR / "quantile_gbr_candidate_report.json"
        # Define the validation prediction output path because selected-model behavior should be inspectable.
        validation_prediction_output_path = MODEL_OUTPUT_DIR / "quantile_gbr_validation_predictions.csv"
        # Define the test prediction output path because final interval behavior should be inspectable.
        test_prediction_output_path = MODEL_OUTPUT_DIR / "quantile_gbr_test_predictions.csv"
        # Build a compact candidate report because full model objects cannot be serialized to JSON.
        candidate_report = []
        # Convert candidate results into a serializable report so all attempted configs are preserved honestly.
        for candidate in candidate_results:
            # Append the serializable candidate summary because model comparison should remain auditable.
            candidate_report.append(
                {
                    "name": candidate["config"]["name"],
                    "n_estimators": candidate["config"]["n_estimators"],
                    "learning_rate": candidate["config"]["learning_rate"],
                    "max_depth": candidate["config"]["max_depth"],
                    "min_samples_leaf": candidate["config"]["min_samples_leaf"],
                    "subsample": candidate["config"]["subsample"],
                    "point_metrics": candidate["point_metrics"],
                    "pinball_metrics": candidate["pinball_metrics"],
                    "coverage_80_interval": candidate["coverage_80_interval"],
                }
            )
        # Build the final summary payload because the selected configuration and its metrics must be persisted cleanly.
        summary_payload = {
            "generated_at_utc": utc_now_iso(),
            "model_family": "GradientBoostingRegressor_Quantile",
            "selected_model_name": best_candidate["config"]["name"],
            "selected_config": best_candidate["config"],
            "feature_columns": get_feature_columns(),
            "validation_metrics": {
                "point_metrics": best_candidate["point_metrics"],
                "pinball_metrics": best_candidate["pinball_metrics"],
                "coverage_80_interval": best_candidate["coverage_80_interval"],
            },
            "test_metrics": {
                "point_metrics": test_result["point_metrics"],
                "pinball_metrics": test_result["pinball_metrics"],
                "coverage_80_interval": test_result["coverage_80_interval"],
            },
            "note": "Wind feature gaps were median-imputed using the training split only, while target gaps had already been excluded upstream.",
        }
        # Persist the summary because it will be compared against other baselines.
        with summary_output_path.open("w", encoding="utf-8") as handle:
            # Write formatted JSON so the summary remains machine-readable and easy to inspect.
            json.dump(summary_payload, handle, indent=2)
        # Persist the candidate report because losing configurations should still remain visible.
        with candidate_output_path.open("w", encoding="utf-8") as handle:
            # Write formatted JSON so the validation search remains auditable.
            json.dump(candidate_report, handle, indent=2)
        # Write the selected validation predictions because diagnostics should be reviewable chronologically.
        validation_prediction_frame.to_csv(validation_prediction_output_path, index=False)
        # Write the final test predictions because interval diagnostics should be reviewable chronologically.
        test_result["prediction_frame"].to_csv(test_prediction_output_path, index=False)
        # Log successful completion so the operator can distinguish a full run from a partial one.
        LOGGER.info("Quantile GBR baseline completed successfully.")
        # Print the compact summary so the operator can inspect the selected model immediately.
        print(json.dumps(summary_payload, indent=2))
        # Return success because the workflow completed without exceptions.
        return 0
    except Exception as exc:
        # Log the full exception because silent model failures would compromise baseline comparison.
        LOGGER.exception("Quantile GBR baseline failed: %s", exc)
        # Return failure so shell execution can detect the issue.
        return 1


# Execute the workflow only when the script is run directly so imports remain side-effect free.
if __name__ == "__main__":
    # Exit with the workflow status so terminal execution receives an accurate process code.
    sys.exit(main())