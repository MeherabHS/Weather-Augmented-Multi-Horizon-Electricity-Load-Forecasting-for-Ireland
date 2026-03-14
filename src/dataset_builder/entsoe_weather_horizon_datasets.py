# Import json so build metadata can be persisted.
import json

# Import logging so pipeline diagnostics remain visible.
import logging

# Import sys so the script can return explicit exit codes.
import sys

# Import datetime utilities for reproducible metadata timestamps.
from datetime import datetime
from datetime import timezone

# Import Path for deterministic filesystem operations.
from pathlib import Path

# Import pandas for time-series dataset construction.
import pandas as pd


LOGGER = logging.getLogger("entsoe_weather_horizon_datasets")


INPUT_TABLE = Path("data/modeling_weather/entsoe/ireland_load_modeling_table_with_weather.csv")

OUTPUT_DIR = Path("data/horizons_weather/entsoe")


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_directory(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def read_table(path: Path) -> pd.DataFrame:

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Modeling table is empty.")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    df.sort_values("timestamp_utc", inplace=True)

    df.reset_index(drop=True, inplace=True)

    return df


def build_horizon(df: pd.DataFrame, lead_hours: int, label: str) -> pd.DataFrame:

    result = df.copy()

    target_column = f"load_target_{label}"

    availability_column = f"target_available_{label}"

    result[target_column] = result["load_mw"].shift(-lead_hours)

    result[availability_column] = result[target_column].notna().astype(int)

    return result


def main():

    configure_logging()

    try:

        ensure_directory(OUTPUT_DIR)

        df = read_table(INPUT_TABLE)

        horizons = [
            ("t_plus_1", 1),
            ("t_plus_24", 24),
            ("t_plus_168", 168),
        ]

        metadata = []

        for label, hours in horizons:

            horizon_df = build_horizon(df, hours, label)

            output_path = OUTPUT_DIR / f"ireland_load_{label}.csv"

            horizon_df.to_csv(output_path, index=False)

            summary = {
                "horizon_label": label,
                "lead_hours": hours,
                "row_count": int(len(horizon_df)),
                "available_target_rows": int(horizon_df[f"target_available_{label}"].sum()),
                "missing_target_rows": int(
                    len(horizon_df) - horizon_df[f"target_available_{label}"].sum()
                ),
                "start_utc": str(horizon_df["timestamp_utc"].min()),
                "end_utc": str(horizon_df["timestamp_utc"].max()),
            }

            metadata.append(summary)

            LOGGER.info("Built weather horizon dataset: %s", output_path)

        metadata_path = OUTPUT_DIR / "weather_horizon_metadata.json"

        with metadata_path.open("w") as f:
            json.dump(
                {
                    "generated_at_utc": utc_now_iso(),
                    "horizons": metadata,
                },
                f,
                indent=2,
            )

        print(
            json.dumps(
                {
                    "metadata_output_path": str(metadata_path),
                    "horizons": metadata,
                },
                indent=2,
            )
        )

        LOGGER.info("Weather horizon dataset build completed successfully.")

        return 0

    except Exception as exc:

        LOGGER.exception("Weather horizon build failed: %s", exc)

        return 1


if __name__ == "__main__":
    sys.exit(main())