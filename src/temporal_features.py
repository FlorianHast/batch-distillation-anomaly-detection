from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ============================================================
# Metadata columns
# ============================================================

METADATA_COLUMNS = [
    "identifier",
    "window",
    "window_start",
    "window_end",
    "phase",
    "batch",
    "operating_point",
    "experiment_type",
    "experiment",
    "anomaly_label",
]


# ============================================================
# Load window features
# ============================================================

def load_window_features():

    window_path = (
        PROCESSED_DIR / "window_features.parquet"
    )

    if not window_path.exists():
        raise FileNotFoundError(
            f"Window features not found: {window_path}\n"
            "Run feature_engineering.py first."
        )

    return pd.read_parquet(
        window_path
    )


# ============================================================
# Identify model features
# ============================================================

def get_model_features(window_features):

    model_features = [
        column
        for column in window_features.columns
        if column not in METADATA_COLUMNS
    ]

    return model_features


# ============================================================
# Create temporal features
# ============================================================

def create_temporal_features(
    window_features,
    model_features,
):

    # --------------------------------------------------------
    # Sort windows within each experiment
    # --------------------------------------------------------

    window_features = (
        window_features
        .sort_values(
            ["identifier", "window"]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Previous-window features
    # --------------------------------------------------------

    previous_features = (
        window_features
        .groupby(
            "identifier",
            sort=False
        )[model_features]
        .shift(1)
    )

    previous_features.columns = [
        f"{column}_prev"
        for column in model_features
    ]

    # --------------------------------------------------------
    # Delta features
    # --------------------------------------------------------

    delta_values = (
        window_features[model_features].to_numpy()
        - previous_features.to_numpy()
    )

    delta_features = pd.DataFrame(
        delta_values,
        columns=[
            f"{column}_delta"
            for column in model_features
        ],
        index=window_features.index,
    )

    # --------------------------------------------------------
    # Current + previous + delta
    # --------------------------------------------------------

    temporal_features = pd.concat(
        [
            window_features,
            previous_features,
            delta_features,
        ],
        axis=1,
    )

    return temporal_features


# ============================================================
# Validation
# ============================================================

def validate_temporal_features(
    window_features,
    temporal_features,
    model_features,
):

    expected_columns = (
        len(window_features.columns)
        + 2 * len(model_features)
    )

    actual_columns = (
        len(temporal_features.columns)
    )

    print("\nTemporal feature matrix:")
    print(
        temporal_features.shape
    )

    print(
        f"Current features:  "
        f"{len(model_features)}"
    )

    print(
        f"Previous features: "
        f"{len(model_features)}"
    )

    print(
        f"Delta features:    "
        f"{len(model_features)}"
    )

    print(
        f"\nExpected total columns: "
        f"{expected_columns}"
    )

    print(
        f"Actual total columns:   "
        f"{actual_columns}"
    )

    if actual_columns != expected_columns:
        raise ValueError(
            "Temporal feature count does not "
            "match the expected count."
        )

    # --------------------------------------------------------
    # First window of each experiment
    # --------------------------------------------------------

    first_windows = (
        temporal_features
        .groupby("identifier")
        .head(1)
    )

    previous_nan_count = (
        first_windows[
            [
                f"{column}_prev"
                for column in model_features
            ]
        ]
        .isna()
        .all(axis=1)
        .sum()
    )

    delta_nan_count = (
        first_windows[
            [
                f"{column}_delta"
                for column in model_features
            ]
        ]
        .isna()
        .all(axis=1)
        .sum()
    )

    print(
        f"\nExperiments with no previous window: "
        f"{previous_nan_count}"
    )

    print(
        f"Experiments with no delta for first window: "
        f"{delta_nan_count}"
    )

    # There should be exactly one first window per experiment.
    expected_experiments = (
        temporal_features[
            "identifier"
        ].nunique()
    )

    if previous_nan_count != expected_experiments:
        raise ValueError(
            "Unexpected number of experiments without "
            "a previous window."
        )

    if delta_nan_count != expected_experiments:
        raise ValueError(
            "Unexpected number of experiments without "
            "a delta for the first window."
        )


# ============================================================
# Save temporal features
# ============================================================

def save_temporal_features(
    temporal_features
):

    output_path = (
        PROCESSED_DIR
        / "temporal_features.parquet"
    )

    temporal_features.to_parquet(
        output_path,
        index=False,
    )

    print(
        f"\nData saved to: {output_path}"
    )


# ============================================================
# Main pipeline
# ============================================================

def run_temporal_feature_engineering():

    print("=" * 70)
    print("TEMPORAL FEATURE ENGINEERING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    window_features = (
        load_window_features()
    )

    # --------------------------------------------------------
    # Identify model features
    # --------------------------------------------------------

    model_features = (
        get_model_features(
            window_features
        )
    )

    print(
        f"Number of model features: "
        f"{len(model_features)}"
    )

    # --------------------------------------------------------
    # Create temporal features
    # --------------------------------------------------------

    temporal_features = (
        create_temporal_features(
            window_features,
            model_features,
        )
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_temporal_features(
        window_features,
        temporal_features,
        model_features,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_temporal_features(
        temporal_features
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    run_temporal_feature_engineering()