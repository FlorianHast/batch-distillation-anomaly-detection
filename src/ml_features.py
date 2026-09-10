from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OPERATION_DATA_PATH = (
    PROCESSED_DIR / "operation_data.parquet"
)

TEMPORAL_FEATURES_PATH = (
    PROCESSED_DIR / "temporal_features.parquet"
)

OUTPUT_PATH = (
    PROCESSED_DIR / "ml_features.parquet"
)


# =========================================================
# Configuration
# =========================================================

FS = 1.0
WINDOW_LENGTH = 60

# Variables for which frequency-domain features are meaningful
FREQUENCY_VARIABLES = [
    # Temperatures
    "T701",
    "T702",
    "T703",
    "T704",
    "T705",
    "T706",
    "T708",
    "T709",
    "T711",
    "T712",

    # Pressure
    "P701",
    "P702",
    "PDI701",
    "PDI702",
    "PY23",

    # Flow
    "FT703",
    "FT704",
    "FYI702",

    # Level
    "LS701",
    "LS702",
]

GROUP_COLUMNS = [
    "identifier",
    "window",
]

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

FINAL_METADATA_ORDER = [
    "identifier",
    "batch",
    "operating_point",
    "experiment_type",
    "experiment",
    "window",
    "anomaly_label",
]


# =========================================================
# Load data
# =========================================================

def load_input_data():
    """
    Load operation data and temporal features.
    """

    operation_data = pd.read_parquet(
        OPERATION_DATA_PATH
    )

    temporal_features = pd.read_parquet(
        TEMPORAL_FEATURES_PATH
    )

    print("Loaded input datasets")
    print("---------------------")
    print(
        f"Operation data:      {operation_data.shape}"
    )
    print(
        f"Temporal features:   {temporal_features.shape}"
    )

    return operation_data, temporal_features


# =========================================================
# Validation of input data
# =========================================================

def validate_input_data(
    operation_data,
    temporal_features,
):
    """
    Validate that all required columns are available.
    """

    required_operation_columns = (
        GROUP_COLUMNS + FREQUENCY_VARIABLES
    )

    missing_operation_columns = [
        column
        for column in required_operation_columns
        if column not in operation_data.columns
    ]

    if missing_operation_columns:
        raise ValueError(
            "Missing columns in operation_data:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_operation_columns
            )
        )

    required_temporal_columns = (
        GROUP_COLUMNS + METADATA_COLUMNS
    )

    missing_temporal_columns = [
        column
        for column in required_temporal_columns
        if column not in temporal_features.columns
    ]

    if missing_temporal_columns:
        raise ValueError(
            "Missing columns in temporal_features:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_temporal_columns
            )
        )

    print("\nInput validation passed.")


# =========================================================
# Frequency-domain feature calculation
# =========================================================

def calculate_frequency_features(signal):
    """
    Calculate frequency-domain features for one complete
    60-second signal sampled at 1 Hz.

    Returns:
        dict containing:
            - total_power
            - dominant_power_ratio
            - spectral_entropy
    """

    signal = np.asarray(
        signal,
        dtype=float,
    )

    # Only complete, valid windows
    if (
        len(signal) != WINDOW_LENGTH
        or np.isnan(signal).any()
    ):
        return {
            "total_power": np.nan,
            "dominant_power_ratio": np.nan,
            "spectral_entropy": np.nan,
        }

    # Remove DC component
    signal = signal - np.mean(signal)

    # FFT
    fft_values = np.fft.rfft(signal)

    frequencies = np.fft.rfftfreq(
        WINDOW_LENGTH,
        d=1 / FS,
    )

    # Power spectrum
    power = (
        np.abs(fft_values) ** 2
        / WINDOW_LENGTH
    )

    # Remove DC component
    frequencies = frequencies[1:]
    power = power[1:]

    # Total spectral power
    total_power = np.sum(power)

    # Constant signal
    if total_power == 0:
        return {
            "total_power": 0.0,
            "dominant_power_ratio": np.nan,
            "spectral_entropy": 0.0,
        }

    # Dominant frequency component
    dominant_idx = np.argmax(power)
    dominant_power = power[dominant_idx]

    dominant_power_ratio = (
        dominant_power / total_power
    )

    # Spectral entropy
    power_distribution = (
        power / total_power
    )

    spectral_entropy = -np.sum(
        power_distribution
        * np.log2(
            power_distribution + 1e-12
        )
    )

    return {
        "total_power": total_power,
        "dominant_power_ratio": dominant_power_ratio,
        "spectral_entropy": spectral_entropy,
    }


# =========================================================
# Create frequency features
# =========================================================

def create_frequency_features(operation_data):
    """
    Calculate frequency-domain features for every
    frequency variable and every experiment window.
    """

    frequency_feature_list = []

    print("\nCreating frequency features...")
    print(
        f"Variables: {len(FREQUENCY_VARIABLES)}"
    )

    for variable in FREQUENCY_VARIABLES:

        grouped = operation_data.groupby(
            GROUP_COLUMNS,
            sort=False,
        )[variable]

        for (
            identifier,
            window,
        ), signal in grouped:

            features = calculate_frequency_features(
                signal.to_numpy()
            )

            frequency_feature_list.append(
                {
                    "identifier": identifier,
                    "window": window,
                    "variable": variable,
                    **features,
                }
            )

    frequency_features_all = pd.DataFrame(
        frequency_feature_list
    )

    print(
        "Frequency feature rows:",
        len(frequency_features_all),
    )

    return frequency_features_all


# =========================================================
# Pivot frequency features
# =========================================================

def pivot_frequency_features(
    frequency_features_all,
):
    """
    Convert long-format frequency features into
    one row per experiment window.
    """

    frequency_features_ml = (
        frequency_features_all
        .pivot(
            index=GROUP_COLUMNS,
            columns="variable",
            values=[
                "total_power",
                "dominant_power_ratio",
                "spectral_entropy",
            ],
        )
    )

    # Flatten MultiIndex columns
    frequency_features_ml.columns = [
        f"{feature}_{variable}"
        for feature, variable
        in frequency_features_ml.columns
    ]

    frequency_features_ml = (
        frequency_features_ml
        .reset_index()
    )

    print(
        "Frequency feature matrix:",
        frequency_features_ml.shape,
    )

    return frequency_features_ml


# =========================================================
# Prepare temporal features
# =========================================================

def prepare_temporal_features(
    temporal_features,
):
    """
    Select identifier/window and all model features
    from the temporal feature dataset.
    """

    temporal_feature_columns = [
        column
        for column in temporal_features.columns
        if column not in METADATA_COLUMNS
    ]

    temporal_features_ml = (
        temporal_features[
            GROUP_COLUMNS
            + temporal_feature_columns
        ]
        .copy()
    )

    print(
        "Temporal model features:",
        len(temporal_feature_columns),
    )

    return temporal_features_ml


# =========================================================
# Merge temporal + frequency features
# =========================================================

def merge_feature_sets(
    temporal_features_ml,
    frequency_features_ml,
):
    """
    Merge temporal and frequency-domain features
    one-to-one by experiment and window.
    """

    ml_features = (
        temporal_features_ml
        .merge(
            frequency_features_ml,
            on=GROUP_COLUMNS,
            how="inner",
            validate="one_to_one",
        )
    )

    print(
        "Merged feature matrix:",
        ml_features.shape,
    )

    return ml_features


# =========================================================
# Add metadata and target
# =========================================================

def add_metadata(
    ml_features,
    temporal_features,
):
    """
    Add process metadata and anomaly target.
    """

    metadata = (
        temporal_features[
            [
                "identifier",
                "window",
                "batch",
                "operating_point",
                "experiment_type",
                "experiment",
                "anomaly_label",
            ]
        ]
        .copy()
    )

    metadata = metadata.drop_duplicates(
        subset=GROUP_COLUMNS
    )

    ml_features = (
        ml_features
        .merge(
            metadata,
            on=GROUP_COLUMNS,
            how="left",
            validate="one_to_one",
        )
    )

    return ml_features


# =========================================================
# Arrange columns
# =========================================================

def arrange_columns(ml_features):
    """
    Put metadata first and model features afterwards.
    """

    feature_columns = [
        column
        for column in ml_features.columns
        if column not in FINAL_METADATA_ORDER
    ]

    ml_features = ml_features[
        FINAL_METADATA_ORDER
        + feature_columns
    ]

    return ml_features


# =========================================================
# Final validation
# =========================================================

def validate_ml_features(
    ml_features,
):
    """
    Validate the final ML feature matrix.
    """

    model_feature_columns = [
        column
        for column in ml_features.columns
        if column not in FINAL_METADATA_ORDER
    ]

    frequency_columns = [
        column
        for column in model_feature_columns
        if (
            "total_power" in column
            or "dominant_power_ratio" in column
            or "spectral_entropy" in column
        )
    ]

    duplicate_windows = (
        ml_features
        .duplicated(
            subset=GROUP_COLUMNS
        )
        .sum()
    )

    missing_target = (
        ml_features["anomaly_label"]
        .isna()
        .sum()
    )

    missing_metadata = (
        ml_features[
            [
                "batch",
                "operating_point",
                "experiment_type",
                "experiment",
            ]
        ]
        .isna()
        .sum()
    )

    missing_feature_values = (
        ml_features[
            model_feature_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    expected_frequency_features = (
        len(FREQUENCY_VARIABLES) * 3
    )

    print("\n" + "=" * 70)
    print("FINAL ML DATASET VALIDATION")
    print("=" * 70)

    print(
        "\nShape:",
        ml_features.shape,
    )

    print(
        "ML features:",
        len(model_feature_columns),
    )

    print(
        "Frequency features:",
        len(frequency_columns),
    )

    print(
        "Frequency variables:",
        len(FREQUENCY_VARIABLES),
    )

    print(
        "Expected frequency features:",
        expected_frequency_features,
    )

    print(
        "Duplicate windows:",
        duplicate_windows,
    )

    print(
        "Missing target:",
        missing_target,
    )

    print("\nMissing metadata:")
    print(missing_metadata)

    print("\nTarget distribution:")
    print(
        ml_features[
            "anomaly_label"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\nMissing feature values:",
        missing_feature_values,
    )

    print("\nFrequency feature columns:")
    for column in frequency_columns:
        print(f"  {column}")

    # Hard validation checks
    if duplicate_windows != 0:
        raise ValueError(
            "Duplicate identifier/window combinations found."
        )

    if missing_target != 0:
        raise ValueError(
            "Missing anomaly labels found."
        )

    if missing_metadata.sum() != 0:
        raise ValueError(
            "Missing metadata values found."
        )

    if len(frequency_columns) != expected_frequency_features:
        raise ValueError(
            "Unexpected number of frequency features. "
            f"Expected {expected_frequency_features}, "
            f"got {len(frequency_columns)}."
        )

    print(
        "\nAll final validation checks passed."
    )


# =========================================================
# Save
# =========================================================

def save_ml_features(
    ml_features,
):
    """
    Save final ML feature matrix to parquet.
    """

    ml_features.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved to:\n{OUTPUT_PATH}"
    )


# =========================================================
# Main pipeline
# =========================================================

def run_ml_feature_engineering():
    """
    Complete ML feature-engineering pipeline.
    """

    print("=" * 70)
    print("ML FEATURE ENGINEERING")
    print("=" * 70)

    # Load
    operation_data, temporal_features = (
        load_input_data()
    )

    # Validate inputs
    validate_input_data(
        operation_data,
        temporal_features,
    )

    # Frequency features
    frequency_features_all = (
        create_frequency_features(
            operation_data
        )
    )

    frequency_features_ml = (
        pivot_frequency_features(
            frequency_features_all
        )
    )

    # Temporal features
    temporal_features_ml = (
        prepare_temporal_features(
            temporal_features
        )
    )

    # Merge
    ml_features = merge_feature_sets(
        temporal_features_ml,
        frequency_features_ml,
    )

    # Metadata / target
    ml_features = add_metadata(
        ml_features,
        temporal_features,
    )

    # Column order
    ml_features = arrange_columns(
        ml_features
    )

    # Validation
    validate_ml_features(
        ml_features
    )

    # Save
    save_ml_features(
        ml_features
    )

    return ml_features


# =========================================================
# Script entry point
# =========================================================

if __name__ == "__main__":
    run_ml_feature_engineering()