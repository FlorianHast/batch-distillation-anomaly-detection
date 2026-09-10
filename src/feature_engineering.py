from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

WINDOW_SIZE = 60


# ============================================================
# Selected process variables
# ============================================================

SELECTED_VARIABLES = [

    # Temperature
    "T701", "T702", "T703", "T704", "T705",
    "T706", "T708", "T709", "T711", "T712",

    # Pressure
    "P701", "P702", "PDI701", "PDI702", "PY23",

    # Flow
    "FT703", "FT704", "FYI702",

    # Level
    "LS701", "LS702",

    # Heating / power
    "H002", "H701", "H702", "H704", "H706", "H708",

    # Vacuum
    "P301", "TV1",

    # Nitrogen
    "AV709",

    # Cooling
    "AV716",
]


# ============================================================
# Feature groups
# ============================================================

TEMPERATURE_SENSORS = [
    "T701", "T702", "T703", "T704", "T705",
    "T706", "T708", "T709", "T711", "T712",
]

CONTINUOUS_PROCESS_VARIABLES = [
    *TEMPERATURE_SENSORS,

    # Pressure
    "P701", "P702", "PDI701", "PDI702", "PY23",

    # Flow
    "FT703", "FT704", "FYI702",

    # Level
    "LS701", "LS702",
]

ACTUATOR_VARIABLES = [
    # Heating / power
    "H002", "H701", "H702", "H704", "H706", "H708",

    # Vacuum
    "P301", "TV1",

    # Nitrogen
    "AV709",

    # Cooling
    "AV716",
]

# All variables used for generic window statistics and slopes
FEATURE_VARIABLES = (
    CONTINUOUS_PROCESS_VARIABLES
    + ACTUATOR_VARIABLES
)


# ============================================================
# Physically meaningful temperature differences
# ============================================================

DIFFERENCE_PAIRS = [
    ("T704", "T706"),
    ("T706", "T708"),
    ("T704", "T708"),
    ("T701", "T702"),
    ("T702", "T703"),
    ("T711", "T712"),
    ("T703", "T709"),
]


# ============================================================
# Metadata
# ============================================================

METADATA_COLUMNS = [
    "identifier",
    "phase",
    "batch",
    "operating_point",
    "experiment_type",
    "experiment",
    "window",
    "window_start",
    "window_end",
    "anomaly_label",
]

FEATURE_KEY = [
    "identifier",
    "window",
]


# ============================================================
# Helper function: slope
# ============================================================

def calculate_slope(values):
    """
    Calculate the linear slope within one 60-second window.
    """

    x = np.arange(len(values))

    return np.polyfit(
        x,
        values,
        1,
    )[0]


# ============================================================
# Load operation data
# ============================================================

def load_operation_data():
    """
    Load the intermediate operation_data dataset created
    by preprocessing.py.
    """

    operation_path = (
        PROCESSED_DIR / "operation_data.parquet"
    )

    if not operation_path.exists():
        raise FileNotFoundError(
            f"Operation data not found: {operation_path}\n"
            "Run preprocessing.py first."
        )

    operation_data = pd.read_parquet(
        operation_path
    )

    return operation_data


# ============================================================
# Validate required columns
# ============================================================

def validate_columns(operation_data):
    """
    Validate that all metadata and process variables required
    for feature engineering are available.
    """

    required_columns = (
        METADATA_COLUMNS
        + SELECTED_VARIABLES
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in operation_data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )


# ============================================================
# Keep complete 60-second windows
# ============================================================

def get_complete_windows(operation_data):
    """
    Keep only windows containing exactly 60 observations.
    """

    complete_windows = (
        operation_data
        .groupby(FEATURE_KEY)
        .size()
        .loc[
            lambda x: x == WINDOW_SIZE
        ]
        .index
    )

    complete_data = (
        operation_data
        .set_index(FEATURE_KEY)
        .loc[complete_windows]
        .reset_index()
    )

    return complete_data, complete_windows


# ============================================================
# Create window metadata
# ============================================================

def create_window_metadata(complete_data):
    """
    Extract one metadata row per complete window.
    """

    return (
        complete_data[
            METADATA_COLUMNS
        ]
        .drop_duplicates(
            subset=FEATURE_KEY
        )
    )


# ============================================================
# 1. Process statistics
#
# Includes all 30 process variables.
# ============================================================

def create_process_features(complete_data):

    process_features = (
        complete_data
        .groupby(FEATURE_KEY)[
            FEATURE_VARIABLES
        ]
        .agg([
            "mean",
            "std",
            "min",
            "max",
        ])
    )

    process_features.columns = [
        f"{sensor}_{stat}"
        for sensor, stat
        in process_features.columns
    ]

    return (
        process_features
        .reset_index()
    )


# ============================================================
# 2. Window slopes
#
# Includes all 30 process variables.
# ============================================================

def create_window_slopes(complete_data):

    window_slopes = (
        complete_data
        .groupby(FEATURE_KEY)[
            FEATURE_VARIABLES
        ]
        .agg(calculate_slope)
    )

    window_slopes.columns = [
        f"{sensor}_slope"
        for sensor
        in window_slopes.columns
    ]

    return (
        window_slopes
        .reset_index()
    )


# ============================================================
# 3. Constant indicators
#
# Applied to actuator/state variables.
# ============================================================

def create_window_constants(complete_data):

    window_constants = (
        complete_data
        .groupby(FEATURE_KEY)[
            ACTUATOR_VARIABLES
        ]
        .std()
        .eq(0)
        .astype("int8")
    )

    window_constants.columns = [
        f"{sensor}_constant"
        for sensor in window_constants.columns
    ]

    return (
        window_constants
        .reset_index()
    )


# ============================================================
# 4. Temperature correlations
# ============================================================

def create_window_correlations(complete_data):

    correlation_rows = []

    for (
        identifier,
        window,
    ), group in complete_data.groupby(
        FEATURE_KEY
    ):

        correlations = (
            group[
                TEMPERATURE_SENSORS
            ]
            .corr()
        )

        row = {
            "identifier": identifier,
            "window": window,
        }

        for i, sensor_a in enumerate(
            TEMPERATURE_SENSORS
        ):

            for sensor_b in (
                TEMPERATURE_SENSORS[i + 1:]
            ):

                row[
                    f"{sensor_a}_{sensor_b}_corr"
                ] = correlations.loc[
                    sensor_a,
                    sensor_b,
                ]

        correlation_rows.append(row)

    return pd.DataFrame(
        correlation_rows
    )


# ============================================================
# 5. Temperature-pair differences
# ============================================================

def create_difference_features(
    complete_data
):

    difference_columns = []

    for (
        sensor_a,
        sensor_b,
    ) in DIFFERENCE_PAIRS:

        column_name = (
            f"{sensor_a}_{sensor_b}_diff"
        )

        complete_data[column_name] = (
            complete_data[sensor_a]
            - complete_data[sensor_b]
        )

        difference_columns.append(
            column_name
        )

    # --------------------------------------------------------
    # Difference statistics
    # --------------------------------------------------------

    difference_features = (
        complete_data
        .groupby(FEATURE_KEY)[
            difference_columns
        ]
        .agg([
            "mean",
            "std",
            "min",
            "max",
        ])
    )

    difference_features.columns = [
        f"{column}_{stat}"
        for column, stat
        in difference_features.columns
    ]

    difference_features = (
        difference_features
        .reset_index()
    )

    # --------------------------------------------------------
    # Difference slopes
    # --------------------------------------------------------

    difference_slopes = (
        complete_data
        .groupby(FEATURE_KEY)[
            difference_columns
        ]
        .agg(calculate_slope)
    )

    difference_slopes.columns = [
        f"{column}_slope"
        for column
        in difference_slopes.columns
    ]

    difference_slopes = (
        difference_slopes
        .reset_index()
    )

    # --------------------------------------------------------
    # Combine difference features
    # --------------------------------------------------------

    difference_features = (
        difference_features
        .merge(
            difference_slopes,
            on=FEATURE_KEY,
            how="inner",
            validate="one_to_one",
        )
    )

    return difference_features


# ============================================================
# Combine all feature blocks
# ============================================================

def create_window_features(
    complete_data,
    window_metadata,
):

    process_features = (
        create_process_features(
            complete_data
        )
    )

    window_slopes = (
        create_window_slopes(
            complete_data
        )
    )

    window_constants = (
        create_window_constants(
            complete_data
        )
    )

    window_correlations = (
        create_window_correlations(
            complete_data
        )
    )

    difference_features = (
        create_difference_features(
            complete_data
        )
    )

    window_features = (
        process_features

        .merge(
            window_slopes,
            on=FEATURE_KEY,
            how="inner",
            validate="one_to_one",
        )

        .merge(
            window_constants,
            on=FEATURE_KEY,
            how="inner",
            validate="one_to_one",
        )

        .merge(
            window_correlations,
            on=FEATURE_KEY,
            how="inner",
            validate="one_to_one",
        )

        .merge(
            difference_features,
            on=FEATURE_KEY,
            how="inner",
            validate="one_to_one",
        )
    )

    # --------------------------------------------------------
    # Add metadata exactly once
    # --------------------------------------------------------

    window_features = (
        window_metadata
        .merge(
            window_features,
            on=FEATURE_KEY,
            how="inner",
            validate="one_to_one",
        )
    )

    return window_features


# ============================================================
# Validation
# ============================================================

def validate_window_features(
    window_features,
):

    duplicate_windows = (
        window_features
        .duplicated(
            subset=FEATURE_KEY
        )
        .sum()
    )

    missing_target = (
        window_features[
            "anomaly_label"
        ]
        .isna()
        .sum()
    )

    feature_columns = [
        column
        for column in window_features.columns
        if column not in METADATA_COLUMNS
    ]

    print("\nFinal window_features")
    print("---------------------")

    print(
        "Shape:",
        window_features.shape,
    )

    print(
        "Duplicate windows:",
        duplicate_windows,
    )

    print(
        "Missing target:",
        missing_target,
    )

    print("\nTarget distribution:")

    print(
        window_features[
            "anomaly_label"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\nNumber of ML features:",
        len(feature_columns),
    )

    if duplicate_windows != 0:
        raise ValueError(
            "Duplicate windows detected."
        )


# ============================================================
# Main pipeline
# ============================================================

def run_feature_engineering():

    print("=" * 70)
    print("WINDOW FEATURE ENGINEERING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    operation_data = load_operation_data()

    print(
        f"Rows: {len(operation_data):,}"
    )

    print(
        "Experiments:",
        operation_data[
            "identifier"
        ].nunique(),
    )

    print(
        "Windows:",
        operation_data[
            FEATURE_KEY
        ]
        .drop_duplicates()
        .shape[0],
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_columns(
        operation_data
    )

    # --------------------------------------------------------
    # Complete windows
    # --------------------------------------------------------

    (
        complete_data,
        complete_windows,
    ) = get_complete_windows(
        operation_data
    )

    print(
        f"Complete windows: "
        f"{len(complete_windows):,}"
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    window_metadata = (
        create_window_metadata(
            complete_data
        )
    )

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    window_features = (
        create_window_features(
            complete_data,
            window_metadata,
        )
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validate_window_features(
        window_features
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        PROCESSED_DIR
        / "window_features.parquet"
    )

    window_features.to_parquet(
        output_path,
        index=False,
    )

    print(
        f"\nData saved to: {output_path}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    run_feature_engineering()