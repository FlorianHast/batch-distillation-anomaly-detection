from pathlib import Path
import re

import pandas as pd


# ============================================================
# Configuration
# ============================================================

# Project root = parent directory of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

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
# Known scaling corrections
# ============================================================

SCALED_1000_COLUMNS = [
    "T701", "T702", "T703", "T704", "T706",
    "T708", "T709", "T711", "T712", "T705",
    "PY23", "TV1", "H702", "H706", "H708",
    "H704", "P701", "P702",
]

SCALED_1000_THRESHOLD = 500

SCALED_1000_SMALL_COLUMNS = [
    "FT703", "FT704", "PDI701", "PDI702",
]

SCALED_1000_SMALL_THRESHOLD = 1


# ============================================================
# Known timestamp error
# ============================================================

KNOWN_TIMESTAMP_ERROR = (
    "Shutdown/"
    "batch_dist_ternary_butan-1-ol+propan-2-ol+water/"
    "operating_point_013/"
    "train_normal/"
    "experiment_001"
)


# ============================================================
# Batch pattern
# ============================================================

BATCH_PATTERN = (
    r"(batch_dist_binary_ethanol\+propan-2-ol|"
    r"batch_dist_ternary_acetone\+butan-1-ol\+methanol|"
    r"batch_dist_ternary_butan-1-ol\+propan-2-ol\+water)"
)


def get_file_metadata(folder_name):
    records = []
    folder = DATA_DIR / folder_name

    for csv_path in folder.rglob("*.csv"):
        path_str = str(csv_path)

        phase_match = re.search(
            r"(Startup|Operation|Shutdown)",
            path_str,
            re.IGNORECASE,
        )

        batch_match = re.search(
            BATCH_PATTERN,
            path_str,
        )

        operating_point_match = re.search(
            r"operating_point_(\d+)",
            path_str,
        )

        experiment_match = re.search(
            r"(train_normal|test_anormal)_experiment_(\d+)",
            csv_path.stem,
        )

        if not (
            phase_match
            and batch_match
            and operating_point_match
            and experiment_match
        ):
            continue

        records.append({
            "file_path": str(csv_path),
            "phase": phase_match.group(1),
            "batch": batch_match.group(1),
            "operating_point": operating_point_match.group(0),
            "experiment_type": experiment_match.group(1),
            "experiment_number": (
                f"experiment_{experiment_match.group(2)}"
            ),
        })

    return pd.DataFrame(records)


# ============================================================
# Load time series
# ============================================================

def load_timeseries(folder_name):
    files = get_file_metadata(folder_name)

    print(f"{folder_name}: {len(files)} matching files")

    data = []

    for _, row in files.iterrows():
        df = pd.read_csv(row["file_path"])

        df["identifier"] = "/".join([
            row["phase"],
            row["batch"],
            row["operating_point"],
            row["experiment_type"],
            row["experiment_number"],
        ])

        data.append(df)

    if not data:
        raise ValueError(
            f"No CSV files found in {folder_name}"
        )

    return pd.concat(
        data,
        ignore_index=True,
    )

# ============================================================
# Load and merge raw data
# ============================================================

def load_raw_data() -> pd.DataFrame:
    """Load anomaly metadata, sensor data and actuator data and merge."""

    anomalies_ts = load_timeseries(
        "00_Timeseries_Label_Anomaly_Metadata"
    )

    sensors_ts = load_timeseries(
        "01_Timeseries_Sensors"
    )

    actuators_ts = load_timeseries(
        "02_Timeseries_Actuators"
    )

    # Missing anomaly labels are normal.
    anomalies_ts.loc[
        anomalies_ts["Time"].isna(),
        "Label (anomaly)",
    ] = 0

    anomalies_ts = anomalies_ts.rename(
        columns={"Label (anomaly)": "anomaly_label"}
    )

    experiments = (
        actuators_ts
        .merge(
            anomalies_ts[
                ["identifier", "Time", "anomaly_label"]
            ],
            on=["identifier", "Time"],
            how="left",
        )
        .merge(
            sensors_ts,
            on=["identifier", "Time"],
            how="left",
        )
    )

    experiments["anomaly_label"] = (
        experiments["anomaly_label"].fillna(0)
    )

    return experiments

# ============================================================
# Create elapsed time
# ============================================================

def add_elapsed_time(experiments: pd.DataFrame) -> pd.DataFrame:
    """Create elapsed seconds within each experiment."""

    experiments = experiments.copy()

    experiments["Time_td"] = pd.to_timedelta(
        experiments["Time"],
        errors="coerce",
    )

    experiments["time_diff"] = (
        experiments
        .groupby("identifier")["Time_td"]
        .diff()
    )

    experiments["time_diff_clean"] = (
        experiments["time_diff"]
        .where(
            ~(
                (experiments["identifier"] == KNOWN_TIMESTAMP_ERROR)
                & (experiments["time_diff"] < pd.Timedelta(0))
            ),
            pd.Timedelta(0),
        )
        .fillna(pd.Timedelta(0))
    )

    experiments["elapsed_time"] = (
        experiments
        .groupby("identifier")["time_diff_clean"]
        .cumsum()
        .dt.total_seconds()
    )

    return experiments.drop(
        columns=[
            "Time_td",
            "time_diff",
            "time_diff_clean",
        ]
    )


# ============================================================
# Add metadata
# ============================================================

def add_identifier_metadata(experiments: pd.DataFrame) -> pd.DataFrame:
    """Split the experiment identifier into process metadata columns."""

    experiments = experiments.copy()

    identifier_parts = experiments["identifier"].str.split(
        "/",
        expand=True,
    )

    identifier_parts.columns = [
        "phase",
        "batch",
        "operating_point",
        "experiment_type",
        "experiment",
    ]

    return pd.concat(
        [experiments, identifier_parts],
        axis=1,
    )


# ============================================================
# Correct known scaling errors
# ============================================================

def correct_scaling_errors(experiments: pd.DataFrame) -> pd.DataFrame:
    """Correct known values that were stored 1000x too large."""

    experiments = experiments.copy()

    for columns, threshold in [
        (
            SCALED_1000_COLUMNS,
            SCALED_1000_THRESHOLD,
        ),
        (
            SCALED_1000_SMALL_COLUMNS,
            SCALED_1000_SMALL_THRESHOLD,
        ),
    ]:
        mask = experiments[columns] > threshold

        experiments[columns] = experiments[columns].mask(
            mask,
            experiments[columns] / 1000,
        )

    return experiments


# ============================================================
# Keep relevant variables
# ============================================================

def select_process_variables(
    experiments: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and retain metadata plus the selected process variables."""

    missing_variables = [
        column
        for column in SELECTED_VARIABLES
        if column not in experiments.columns
    ]

    if missing_variables:
        raise ValueError(
            f"Selected variables not found: {missing_variables}"
        )

    metadata_columns = [
        "Time",
        "elapsed_time",
        "anomaly_label",
        "identifier",
        "phase",
        "batch",
        "operating_point",
        "experiment_type",
        "experiment",
    ]

    return experiments[
        metadata_columns + SELECTED_VARIABLES
    ].copy()


# ============================================================
# Create Operation-phase data
# ============================================================

def create_operation_data(
    experiments: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only the Operation phase and assign 60-second windows."""

    operation_data = experiments[
        experiments["phase"].str.lower() == "operation"
    ].copy()

    operation_data["window"] = (
        operation_data["elapsed_time"] // WINDOW_SIZE
    ).astype(int)

    operation_data["window_start"] = (
        operation_data["window"] * WINDOW_SIZE
    )

    operation_data["window_end"] = (
        operation_data["window_start"] + WINDOW_SIZE
    )

    return operation_data


# ============================================================
# Identify complete windows
# ============================================================

def get_complete_windows(
    operation_data: pd.DataFrame,
) -> pd.DataFrame:
    """Return experiment/window combinations containing exactly 60 samples."""

    window_counts = (
        operation_data
        .groupby(["identifier", "window"])
        .size()
        .reset_index(name="n_samples")
    )

    return window_counts[
        window_counts["n_samples"] == WINDOW_SIZE
    ][
        ["identifier", "window"]
    ]


# ============================================================
# Create raw-signal baseline
# ============================================================

def create_baseline_raw_windows(
    operation_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Flatten each complete 60-second window into one ML observation.

    The canonical window label is the label at the beginning of the
    window (the first row after chronological sorting).
    """

    complete_windows = get_complete_windows(operation_data)

    baseline_data = operation_data.merge(
        complete_windows,
        on=["identifier", "window"],
        how="inner",
    )

    baseline_data = baseline_data.sort_values(
        [
            "identifier",
            "window",
            "elapsed_time",
        ]
    )

    baseline_rows = []

    for (identifier, window), group in baseline_data.groupby(
        ["identifier", "window"],
        sort=False,
    ):
        if len(group) != WINDOW_SIZE:
            continue

        row = {
            "identifier": identifier,
            "batch": group["batch"].iloc[0],
            "operating_point": group["operating_point"].iloc[0],
            "experiment": group["experiment"].iloc[0],
            "experiment_type": group["experiment_type"].iloc[0],
            "window": window,
            "window_start": group["window_start"].iloc[0],
            "window_end": group["window_end"].iloc[0],

            # Canonical window label:
            # label at the beginning of the 60-second window.
            "anomaly_label": group["anomaly_label"].iloc[0],
        }

        for variable in SELECTED_VARIABLES:
            values = group[variable].to_numpy()

            for i, value in enumerate(values):
                row[f"{variable}_{i:02d}"] = value

        baseline_rows.append(row)

    baseline_raw_windows = pd.DataFrame(baseline_rows)

    # Observable anomaly target:
    # 0 = normal
    # 1 = detectable anomaly
    # NA = unobservable/blind fault
    baseline_raw_windows["observable_anomaly"] = (
        baseline_raw_windows["anomaly_label"]
        .map(
            {
                0: 0,
                1: pd.NA,
                2: 1,
                3: 1,
            }
        )
        .astype("Int64")
    )

    return baseline_raw_windows


# ============================================================
# Validation
# ============================================================

def validate_baseline(
    baseline_raw_windows: pd.DataFrame,
) -> dict:
    """Validate the expected structure of the raw-signal baseline."""

    raw_feature_columns = [
        column
        for column in baseline_raw_windows.columns
        if column.rsplit("_", 1)[-1].isdigit()
    ]

    expected_feature_count = (
        len(SELECTED_VARIABLES) * WINDOW_SIZE
    )

    if len(raw_feature_columns) != expected_feature_count:
        raise ValueError(
            "Unexpected raw feature count: "
            f"{len(raw_feature_columns)} "
            f"(expected {expected_feature_count})"
        )

    return {
        "shape": baseline_raw_windows.shape,
        "complete_windows": len(baseline_raw_windows),
        "experiments": baseline_raw_windows["identifier"].nunique(),
        "variables": len(SELECTED_VARIABLES),
        "raw_signal_features": len(raw_feature_columns),
        "expected_features": expected_feature_count,
        "target_distribution": (
            baseline_raw_windows["observable_anomaly"]
            .value_counts(dropna=False)
            .sort_index()
            .to_dict()
        ),
    }


# ============================================================
# Main pipeline
# ============================================================

def run_preprocessing() -> pd.DataFrame:
    """Run the complete preprocessing pipeline and save the baseline."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    experiments = load_raw_data()
    experiments = add_elapsed_time(experiments)
    experiments = add_identifier_metadata(experiments)
    experiments = correct_scaling_errors(experiments)
    experiments = select_process_variables(experiments)

    operation_data = create_operation_data(experiments)

    baseline_raw_windows = create_baseline_raw_windows(
        operation_data
    )

    baseline_path = (
        OUTPUT_DIR / "baseline_raw_windows.parquet"
    )

    baseline_raw_windows.to_parquet(
        baseline_path,
        index=False,
    )

    operation_path = (
        OUTPUT_DIR / "operation_data.parquet"
    )

    operation_data.to_parquet(
        operation_path,
        index=False,
    )

    validation = validate_baseline(
        baseline_raw_windows
    )

    print("=" * 70)
    print("RAW-SIGNAL BASELINE")
    print("=" * 70)
    print(f"Shape:                  {validation['shape']}")
    print(f"Complete windows:       {validation['complete_windows']}")
    print(f"Experiments:            {validation['experiments']}")
    print(f"Variables:              {validation['variables']}")
    print(f"Raw signal features:    {validation['raw_signal_features']}")
    print(
        f"Expected features:      "
        f"{len(SELECTED_VARIABLES)} × {WINDOW_SIZE} = "
        f"{validation['expected_features']}"
    )
    print(f"Saved to:               {baseline_path}")
    print("\nTarget distribution:")
    print(
        baseline_raw_windows["observable_anomaly"]
        .value_counts(dropna=False)
        .sort_index()
    )

    return baseline_raw_windows


if __name__ == "__main__":
    run_preprocessing()
