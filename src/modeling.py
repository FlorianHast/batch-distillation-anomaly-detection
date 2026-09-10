from pathlib import Path
import itertools

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42

# Default/reference model
DEFAULT_N_ESTIMATORS = 300

# Missing-value handling
MAX_MISSING_FRACTION = 0.95

# Hyperparameter optimization
PARAM_GRID = {
    "n_estimators": [200, 300, 500],
    "max_samples": ["auto", 0.5, 0.75, 1.0],
    "max_features": [0.5, 0.75, 1.0],
}

CONTAMINATION = "auto"

# Threshold optimization
N_THRESHOLDS = 1000


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

BASELINE_RAW_WINDOWS_PATH = (
    PROCESSED_DIR / "baseline_raw_windows.parquet"
)

WINDOW_FEATURES_PATH = (
    PROCESSED_DIR / "window_features.parquet"
)

TEMPORAL_FEATURES_PATH = (
    PROCESSED_DIR / "temporal_features.parquet"
)

ML_FEATURES_PATH = (
    PROCESSED_DIR / "ml_features.parquet"
)

EXPERIMENT_SPLIT_PATH = (
    PROCESSED_DIR / "experiment_split.parquet"
)

# Results
RESULTS_DIR = (
    PROCESSED_DIR / "model_results"
)

DEFAULT_RESULTS_PATH = (
    RESULTS_DIR / "default_model_results.parquet"
)

OPTIMIZATION_RESULTS_PATH = (
    RESULTS_DIR / "hyperparameter_results.parquet"
)

BEST_CONFIG_PATH = (
    RESULTS_DIR / "best_configurations.parquet"
)

THRESHOLD_RESULTS_PATH = (
    RESULTS_DIR / "threshold_results.parquet"
)

FINAL_TEST_RESULTS_PATH = (
    RESULTS_DIR / "final_test_results.parquet"
)


# ============================================================
# Model definitions
# ============================================================

MODEL_PATHS = {
    "baseline_raw_windows": BASELINE_RAW_WINDOWS_PATH,
    "window_features": WINDOW_FEATURES_PATH,
    "temporal_features": TEMPORAL_FEATURES_PATH,
    "ml_features": ML_FEATURES_PATH,
}


EXPECTED_FEATURE_COUNTS = {
    "baseline_raw_windows": 1800,
    "window_features": 240,
    "temporal_features": 720,
    "ml_features": 780,
}


# ============================================================
# Metadata columns
# ============================================================

# These columns are never passed to Isolation Forest.
#
# They are retained for:
# - experiment identification
# - process context
# - split assignment
# - labels
# - traceability
# - later analysis

METADATA_COLUMNS = [
    "identifier",
    "phase",
    "batch",
    "operating_point",
    "experiment",
    "experiment_type",
    "window",
    "window_start",
    "window_end",
    "anomaly_label",
    "observable_anomaly",
    "model_split",
]


# ============================================================
# Utility functions
# ============================================================

def get_feature_columns(df):
    """
    Return only actual ML feature columns.
    """

    return [
        column
        for column in df.columns
        if column not in METADATA_COLUMNS
    ]


def validate_feature_columns(
    model_name,
    df,
):
    """
    Validate expected feature count.
    """

    feature_columns = get_feature_columns(df)

    expected_count = EXPECTED_FEATURE_COUNTS[
        model_name
    ]

    if len(feature_columns) != expected_count:
        raise ValueError(
            f"{model_name}: expected "
            f"{expected_count} features, "
            f"found {len(feature_columns)}."
        )

    return feature_columns


def check_infinite_values(
    df,
    feature_columns,
    dataset_name,
):
    """
    Check for infinite numerical values.
    """

    numeric_features = (
        df[feature_columns]
        .select_dtypes(include=np.number)
    )

    n_inf = np.isinf(
        numeric_features
    ).sum().sum()

    if n_inf > 0:
        raise ValueError(
            f"{dataset_name}: "
            f"{n_inf:,} infinite values found."
        )


# ============================================================
# Load datasets
# ============================================================

def load_datasets():
    """
    Load all four feature datasets and the locked split.
    """

    print("=" * 80)
    print("LOADING MODELING DATA")
    print("=" * 80)

    datasets = {}

    for model_name, path in MODEL_PATHS.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Required dataset not found:\n{path}"
            )

        datasets[model_name] = pd.read_parquet(
            path
        )

        print(
            f"{model_name:25s}: "
            f"{datasets[model_name].shape}"
        )

    if not EXPERIMENT_SPLIT_PATH.exists():
        raise FileNotFoundError(
            "Locked experiment split not found:\n"
            f"{EXPERIMENT_SPLIT_PATH}"
        )

    experiment_split = pd.read_parquet(
        EXPERIMENT_SPLIT_PATH
    )

    print(
        f"\nExperiment split:        "
        f"{experiment_split.shape}"
    )

    return datasets, experiment_split


# ============================================================
# Validate experiment split
# ============================================================

def validate_experiment_split(
    experiment_split,
):
    """
    Validate the locked experiment split.
    """

    required_columns = [
        "identifier",
        "batch",
        "operating_point",
        "experiment_class",
        "split",
    ]

    missing = [
        column
        for column in required_columns
        if column not in experiment_split.columns
    ]

    if missing:
        raise ValueError(
            "Experiment split is missing columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    if experiment_split[
        "identifier"
    ].duplicated().any():

        raise ValueError(
            "Duplicate experiment identifiers "
            "found in experiment_split."
        )

    allowed_splits = {
        "train",
        "validation",
        "final_test",
        "excluded",
    }

    actual_splits = set(
        experiment_split["split"].dropna().unique()
    )

    unexpected = actual_splits - allowed_splits

    if unexpected:
        raise ValueError(
            f"Unexpected split labels: {unexpected}"
        )

    print("\nLocked experiment split:")
    print(
        experiment_split["split"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )


# ============================================================
# Assign model split
# ============================================================

def assign_model_split(
    df,
    experiment_split,
):
    """
    Apply the locked experiment-level split.

    Normal training experiments are assigned to train.

    Validation, final_test and excluded assignments
    come from experiment_split.parquet.
    """

    df = df.copy()

    if df["identifier"].isna().any():
        raise ValueError(
            "Missing experiment identifiers found."
        )

    split_mapping = (
        experiment_split
        .set_index("identifier")["split"]
    )

    mapped_split = df[
        "identifier"
    ].map(split_mapping)

    # Normal training experiments
    train_mask = (
        df["experiment_type"]
        == "train_normal"
    )

    df["model_split"] = pd.Series(
        pd.NA,
        index=df.index,
        dtype="string",
    )

    df.loc[
        train_mask,
        "model_split"
    ] = "train"

    # Apply locked split
    mapped_mask = mapped_split.notna()

    df.loc[
        mapped_mask,
        "model_split"
    ] = mapped_split.loc[
        mapped_mask
    ].astype("string")

    return df


# ============================================================
# Observable anomaly target
# ============================================================

def add_observable_anomaly_target(df):
    """
    Convert anomaly_label into the binary ML target.

    0 = normal
    1 = observable anomaly
    NaN = excluded

    Original labels:
        0 = normal
        1 = blind / unobservable fault
        2 = observable anomaly
        3 = observable after-effect
    """

    df = df.copy()

    if "anomaly_label" not in df.columns:
        raise ValueError(
            "'anomaly_label' column not found."
        )

    df["observable_anomaly"] = np.nan

    df.loc[
        df["anomaly_label"] == 0,
        "observable_anomaly"
    ] = 0

    df.loc[
        df["anomaly_label"].isin([2, 3]),
        "observable_anomaly"
    ] = 1

    return df


# ============================================================
# Prepare all datasets
# ============================================================

def prepare_datasets(
    datasets,
    experiment_split,
):
    """
    Apply split assignment and target creation
    to all four model datasets.
    """

    prepared = {}

    for model_name, df in datasets.items():

        df = assign_model_split(
            df,
            experiment_split,
        )

        df = add_observable_anomaly_target(
            df
        )

        validate_feature_columns(
            model_name,
            df,
        )

        feature_columns = get_feature_columns(
            df
        )

        check_infinite_values(
            df,
            feature_columns,
            model_name,
        )

        prepared[model_name] = df

    return prepared


# ============================================================
# Create train / validation / final-test datasets
# ============================================================

def create_model_datasets(df):
    """
    Create the three ML datasets.

    TRAIN:
        Only normal windows from train experiments.

    VALIDATION:
        All validation windows except blind-phase windows.

    FINAL TEST:
        All final-test windows except blind-phase windows.
    """

    train = df[
        (df["model_split"] == "train")
        & (df["observable_anomaly"] == 0)
    ].copy()

    validation = df[
        df["model_split"] == "validation"
    ].copy()

    validation = validation[
        validation["observable_anomaly"].notna()
    ].copy()

    final_test = df[
        df["model_split"] == "final_test"
    ].copy()

    final_test = final_test[
        final_test["observable_anomaly"].notna()
    ].copy()

    return train, validation, final_test


# ============================================================
# Prepare numerical feature matrices
# ============================================================

def prepare_features(
    model_name,
    train_df,
    evaluation_df,
):
    """
    Prepare feature matrices.

    Processing is fitted on TRAINING data only:

    1. Identify features.
    2. Remove features with >95% missingness.
    3. Fit median imputer.
    4. Transform evaluation data.

    Returns:
        X_train
        X_evaluation
        features_used
        imputer
    """

    feature_columns = validate_feature_columns(
        model_name,
        train_df,
    )

    check_infinite_values(
        train_df,
        feature_columns,
        f"{model_name} training",
    )

    check_infinite_values(
        evaluation_df,
        feature_columns,
        f"{model_name} evaluation",
    )

    # --------------------------------------------------------
    # Missing-value filtering
    # --------------------------------------------------------

    missing_fraction = (
        train_df[feature_columns]
        .isna()
        .mean()
    )

    features_to_remove = (
        missing_fraction
        > MAX_MISSING_FRACTION
    )

    features_used = (
        missing_fraction[
            ~features_to_remove
        ]
        .index
        .tolist()
    )

    if not features_used:
        raise ValueError(
            f"{model_name}: no features remain "
            "after missing-value filtering."
        )

    # --------------------------------------------------------
    # Raw matrices
    # --------------------------------------------------------

    X_train_raw = train_df[
        features_used
    ]

    X_evaluation_raw = evaluation_df[
        features_used
    ]

    # --------------------------------------------------------
    # Median imputation
    #
    # FIT ONLY ON TRAINING DATA
    # --------------------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train = imputer.fit_transform(
        X_train_raw
    )

    X_evaluation = imputer.transform(
        X_evaluation_raw
    )

    return (
        X_train,
        X_evaluation,
        features_used,
        imputer,
    )


# ============================================================
# Default Isolation Forest evaluation
# ============================================================

def run_default_models(
    model_datasets,
):
    """
    Run one default Isolation Forest per feature set.

    These results are a reference point before
    hyperparameter optimization.
    """

    print("\n")
    print("=" * 80)
    print("DEFAULT ISOLATION FOREST EVALUATION")
    print("=" * 80)

    results = []

    for model_name, data in model_datasets.items():

        print(
            f"\nRunning: {model_name}"
        )

        train_df = data["train"]
        validation_df = data["validation"]

        (
            X_train,
            X_validation,
            features_used,
            imputer,
        ) = prepare_features(
            model_name,
            train_df,
            validation_df,
        )

        y_validation = (
            validation_df[
                "observable_anomaly"
            ]
            .astype(int)
            .to_numpy()
        )

        model = IsolationForest(
            n_estimators=DEFAULT_N_ESTIMATORS,
            contamination=CONTAMINATION,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        model.fit(X_train)

        anomaly_scores = (
            -model.score_samples(
                X_validation
            )
        )

        auprc = average_precision_score(
            y_validation,
            anomaly_scores,
        )

        y_pred = (
            model.predict(X_validation)
            == -1
        ).astype(int)

        f1 = f1_score(
            y_validation,
            y_pred,
            zero_division=0,
        )

        precision = precision_score(
            y_validation,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_validation,
            y_pred,
            zero_division=0,
        )

        tn, fp, fn, tp = confusion_matrix(
            y_validation,
            y_pred,
            labels=[0, 1],
        ).ravel()

        results.append({
            "model": model_name,
            "n_train": len(train_df),
            "n_validation": len(validation_df),
            "n_features_original": len(
                get_feature_columns(
                    train_df
                )
            ),
            "n_features_used": len(
                features_used
            ),
            "AUPRC": auprc,
            "F1": f1,
            "Precision": precision,
            "Recall": recall,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
        })

    results_df = (
        pd.DataFrame(results)
        .sort_values(
            "AUPRC",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print("\nDefault-model results:")
    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    return results_df


# ============================================================
# Create hyperparameter combinations
# ============================================================

def create_hyperparameter_combinations():
    """
    Create all 36 parameter combinations.
    """

    combinations = []

    for (
        n_estimators,
        max_samples,
        max_features,
    ) in itertools.product(
        PARAM_GRID["n_estimators"],
        PARAM_GRID["max_samples"],
        PARAM_GRID["max_features"],
    ):

        combinations.append({
            "n_estimators": n_estimators,
            "max_samples": max_samples,
            "max_features": max_features,
            "contamination": CONTAMINATION,
        })

    return combinations


# ============================================================
# Hyperparameter optimization
# ============================================================

def optimize_models(
    model_datasets,
):
    """
    Optimize all four Isolation Forest models.

    Primary selection metric:
        AUPRC

    Training:
        normal training windows only

    Validation:
        observable validation windows only

    Final test:
        never used
    """

    combinations = (
        create_hyperparameter_combinations()
    )

    print("\n")
    print("=" * 80)
    print("ISOLATION FOREST HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)

    print(
        f"\nConfigurations per model: "
        f"{len(combinations)}"
    )

    print(
        f"Models: "
        f"{len(model_datasets)}"
    )

    print(
        f"Total model fits: "
        f"{len(combinations) * len(model_datasets)}"
    )

    optimization_results = []

    best_models = {}
    best_preprocessors = {}
    best_features = {}
    best_parameters = {}

    for model_name, data in model_datasets.items():

        print("\n")
        print("-" * 80)
        print(
            f"OPTIMIZING: {model_name}"
        )
        print("-" * 80)

        train_df = data["train"]
        validation_df = data["validation"]

        (
            X_train,
            X_validation,
            features_used,
            imputer,
        ) = prepare_features(
            model_name,
            train_df,
            validation_df,
        )

        y_validation = (
            validation_df[
                "observable_anomaly"
            ]
            .astype(int)
            .to_numpy()
        )

        print(
            f"Training windows:   {len(train_df):,}"
        )

        print(
            f"Validation windows: {len(validation_df):,}"
        )

        print(
            f"Features used:      {len(features_used):,}"
        )

        model_results = []

        for run_number, params in enumerate(
            combinations,
            start=1,
        ):

            model = IsolationForest(
                n_estimators=params[
                    "n_estimators"
                ],
                max_samples=params[
                    "max_samples"
                ],
                max_features=params[
                    "max_features"
                ],
                contamination=params[
                    "contamination"
                ],
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )

            # Train ONLY on normal training data
            model.fit(X_train)

            anomaly_scores = (
                -model.score_samples(
                    X_validation
                )
            )

            auprc = average_precision_score(
                y_validation,
                anomaly_scores,
            )

            result = {
                "model": model_name,
                "run": run_number,
                "n_estimators": params[
                    "n_estimators"
                ],
                "max_samples": params[
                    "max_samples"
                ],
                "max_features": params[
                    "max_features"
                ],
                "contamination": params[
                    "contamination"
                ],
                "AUPRC": auprc,
                "n_train": len(train_df),
                "n_validation": len(validation_df),
                "n_features": len(features_used),
            }

            optimization_results.append(
                result
            )

            model_results.append(
                result
            )

            print(
                f"[{run_number:02d}/"
                f"{len(combinations):02d}] "
                f"n_estimators="
                f"{params['n_estimators']}, "
                f"max_samples="
                f"{params['max_samples']}, "
                f"max_features="
                f"{params['max_features']} "
                f"-> AUPRC={auprc:.6f}"
            )

        model_results_df = (
            pd.DataFrame(model_results)
            .sort_values(
                "AUPRC",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        best_result = (
            model_results_df.iloc[0]
        )

        best_params = {
            "n_estimators": int(
                best_result[
                    "n_estimators"
                ]
            ),
            "max_samples": best_result[
                "max_samples"
            ],
            "max_features": best_result[
                "max_features"
            ],
            "contamination": CONTAMINATION,
        }

        print("\nBest configuration:")
        print(
            f"  n_estimators = "
            f"{best_params['n_estimators']}"
        )

        print(
            f"  max_samples  = "
            f"{best_params['max_samples']}"
        )

        print(
            f"  max_features = "
            f"{best_params['max_features']}"
        )

        print(
            f"  AUPRC        = "
            f"{best_result['AUPRC']:.6f}"
        )

        # ----------------------------------------------------
        # Refit best model
        #
        # Still training data ONLY.
        # ----------------------------------------------------

        best_model = IsolationForest(
            n_estimators=best_params[
                "n_estimators"
            ],
            max_samples=best_params[
                "max_samples"
            ],
            max_features=best_params[
                "max_features"
            ],
            contamination=CONTAMINATION,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        best_model.fit(X_train)

        best_models[
            model_name
        ] = best_model

        best_preprocessors[
            model_name
        ] = imputer

        best_features[
            model_name
        ] = features_used

        best_parameters[
            model_name
        ] = best_params

    optimization_results_df = (
        pd.DataFrame(
            optimization_results
        )
        .sort_values(
            [
                "model",
                "AUPRC",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    best_configuration_rows = []

    for model_name, params in best_parameters.items():

        model_rows = optimization_results_df[
            optimization_results_df["model"]
            == model_name
        ]

        best_row = model_rows.iloc[0]

        best_configuration_rows.append({
            "model": model_name,
            "n_estimators": params[
                "n_estimators"
            ],
            "max_samples": params[
                "max_samples"
            ],
            "max_features": params[
                "max_features"
            ],
            "contamination": params[
                "contamination"
            ],
            "AUPRC": best_row["AUPRC"],
            "n_train": best_row["n_train"],
            "n_validation": best_row[
                "n_validation"
            ],
            "n_features": best_row[
                "n_features"
            ],
        })

    best_configuration_df = (
        pd.DataFrame(
            best_configuration_rows
        )
        .sort_values(
            "AUPRC",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return (
        optimization_results_df,
        best_configuration_df,
        best_models,
        best_preprocessors,
        best_features,
    )


# ============================================================
# Threshold optimization
# ============================================================

def optimize_thresholds(
    model_datasets,
    best_models,
    best_preprocessors,
    best_features,
):
    """
    Optimize the classification threshold on validation.

    Primary threshold criterion:
        maximum F1

    The threshold is selected independently for each model.
    """

    print("\n")
    print("=" * 80)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 80)

    threshold_results = []
    threshold_curves = {}

    for model_name in best_models.keys():

        print("\n")
        print(
            f"Optimizing threshold: "
            f"{model_name}"
        )

        model = best_models[
            model_name
        ]

        imputer = best_preprocessors[
            model_name
        ]

        features_used = best_features[
            model_name
        ]

        validation_df = (
            model_datasets[
                model_name
            ]["validation"]
        )

        X_validation = validation_df[
            features_used
        ]

        y_validation = (
            validation_df[
                "observable_anomaly"
            ]
            .astype(int)
            .to_numpy()
        )

        X_validation_imputed = (
            imputer.transform(
                X_validation
            )
        )

        validation_scores = (
            -model.score_samples(
                X_validation_imputed
            )
        )

        thresholds = np.linspace(
            validation_scores.min(),
            validation_scores.max(),
            N_THRESHOLDS,
        )

        rows = []

        for threshold in thresholds:

            y_pred = (
                validation_scores >= threshold
            ).astype(int)

            precision = precision_score(
                y_validation,
                y_pred,
                zero_division=0,
            )

            recall = recall_score(
                y_validation,
                y_pred,
                zero_division=0,
            )

            f1 = f1_score(
                y_validation,
                y_pred,
                zero_division=0,
            )

            tn, fp, fn, tp = (
                confusion_matrix(
                    y_validation,
                    y_pred,
                    labels=[0, 1],
                ).ravel()
            )

            rows.append({
                "model": model_name,
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            })

        threshold_df = pd.DataFrame(
            rows
        )

        best_idx = threshold_df[
            "f1"
        ].idxmax()

        best_row = threshold_df.loc[
            best_idx
        ]

        threshold_curves[
            model_name
        ] = threshold_df

        threshold_results.append({
            "model": model_name,
            "threshold": best_row[
                "threshold"
            ],
            "precision": best_row[
                "precision"
            ],
            "recall": best_row[
                "recall"
            ],
            "f1": best_row[
                "f1"
            ],
            "tn": int(
                best_row["tn"]
            ),
            "fp": int(
                best_row["fp"]
            ),
            "fn": int(
                best_row["fn"]
            ),
            "tp": int(
                best_row["tp"]
            ),
        })

        print(
            f"Best threshold: "
            f"{best_row['threshold']:.6f}"
        )

        print(
            f"Precision:       "
            f"{best_row['precision']:.4f}"
        )

        print(
            f"Recall:          "
            f"{best_row['recall']:.4f}"
        )

        print(
            f"F1:              "
            f"{best_row['f1']:.4f}"
        )

    threshold_results_df = (
        pd.DataFrame(
            threshold_results
        )
    )

    return (
        threshold_results_df,
        threshold_curves,
    )


# ============================================================
# Final test evaluation
# ============================================================

def evaluate_final_test(
    model_datasets,
    best_models,
    best_preprocessors,
    best_features,
    threshold_results_df,
):
    """
    Evaluate the frozen models on final_test.

    IMPORTANT:
        - Model parameters selected on validation.
        - Imputer fitted on training.
        - Threshold selected on validation.
        - Final test is not used for any decision.
    """

    print("\n")
    print("=" * 80)
    print("FINAL TEST EVALUATION")
    print("=" * 80)

    final_results = []

    for model_name in best_models.keys():

        print(
            f"\nEvaluating: {model_name}"
        )

        model = best_models[
            model_name
        ]

        imputer = best_preprocessors[
            model_name
        ]

        features_used = best_features[
            model_name
        ]

        threshold_row = (
            threshold_results_df[
                threshold_results_df["model"]
                == model_name
            ]
            .iloc[0]
        )

        threshold = (
            threshold_row["threshold"]
        )

        final_test_df = (
            model_datasets[
                model_name
            ]["final_test"]
        )

        X_final_test = final_test_df[
            features_used
        ]

        y_final_test = (
            final_test_df[
                "observable_anomaly"
            ]
            .astype(int)
            .to_numpy()
        )

        X_final_test_imputed = (
            imputer.transform(
                X_final_test
            )
        )

        # Higher score = more anomalous
        final_test_scores = (
            -model.score_samples(
                X_final_test_imputed
            )
        )

        # ----------------------------------------------------
        # AUPRC uses continuous scores
        # ----------------------------------------------------

        auprc = average_precision_score(
            y_final_test,
            final_test_scores,
        )

        # ----------------------------------------------------
        # Classification uses frozen validation threshold
        # ----------------------------------------------------

        y_final_pred = (
            final_test_scores >= threshold
        ).astype(int)

        precision = precision_score(
            y_final_test,
            y_final_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_final_test,
            y_final_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_final_test,
            y_final_pred,
            zero_division=0,
        )

        tn, fp, fn, tp = confusion_matrix(
            y_final_test,
            y_final_pred,
            labels=[0, 1],
        ).ravel()

        final_results.append({
            "model": model_name,
            "threshold": threshold,
            "AUPRC": auprc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "n_test": len(final_test_df),
            "n_anomalies": int(
                y_final_test.sum()
            ),
            "n_normal": int(
                (y_final_test == 0).sum()
            ),
        })

        print(
            f"  AUPRC:     {auprc:.6f}"
        )

        print(
            f"  Precision:  {precision:.6f}"
        )

        print(
            f"  Recall:     {recall:.6f}"
        )

        print(
            f"  F1:         {f1:.6f}"
        )

        print(
            f"  TN={tn}, FP={fp}, "
            f"FN={fn}, TP={tp}"
        )

    final_results_df = (
        pd.DataFrame(
            final_results
        )
        .sort_values(
            "AUPRC",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return final_results_df


# ============================================================
# Save results
# ============================================================

def save_results(
    default_results_df,
    optimization_results_df,
    best_configuration_df,
    threshold_results_df,
    final_test_results_df,
):
    """
    Save numerical modeling results for later visualization.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    default_results_df.to_parquet(
        DEFAULT_RESULTS_PATH,
        index=False,
    )

    optimization_results_df["n_estimators"] = (
        optimization_results_df["n_estimators"].astype(int)
    )

    optimization_results_df["max_samples"] = (
        optimization_results_df["max_samples"].astype(str)
    )

    optimization_results_df["max_features"] = (
        optimization_results_df["max_features"].astype(str)
    )

    optimization_results_df.to_parquet(
        OPTIMIZATION_RESULTS_PATH,
        index=False,
    )

    best_configuration_df.to_parquet(
        BEST_CONFIG_PATH,
        index=False,
    )

    threshold_results_df.to_parquet(
        THRESHOLD_RESULTS_PATH,
        index=False,
    )

    final_test_results_df.to_parquet(
        FINAL_TEST_RESULTS_PATH,
        index=False,
    )

    print("\nResults saved:")
    print(
        f"  Default models:       "
        f"{DEFAULT_RESULTS_PATH}"
    )

    print(
        f"  Hyperparameters:      "
        f"{OPTIMIZATION_RESULTS_PATH}"
    )

    print(
        f"  Best configurations:  "
        f"{BEST_CONFIG_PATH}"
    )

    print(
        f"  Thresholds:           "
        f"{THRESHOLD_RESULTS_PATH}"
    )

    print(
        f"  Final test:           "
        f"{FINAL_TEST_RESULTS_PATH}"
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    datasets, experiment_split = (
        load_datasets()
    )

    # --------------------------------------------------------
    # Validate locked split
    # --------------------------------------------------------

    validate_experiment_split(
        experiment_split
    )

    # --------------------------------------------------------
    # Apply split + target
    # --------------------------------------------------------

    datasets = prepare_datasets(
        datasets,
        experiment_split,
    )

    # --------------------------------------------------------
    # Create model datasets
    # --------------------------------------------------------

    model_datasets = {}

    print("\n")
    print("=" * 80)
    print("MODEL DATASETS")
    print("=" * 80)

    for model_name, df in datasets.items():

        train, validation, final_test = (
            create_model_datasets(df)
        )

        model_datasets[
            model_name
        ] = {
            "train": train,
            "validation": validation,
            "final_test": final_test,
        }

        print(
            f"\n{model_name}"
        )

        print(
            f"  train:      {len(train):,}"
        )

        print(
            f"  validation: {len(validation):,}"
        )

        print(
            f"  final_test: {len(final_test):,}"
        )

    # --------------------------------------------------------
    # Stage 1: default models
    # --------------------------------------------------------

    default_results_df = (
        run_default_models(
            model_datasets
        )
    )

    # --------------------------------------------------------
    # Stage 2: hyperparameter optimization
    # --------------------------------------------------------

    (
        optimization_results_df,
        best_configuration_df,
        best_models,
        best_preprocessors,
        best_features,
    ) = optimize_models(
        model_datasets
    )

    # --------------------------------------------------------
    # Stage 3: threshold optimization
    # --------------------------------------------------------

    (
        threshold_results_df,
        threshold_curves,
    ) = optimize_thresholds(
        model_datasets,
        best_models,
        best_preprocessors,
        best_features,
    )

    # --------------------------------------------------------
    # Stage 4: final test
    # --------------------------------------------------------

    final_test_results_df = (
        evaluate_final_test(
            model_datasets,
            best_models,
            best_preprocessors,
            best_features,
            threshold_results_df,
        )
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    save_results(
        default_results_df,
        optimization_results_df,
        best_configuration_df,
        threshold_results_df,
        final_test_results_df,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 100)
    print("FINAL MODEL COMPARISON")
    print("=" * 100)

    print(
        final_test_results_df[
            [
                "model",
                "AUPRC",
                "F1",
                "Precision",
                "Recall",
                "TN",
                "FP",
                "FN",
                "TP",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    best_model = (
        final_test_results_df.iloc[0]
    )

    print("\n")
    print("=" * 100)
    print("BEST FINAL MODEL")
    print("=" * 100)

    print(
        f"\nModel:   {best_model['model']}"
    )

    print(
        f"AUPRC:   {best_model['AUPRC']:.6f}"
    )

    print(
        f"F1:      {best_model['F1']:.6f}"
    )

    print("\n")
    print("=" * 100)
    print("MODELING COMPLETE")
    print("=" * 100)

    print(
        "\n✓ Train data contained normal windows only."
    )

    print(
        "✓ Hyperparameters were selected using validation."
    )

    print(
        "✓ Thresholds were selected using validation."
    )

    print(
        "✓ Final-test data was not used for model selection."
    )

    print(
        "✓ Final-test evaluation was performed after "
        "model and threshold selection."
    )

    print(
        "\n✓ Numerical results are ready for visualization."
    )


if __name__ == "__main__":
    main()