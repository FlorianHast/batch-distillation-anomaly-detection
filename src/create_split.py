from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_PATH = PROCESSED_DIR / "baseline_raw_windows.parquet"
OUTPUT_PATH = PROCESSED_DIR / "experiment_split.parquet"

RANDOM_STATE_SEARCH = range(1000)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CREATE EXPERIMENT-LEVEL TRAIN / VALIDATION / FINAL-TEST SPLIT")
print("=" * 70)

print(f"\nProject root:")
print(PROJECT_ROOT)

print(f"\nLoading:")
print(INPUT_PATH)

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_PATH}"
    )

baseline = pd.read_parquet(INPUT_PATH)

print(f"Loaded baseline data: {baseline.shape}")


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "identifier",
    "batch",
    "operating_point",
    "experiment",
    "experiment_type",
    "anomaly_label",
]

missing_columns = [
    column
    for column in required_columns
    if column not in baseline.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(f"  - {column}" for column in missing_columns)
    )


# ============================================================
# EXPERIMENT-LEVEL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT-LEVEL CLASSIFICATION")
print("=" * 70)

experiment_summary = (
    baseline
    .groupby("identifier", as_index=False)
    .agg(
        batch=("batch", "first"),
        operating_point=("operating_point", "first"),
        experiment=("experiment", "first"),
        experiment_type=("experiment_type", "first"),
        anomaly_labels=(
            "anomaly_label",
            lambda x: sorted(
                pd.Series(x)
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            ),
        ),
    )
)


# ============================================================
# CLASSIFY EXPERIMENTS
# ============================================================

def classify_experiment(labels):
    """
    Classify an experiment based on its anomaly labels.

    Label 0 = normal
    Label 1 = unobservable / blind fault
    Labels 2/3 = observable anomaly

    Priority:
    - If label 2 or 3 occurs anywhere -> observable_anomaly
    - Else if label 1 occurs -> unobservable_fault
    - Else -> train_normal
    """

    labels = set(labels)

    if 2 in labels or 3 in labels:
        return "observable_anomaly"

    if 1 in labels:
        return "unobservable_fault"

    return "train_normal"


experiment_summary["experiment_class"] = (
    experiment_summary["anomaly_labels"]
    .apply(classify_experiment)
)


# ============================================================
# PRINT CLASSIFICATION
# ============================================================

class_counts = (
    experiment_summary["experiment_class"]
    .value_counts()
    .sort_index()
)

print("\nExperiment classification:")

for experiment_class, count in class_counts.items():
    print(f"  {experiment_class:22s}: {count:3d}")


# ============================================================
# CHECK FOR MIXED / UNEXPECTED EXPERIMENTS
# ============================================================

valid_classes = {
    "train_normal",
    "observable_anomaly",
    "unobservable_fault",
}

unexpected_classes = set(
    experiment_summary["experiment_class"]
) - valid_classes

if unexpected_classes:
    raise ValueError(
        f"Unexpected experiment classes: {unexpected_classes}"
    )


# ============================================================
# SELECT OBSERVABLE-ANOMALY EXPERIMENTS
# ============================================================

observable_experiments = experiment_summary[
    experiment_summary["experiment_class"]
    == "observable_anomaly"
].copy()

normal_experiments = experiment_summary[
    experiment_summary["experiment_class"]
    == "train_normal"
].copy()

unobservable_experiments = experiment_summary[
    experiment_summary["experiment_class"]
    == "unobservable_fault"
].copy()


print("\nObservable anomaly experiments:")
print(f"  {len(observable_experiments)}")

print("Normal experiments:")
print(f"  {len(normal_experiments)}")

print("Unobservable-fault experiments:")
print(f"  {len(unobservable_experiments)}")


# ============================================================
# GROUP OBSERVABLE ANOMALIES BY PROCESS CONTEXT
# ============================================================

GROUP_COLUMNS = [
    "batch",
    "operating_point",
]

group_summary = (
    observable_experiments
    .groupby(GROUP_COLUMNS, dropna=False)
    .agg(
        n_experiments=("identifier", "nunique")
    )
    .reset_index()
)

print("\n" + "=" * 70)
print("PROCESS-CONTEXT GROUPS")
print("=" * 70)

print(f"\nNumber of observable process-context groups: {len(group_summary)}")

print("\nExperiments per group:")
print(
    group_summary["n_experiments"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# FIND BALANCED VALIDATION / FINAL-TEST GROUP SPLIT
# ============================================================

print("\n" + "=" * 70)
print("SEARCHING FOR BALANCED GROUP SPLIT")
print("=" * 70)

if len(group_summary) < 2:
    raise ValueError(
        "At least two process-context groups are required "
        "to create validation and final_test sets."
    )


best_split = None
best_score = None


# We want approximately equal numbers of experiments
# in validation and final_test while keeping every
# (batch, operating_point) group entirely in one split.

for random_state in RANDOM_STATE_SEARCH:

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.5,
        random_state=random_state,
    )

    group_indices = group_summary.index.to_numpy()

    train_indices, test_indices = next(
        splitter.split(
            group_indices,
            groups=group_indices,
        )
    )

    validation_groups = group_summary.iloc[train_indices]
    final_test_groups = group_summary.iloc[test_indices]

    validation_count = int(
        validation_groups["n_experiments"].sum()
    )

    final_test_count = int(
        final_test_groups["n_experiments"].sum()
    )

    # Primary objective:
    # minimize difference in number of experiments.
    score = abs(
        validation_count - final_test_count
    )

    if best_score is None or score < best_score:
        best_score = score

        best_split = {
            "random_state": random_state,
            "validation_groups": validation_groups.copy(),
            "final_test_groups": final_test_groups.copy(),
            "validation_count": validation_count,
            "final_test_count": final_test_count,
        }


if best_split is None:
    raise RuntimeError(
        "Could not find a validation/final-test split."
    )


print("\nBest split found:")
print(f"  Random state:       {best_split['random_state']}")
print(f"  Validation groups:  {len(best_split['validation_groups'])}")
print(f"  Final-test groups:  {len(best_split['final_test_groups'])}")
print(f"  Validation experiments: {best_split['validation_count']}")
print(f"  Final-test experiments: {best_split['final_test_count']}")
print(f"  Difference:         {best_score}")


# ============================================================
# ASSIGN SPLITS TO EXPERIMENTS
# ============================================================

validation_group_keys = set(
    zip(
        best_split["validation_groups"]["batch"],
        best_split["validation_groups"]["operating_point"],
    )
)

final_test_group_keys = set(
    zip(
        best_split["final_test_groups"]["batch"],
        best_split["final_test_groups"]["operating_point"],
    )
)


def assign_split(row):
    """
    Assign an experiment to train, validation, or final_test.
    """

    experiment_class = row["experiment_class"]

    if experiment_class == "train_normal":
        return "train"

    if experiment_class == "unobservable_fault":
        return "excluded"

    group_key = (
        row["batch"],
        row["operating_point"],
    )

    if group_key in validation_group_keys:
        return "validation"

    if group_key in final_test_group_keys:
        return "final_test"

    raise ValueError(
        "Observable anomaly experiment does not belong "
        "to validation or final_test group."
    )


experiment_summary["split"] = (
    experiment_summary
    .apply(assign_split, axis=1)
)


# ============================================================
# BUILD FINAL SPLIT TABLE
# ============================================================

OUTPUT_COLUMNS = [
    "identifier",
    "batch",
    "operating_point",
    "experiment",
    "experiment_type",
    "experiment_class",
    "split",
]

experiment_split = experiment_summary[
    OUTPUT_COLUMNS
].copy()


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATING SPLIT")
print("=" * 70)


# ------------------------------------------------------------
# Every experiment must occur exactly once
# ------------------------------------------------------------

if experiment_split["identifier"].duplicated().any():
    raise ValueError(
        "An experiment identifier occurs more than once."
    )


# ------------------------------------------------------------
# Split counts
# ------------------------------------------------------------

split_counts = (
    experiment_split["split"]
    .value_counts()
    .sort_index()
)

print("\nExperiment counts by split:")

for split_name, count in split_counts.items():
    print(f"  {split_name:12s}: {count:3d}")


# ------------------------------------------------------------
# Check normal experiments
# ------------------------------------------------------------

normal_split = experiment_split[
    experiment_split["experiment_class"] == "train_normal"
]

if not (normal_split["split"] == "train").all():
    raise ValueError(
        "Not all normal experiments were assigned to train."
    )


# ------------------------------------------------------------
# Check unobservable faults
# ------------------------------------------------------------

unobservable_split = experiment_split[
    experiment_split["experiment_class"]
    == "unobservable_fault"
]

if not (unobservable_split["split"] == "excluded").all():
    raise ValueError(
        "Not all unobservable-fault experiments were excluded."
    )


# ------------------------------------------------------------
# Check observable anomalies
# ------------------------------------------------------------

observable_split = experiment_split[
    experiment_split["experiment_class"]
    == "observable_anomaly"
]

if not observable_split["split"].isin(
    ["validation", "final_test"]
).all():
    raise ValueError(
        "Observable anomaly experiments must be assigned "
        "to validation or final_test."
    )


# ------------------------------------------------------------
# Check process-context leakage
# ------------------------------------------------------------

group_split_counts = (
    observable_split
    .groupby(GROUP_COLUMNS)["split"]
    .nunique()
)

leaking_groups = group_split_counts[
    group_split_counts > 1
]

if len(leaking_groups) > 0:
    raise ValueError(
        "Process-context leakage detected. "
        "The following (batch, operating_point) groups "
        "occur in both validation and final_test:\n"
        f"{leaking_groups}"
    )


# ------------------------------------------------------------
# Check experiment leakage against baseline data
# ------------------------------------------------------------

baseline_experiment_ids = set(
    baseline["identifier"].unique()
)

split_experiment_ids = set(
    experiment_split["identifier"].unique()
)

if baseline_experiment_ids != split_experiment_ids:
    missing = baseline_experiment_ids - split_experiment_ids
    extra = split_experiment_ids - baseline_experiment_ids

    raise ValueError(
        "Experiment IDs do not match baseline data.\n"
        f"Missing: {missing}\n"
        f"Extra: {extra}"
    )


# ============================================================
# SAVE SPLIT
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

experiment_split.to_parquet(
    OUTPUT_PATH,
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SPLIT CREATED SUCCESSFULLY")
print("=" * 70)

print(f"\nSaved to:")
print(OUTPUT_PATH)

print("\nFinal experiment distribution:")
print(
    experiment_split[
        ["experiment_class", "split"]
    ]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nProcess-context distribution:")

process_context_distribution = (
    experiment_split[
        experiment_split["split"].isin(
            ["validation", "final_test"]
        )
    ]
    .groupby(
        ["split"] + GROUP_COLUMNS
    )
    .size()
    .reset_index(name="n_experiments")
)

print(
    process_context_distribution
    .to_string(index=False)
)

print("\nValidation checks passed.")