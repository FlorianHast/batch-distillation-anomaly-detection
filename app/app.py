from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROCESSED_DIR / "model_results"


MODEL_ORDER = [
    "baseline_raw_windows",
    "window_features",
    "temporal_features",
    "ml_features",
]

MODEL_NAMES = {
    "baseline_raw_windows": "Raw Window Baseline",
    "window_features": "Window Features",
    "temporal_features": "Temporal Features",
    "ml_features": "ML Features",
}


PROCESS_VARIABLES = [
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
    "P701",
    "P702",
    "PDI701",
    "PDI702",
    "PY23",
    "FT703",
    "FT704",
    "FYI702",
    "LS701",
    "LS702",
    "H002",
    "H701",
    "H702",
    "H704",
    "H706",
    "H708",
    "P301",
    "TV1",
    "AV709",
    "AV716",
]


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Batch Distillation Anomaly Detection",
    page_icon="",
    layout="wide",
)


# ============================================================
# Helper functions
# ============================================================

@st.cache_data
def load_model_results():
    path = RESULTS_DIR / "final_test_results.parquet"

    if not path.exists():
        return None

    return pd.read_parquet(path)


@st.cache_data
def load_threshold_results():
    path = RESULTS_DIR / "threshold_results.parquet"

    if not path.exists():
        return None

    return pd.read_parquet(path)


@st.cache_data
def load_hyperparameter_results():
    path = RESULTS_DIR / "hyperparameter_results.parquet"

    if not path.exists():
        return None

    return pd.read_parquet(path)


@st.cache_data
def load_window_predictions():
    path = RESULTS_DIR / "window_predictions.parquet"

    if not path.exists():
        return None

    return pd.read_parquet(path)


@st.cache_data
def load_baseline_windows():
    path = PROCESSED_DIR / "baseline_raw_windows.parquet"

    if not path.exists():
        return None

    return pd.read_parquet(path)


def format_model_name(name):
    return MODEL_NAMES.get(name, name)


def get_validation_thresholds(threshold_results):
    """
    Return the validation-selected threshold for each model.
    """

    if threshold_results is None:
        return {}

    threshold_df = threshold_results.copy()

    threshold_df = threshold_df.sort_values(
        "f1",
        ascending=False,
    )

    best_thresholds = (
        threshold_df
        .groupby("model", as_index=False)
        .first()
    )

    return dict(
        zip(
            best_thresholds["model"],
            best_thresholds["threshold"],
        )
    )


def get_best_validation_configs(hyperparameter_results):
    """
    Return the best validation configuration for each model
    based on AUPRC.
    """

    if hyperparameter_results is None:
        return None

    results = hyperparameter_results.copy()

    results = results.sort_values(
        "AUPRC",
        ascending=False,
    )

    return (
        results
        .groupby("model", as_index=False)
        .first()
    )


def get_raw_window_data(
    baseline_windows,
    identifier,
    window,
):
    """
    Extract one 60-second raw window and reshape the
    flattened VARIABLE_00 ... VARIABLE_59 columns
    into a time-series dataframe.
    """

    if baseline_windows is None:
        return None

    row = baseline_windows[
        (baseline_windows["identifier"] == identifier)
        & (baseline_windows["window"] == window)
    ]

    if row.empty:
        return None

    row = row.iloc[0]

    data = {}

    for variable in PROCESS_VARIABLES:

        columns = [
            f"{variable}_{i:02d}"
            for i in range(60)
        ]

        # Only use the variable if all 60 samples exist.
        if all(column in baseline_windows.columns for column in columns):

            data[variable] = [
                row[column]
                for column in columns
            ]

    time_values = list(range(60))

    signal_df = pd.DataFrame(data)
    signal_df.insert(
        0,
        "Time (s)",
        time_values,
    )

    return signal_df


# ============================================================
# Load data
# ============================================================

model_results = load_model_results()
threshold_results = load_threshold_results()
hyperparameter_results = load_hyperparameter_results()
window_predictions = load_window_predictions()
baseline_windows = load_baseline_windows()

validation_thresholds = get_validation_thresholds(
    threshold_results
)

best_validation_configs = get_best_validation_configs(
    hyperparameter_results
)


# ============================================================
# Sidebar navigation
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select section",
    [
        "Project Overview",
        "Process & Data",
        "Feature Engineering",
        "Model Comparison",
        "Anomaly Detection Demo",
    ],
)


# ============================================================
# 1. Project Overview
# ============================================================

if page == "Project Overview":

    st.title("From Process Data to Fault Diagnosis")

    st.subheader(
        "Anomaly Detection for Batch Distillation"
    )

    st.markdown(
        """
        This project investigates whether machine learning can detect
        observable process anomalies in batch distillation data.

        The workflow compares four increasingly engineered data
        representations using Isolation Forest.
        """
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Experiments",
            "119",
        )

    with col2:
        st.metric(
            "60-second windows",
            "11,512",
        )

    with col3:
        st.metric(
            "Process variables",
            "30",
        )

    with col4:
        st.metric(
            "Models",
            "4",
        )

    st.divider()

    st.markdown("### Modeling Strategy")

    st.markdown(
        """
        **Training**
        - Normal experiments only

        **Validation**
        - Used for hyperparameter selection
        - Used to select the anomaly classification threshold

        **Final test**
        - Unseen experiments
        - Evaluated only after model selection was completed

        **Primary metric**
        - AUPRC

        **Secondary metric**
        - F1 score
        """
    )

    st.info(
        "Explainable fault diagnosis and NMR integration are "
        "planned as future extensions of the project."
    )


# ============================================================
# 2. Process & Data
# ============================================================

elif page == "Process & Data":

    st.title("Process & Data")

    st.markdown(
        """
        The dataset contains time-series measurements from a
        batch distillation process.

        For the current analysis, only the **Operation phase**
        is used. Startup and shutdown periods are excluded.
        """
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Dataset")

        st.markdown(
            """
            - 119 experiments
            - approximately 1 Hz sampling
            - 30 process variables
            - 60-second observation windows
            - 11,512 complete windows
            """
        )

    with col2:

        st.markdown("### Process Variables")

        variable_categories = {
            "Temperature": 10,
            "Pressure": 5,
            "Flow": 3,
            "Level": 2,
            "Heating / Power": 6,
            "Vacuum": 2,
            "Nitrogen": 1,
            "Cooling": 1,
        }

        category_df = pd.DataFrame(
            {
                "Category": variable_categories.keys(),
                "Variables": variable_categories.values(),
            }
        )

        st.dataframe(
            category_df,
            hide_index=True,
            use_container_width=True,
        )

    st.divider()

    st.markdown("### Windowing")

    st.markdown(
        """
        Each ML observation corresponds to exactly one complete
        60-second process window.

        This ensures that all four model representations are
        evaluated on the same underlying observations.
        """
    )


# ============================================================
# 3. Feature Engineering
# ============================================================

elif page == "Feature Engineering":

    st.title("Feature Engineering")

    st.markdown(
        """
        The project compares four representations of the same
        60-second process observations.
        """
    )

    st.divider()

    feature_table = pd.DataFrame(
        {
            "Representation": [
                "Raw Window Baseline",
                "Window Features",
                "Temporal Features",
                "ML Features",
            ],
            "Description": [
                "30 raw signals × 60 samples",
                "Statistical and process features",
                "Current + previous window + changes",
                "Temporal features + frequency-domain features",
            ],
            "Features": [
                1800,
                240,
                720,
                780,
            ],
        }
    )

    st.dataframe(
        feature_table,
        hide_index=True,
        use_container_width=True,
    )

    st.divider()

    st.markdown("### Feature Pipeline")

    st.code(
        """
Raw time-series
      ↓
60-second windows
      ↓
Raw Window Baseline
      ↓
Statistical / Process Features
      ↓
Temporal Features
      ↓
Frequency-Domain Features
      ↓
Isolation Forest
        """,
        language="text",
    )


# ============================================================
# 4. Model Comparison
# ============================================================

elif page == "Model Comparison":

    st.title("Model Comparison")

    if (
        model_results is None
        or threshold_results is None
        or hyperparameter_results is None
    ):

        st.error(
            "Model result files are missing. "
            "Run the modeling pipeline first."
        )

        st.stop()

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    st.header("Model Selection — Validation Set")

    st.markdown(
        """
        The validation set is used for model selection.
        Hyperparameters are selected using AUPRC, and the
        classification threshold is subsequently selected
        using F1 score.
        """
    )

    best_validation = (
        hyperparameter_results
        .sort_values("AUPRC", ascending=False)
        .groupby("model", as_index=False)
        .first()
    )

    best_validation["Model"] = (
        best_validation["model"]
        .map(format_model_name)
    )

    model_order_display = [
        "Raw Window Baseline",
        "Window Features",
        "Temporal Features",
        "ML Features",
    ]

    best_validation["Model"] = pd.Categorical(
        best_validation["Model"],
        categories=model_order_display,
        ordered=True,
    )

    best_validation = best_validation.sort_values("Model")

    chart_df = best_validation[
        ["Model", "AUPRC"]
    ].copy()

    fig = px.bar(
        chart_df,
        x="Model",
        y="AUPRC",
        category_orders={
            "Model": model_order_display
        },
        labels={
            "Model": "",
            "AUPRC": "AUPRC",
        },
    )

    fig.update_yaxes(
        range=[0, 1],
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    selected_model = (
        best_validation
        .sort_values("AUPRC", ascending=False)
        .iloc[0]
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Selected Model",
            format_model_name(
                selected_model["model"]
            ),
        )

    with col2:
        st.metric(
            "Validation AUPRC",
            f"{selected_model['AUPRC']:.3f}",
        )

    with col3:

        selected_threshold = validation_thresholds.get(
            selected_model["model"],
            None,
        )

        if selected_threshold is not None:
            st.metric(
                "Validation Threshold",
                f"{selected_threshold:.3f}",
            )

    st.divider()

    # --------------------------------------------------------
    # Threshold selection
    # --------------------------------------------------------

    st.header("Validation Threshold Selection")

    threshold_display = threshold_results.copy()

    threshold_display["Model"] = (
        threshold_display["model"]
        .map(format_model_name)
    )

    threshold_display = threshold_display.sort_values(
        "f1",
        ascending=False,
    )

    threshold_display = (
        threshold_display
        .groupby("model", as_index=False)
        .first()
    )

    threshold_display["Model"] = (
        threshold_display["model"]
        .map(format_model_name)
    )

    threshold_display["Threshold"] = (
        threshold_display["threshold"]
        .round(3)
    )

    threshold_display["Precision"] = (
        threshold_display["precision"]
        .round(3)
    )

    threshold_display["Recall"] = (
        threshold_display["recall"]
        .round(3)
    )

    threshold_display["F1"] = (
        threshold_display["f1"]
        .round(3)
    )

    threshold_display = threshold_display[
        [
            "Model",
            "Threshold",
            "Precision",
            "Recall",
            "F1",
        ]
    ]

    st.dataframe(
        threshold_display,
        hide_index=True,
        use_container_width=True,
    )

    st.divider()

    # --------------------------------------------------------
    # Final test
    # --------------------------------------------------------

    st.header("Final Evaluation — Unseen Test Set")

    final_display = model_results.copy()

    final_display["Model"] = (
        final_display["model"]
        .map(format_model_name)
    )

    final_display["Model"] = pd.Categorical(
        final_display["Model"],
        categories=model_order_display,
        ordered=True,
    )

    final_display = final_display.sort_values(
        "Model"
    )

    chart_df = final_display[
        ["Model", "AUPRC"]
    ].copy()

    fig = px.bar(
        chart_df,
        x="Model",
        y="AUPRC",
        category_orders={
            "Model": model_order_display
        },
        labels={
            "Model": "",
            "AUPRC": "AUPRC",
        },
    )

    fig.update_yaxes(
        range=[0, 1],
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.dataframe(
        final_display[
            [
                "Model",
                "AUPRC",
                "F1",
                "Precision",
                "Recall",
            ]
        ].round(3),
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# 5. Anomaly Detection Demo
# ============================================================

elif page == "Anomaly Detection Demo":

    st.title("Anomaly Detection Demo")

    st.markdown(
        """
        Inspect individual 60-second windows from the unseen
        final-test set and compare the model prediction with
        the ground-truth anomaly label.
        """
    )

    if window_predictions is None:

        st.error(
            "window_predictions.parquet was not found. "
            "Run the modeling pipeline first."
        )

        st.stop()

    if baseline_windows is None:

        st.error(
            "baseline_raw_windows.parquet was not found."
        )

        st.stop()

    st.divider()

    # --------------------------------------------------------
    # Model selection
    # --------------------------------------------------------

    selected_model = st.selectbox(
        "Model",
        options=MODEL_ORDER,
        format_func=format_model_name,
    )

    model_predictions = window_predictions[
        window_predictions["model"] == selected_model
    ].copy()

    if model_predictions.empty:

        st.warning(
            "No predictions available for this model."
        )

        st.stop()

    # --------------------------------------------------------
    # Experiment selection
    # --------------------------------------------------------

    experiments = (
        model_predictions["identifier"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    selected_identifier = st.selectbox(
        "Experiment",
        options=experiments,
    )

    experiment_predictions = model_predictions[
        model_predictions["identifier"]
        == selected_identifier
    ].copy()

    experiment_predictions = experiment_predictions.sort_values(
        "window"
    )

    # --------------------------------------------------------
    # Window selection
    # --------------------------------------------------------

    available_windows = (
        experiment_predictions["window"]
        .astype(int)
        .tolist()
    )

    selected_window = st.selectbox(
        "60-second window",
        options=available_windows,
        format_func=lambda x: (
            f"Window {x} "
            f"({x * 60:.0f}–{(x + 1) * 60:.0f} s)"
        ),
    )

    selected_prediction = experiment_predictions[
        experiment_predictions["window"]
        == selected_window
    ].iloc[0]

    # --------------------------------------------------------
    # Prediction information
    # --------------------------------------------------------

    anomaly_score = float(
        selected_prediction["anomaly_score"]
    )

    threshold = validation_thresholds.get(
        selected_model,
        None,
    )

    prediction = int(
        selected_prediction["prediction"]
    )

    ground_truth = selected_prediction[
        "observable_anomaly"
    ]

    anomaly_label = selected_prediction[
        "anomaly_label"
    ]

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Anomaly Score",
            f"{anomaly_score:.3f}",
        )

    with col2:

        if threshold is not None:

            st.metric(
                "Validation Threshold",
                f"{threshold:.3f}",
            )

        else:

            st.metric(
                "Validation Threshold",
                "N/A",
            )

    with col3:

        prediction_text = (
            "Anomaly"
            if prediction == 1
            else "Normal"
        )

        st.metric(
            "Model Prediction",
            prediction_text,
        )

    with col4:

        if pd.isna(ground_truth):

            ground_truth_text = "Excluded"

        elif int(ground_truth) == 1:

            ground_truth_text = "Anomaly"

        else:

            ground_truth_text = "Normal"

        st.metric(
            "Ground Truth",
            ground_truth_text,
        )

    # --------------------------------------------------------
    # Score / threshold visualization
    # --------------------------------------------------------

    st.subheader("Anomaly Score")

    score_df = pd.DataFrame(
        {
            "Metric": [
                "Anomaly Score",
                "Validation Threshold",
            ],
            "Value": [
                anomaly_score,
                threshold if threshold is not None else 0,
            ],
        }
    )

    fig = px.bar(
        score_df,
        x="Metric",
        y="Value",
        labels={
            "Metric": "",
            "Value": "Score",
        },
    )

    fig.update_yaxes(
        range=[
            0,
            max(
                1.0,
                anomaly_score * 1.15,
                (threshold or 0) * 1.15,
            ),
        ],
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Window metadata
    # --------------------------------------------------------

    st.subheader("Window Information")

    metadata_col1, metadata_col2, metadata_col3 = st.columns(3)

    with metadata_col1:

        st.write(
            f"**Batch:** "
            f"{selected_prediction['batch']}"
        )

        st.write(
            f"**Operating point:** "
            f"{selected_prediction['operating_point']}"
        )

    with metadata_col2:

        st.write(
            f"**Experiment:** "
            f"{selected_prediction['experiment']}"
        )

        st.write(
            f"**Experiment type:** "
            f"{selected_prediction['experiment_type']}"
        )

    with metadata_col3:

        st.write(
            f"**Window:** "
            f"{selected_window}"
        )

        st.write(
            f"**Time:** "
            f"{selected_prediction['window_start']:.0f}–"
            f"{selected_prediction['window_end']:.0f} s"
        )

    if pd.notna(anomaly_label):

        st.write(
            f"**Original anomaly label:** "
            f"{int(anomaly_label)}"
        )

    # --------------------------------------------------------
    # Raw process signals
    # --------------------------------------------------------

    st.divider()

    st.subheader("Process Signals")

    st.markdown(
        """
        Select process variables to inspect the underlying
        60-second process window.
        """
    )

    selected_variables = st.multiselect(
        "Process variables",
        options=PROCESS_VARIABLES,
        default=[
            "T701",
            "T704",
            "T706",
        ],
        max_selections=4,
    )

    if not selected_variables:

        st.info(
            "Select at least one process variable."
        )

    else:

        signal_df = get_raw_window_data(
            baseline_windows,
            selected_identifier,
            selected_window,
        )

        if signal_df is None:

            st.warning(
                "Raw window data could not be loaded."
            )

        else:

            available_variables = [
                variable
                for variable in selected_variables
                if variable in signal_df.columns
            ]

            if not available_variables:

                st.warning(
                    "Selected variables are not available."
                )

            else:

                plot_df = signal_df[
                    ["Time (s)"] + available_variables
                ]

                plot_long = plot_df.melt(
                    id_vars="Time (s)",
                    var_name="Variable",
                    value_name="Value",
                )

                fig = px.line(
                    plot_long,
                    x="Time (s)",
                    y="Value",
                    color="Variable",
                    labels={
                        "Time (s)": "Time within window (s)",
                        "Value": "Process value",
                    },
                )

                fig.update_layout(
                    hovermode="x unified",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

    # --------------------------------------------------------
    # Window comparison
    # --------------------------------------------------------

    st.divider()

    st.subheader("Experiment Anomaly Timeline")

    timeline_df = experiment_predictions[
        [
            "window",
            "window_start",
            "window_end",
            "anomaly_score",
            "prediction",
            "observable_anomaly",
        ]
    ].copy()

    timeline_df["Prediction"] = timeline_df[
        "prediction"
    ].map(
        {
            0: "Normal",
            1: "Anomaly",
        }
    )

    timeline_df["Ground Truth"] = timeline_df[
        "observable_anomaly"
    ].map(
        {
            0.0: "Normal",
            1.0: "Anomaly",
        }
    )

    fig = px.line(
        timeline_df,
        x="window_start",
        y="anomaly_score",
        markers=True,
        labels={
            "window_start": "Experiment time (s)",
            "anomaly_score": "Anomaly score",
        },
        hover_data=[
            "window",
            "window_end",
            "Prediction",
            "Ground Truth",
        ],
    )

    if threshold is not None:

        fig.add_hline(
            y=threshold,
            line_dash="dash",
            annotation_text="Validation threshold",
        )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.caption(
        "The dashed line represents the classification threshold "
        "selected on the validation set. The final-test windows "
        "remain completely unseen during threshold selection."
    )