from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "window_features.parquet"

META_COLUMNS = [
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


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Batch Distillation Anomaly Detection",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            font-size: 1.05rem;
            color: #666;
            margin-bottom: 1.5rem;
        }

        .section-title {
            font-size: 1.35rem;
            font-weight: 600;
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Data loading
# ============================================================

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    """Load the processed window-level feature dataset."""

    df = pd.read_parquet(path)

    for column in ["window_start", "window_end"]:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
            )

    return df


# ============================================================
# Helper functions
# ============================================================

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all columns that are not metadata columns."""

    return [
        column
        for column in df.columns
        if column not in META_COLUMNS
    ]


def get_numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns only."""

    feature_columns = get_feature_columns(df)

    return [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]


def create_feature_plot(
    df: pd.DataFrame,
    feature: str,
) -> go.Figure:
    """Create a window-level feature plot."""

    plot_df = df.sort_values("window").copy()

    fig = go.Figure()

    # Normal windows
    normal = plot_df[
        plot_df["anomaly_label"] == 0
    ]

    if not normal.empty:
        fig.add_trace(
            go.Scatter(
                x=normal["window"],
                y=normal[feature],
                mode="lines+markers",
                name="Normal",
                hovertemplate=(
                    "Window: %{x}<br>"
                    f"{feature}: %{{y}}"
                    "<extra></extra>"
                ),
            )
        )

    # Anomalous windows
    anomaly = plot_df[
        plot_df["anomaly_label"] == 1
    ]

    if not anomaly.empty:
        fig.add_trace(
            go.Scatter(
                x=anomaly["window"],
                y=anomaly[feature],
                mode="markers",
                name="Anomaly",
                marker=dict(
                    size=10,
                    symbol="x",
                ),
                hovertemplate=(
                    "Window: %{x}<br>"
                    f"{feature}: %{{y}}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=feature,
        xaxis_title="60-second window",
        yaxis_title=feature,
        hovermode="x unified",
        height=450,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    return fig


# ============================================================
# Check data availability
# ============================================================

if not DATA_PATH.exists():
    st.error(
        "Data file not found:\n\n"
        f"{DATA_PATH}"
    )
    st.stop()


# ============================================================
# Load dataset
# ============================================================

df = load_data(DATA_PATH)


# ============================================================
# Validate required columns
# ============================================================

required_columns = [
    "batch",
    "operating_point",
    "experiment",
    "experiment_type",
    "phase",
    "identifier",
    "window",
    "window_start",
    "window_end",
    "anomaly_label",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        "The dataset is missing the following required columns:\n\n"
        + "\n".join(
            f"- {column}"
            for column in missing_columns
        )
    )
    st.stop()


# ============================================================
# Header
# ============================================================

st.markdown(
    '<div class="main-title">'
    "Batch Distillation Anomaly Detection"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Explainable anomaly detection for industrial batch
        distillation processes
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("Experiment Explorer")


# Batch
batches = sorted(
    df["batch"]
    .dropna()
    .unique()
)

if not batches:
    st.error("No batches found in the dataset.")
    st.stop()

selected_batch = st.sidebar.selectbox(
    "Batch",
    batches,
)

batch_df = df[
    df["batch"] == selected_batch
].copy()


# Operating point
operating_points = sorted(
    batch_df["operating_point"]
    .dropna()
    .unique()
)

if not operating_points:
    st.error(
        "No operating points found for the selected batch."
    )
    st.stop()

selected_operating_point = st.sidebar.selectbox(
    "Operating point",
    operating_points,
)

filtered_df = batch_df[
    batch_df["operating_point"] == selected_operating_point
].copy()


# Experiment
experiments = sorted(
    filtered_df["experiment"]
    .dropna()
    .unique()
)

if not experiments:
    st.error(
        "No experiments found for the selected operating point."
    )
    st.stop()

selected_experiment = st.sidebar.selectbox(
    "Experiment",
    experiments,
)

experiment_df = filtered_df[
    filtered_df["experiment"] == selected_experiment
].copy()


# ============================================================
# Experiment overview
# ============================================================

st.markdown(
    '<div class="section-title">'
    "Experiment Overview"
    "</div>",
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric(
        "Windows",
        len(experiment_df),
    )


with col2:
    anomaly_windows = int(
        (
            experiment_df["anomaly_label"] == 1
        ).sum()
    )

    st.metric(
        "Anomalous windows",
        anomaly_windows,
    )


with col3:
    experiment_type = (
        experiment_df["experiment_type"].iloc[0]
        if not experiment_df.empty
        else "—"
    )

    st.metric(
        "Experiment type",
        experiment_type,
    )


with col4:
    phase_values = (
        experiment_df["phase"]
        .dropna()
        .unique()
    )

    phase_display = (
        ", ".join(
            map(str, phase_values)
        )
        if len(phase_values) > 0
        else "—"
    )

    st.metric(
        "Phase",
        phase_display,
    )


# ============================================================
# Experiment metadata
# ============================================================

with st.expander(
    "Experiment metadata",
    expanded=False,
):

    metadata = {
        "Batch": selected_batch,
        "Operating point": selected_operating_point,
        "Experiment": selected_experiment,
        "Experiment type": experiment_type,
        "Identifier": (
            experiment_df["identifier"].iloc[0]
            if not experiment_df.empty
            else "—"
        ),
    }

    metadata_df = pd.DataFrame(
        metadata.items(),
        columns=[
            "Property",
            "Value",
        ],
    )

    st.dataframe(
        metadata_df,
        width="stretch",
        hide_index=True,
    )


# ============================================================
# Window-level anomaly overview
# ============================================================

st.markdown(
    '<div class="section-title">'
    "Window-level Anomaly Overview"
    "</div>",
    unsafe_allow_html=True,
)

overview_df = experiment_df[
    [
        "window",
        "window_start",
        "window_end",
        "anomaly_label",
    ]
].sort_values("window")


fig = go.Figure()


# Normal windows
normal = overview_df[
    overview_df["anomaly_label"] == 0
]

if not normal.empty:
    fig.add_trace(
        go.Scatter(
            x=normal["window"],
            y=[0] * len(normal),
            mode="markers",
            name="Normal",
            marker=dict(
                size=9,
                symbol="circle",
            ),
            text=normal[
                "window_start"
            ].astype(str),
            hovertemplate=(
                "Window: %{x}<br>"
                "Start: %{text}"
                "<extra></extra>"
            ),
        )
    )


# Anomalous windows
anomaly = overview_df[
    overview_df["anomaly_label"] == 1
]

if not anomaly.empty:
    fig.add_trace(
        go.Scatter(
            x=anomaly["window"],
            y=[1] * len(anomaly),
            mode="markers",
            name="Anomaly",
            marker=dict(
                size=11,
                symbol="x",
            ),
            text=anomaly[
                "window_start"
            ].astype(str),
            hovertemplate=(
                "Window: %{x}<br>"
                "Start: %{text}"
                "<extra></extra>"
            ),
        )
    )


fig.update_layout(
    height=250,
    xaxis_title="60-second window",
    yaxis=dict(
        title="Status",
        tickmode="array",
        tickvals=[0, 1],
        ticktext=[
            "Normal",
            "Anomaly",
        ],
        range=[
            -0.3,
            1.3,
        ],
    ),
    hovermode="closest",
    margin=dict(
        l=20,
        r=20,
        t=30,
        b=20,
    ),
)

st.plotly_chart(
    fig,
    width="stretch",
)


# ============================================================
# Feature explorer
# ============================================================

st.markdown(
    '<div class="section-title">'
    "Feature Explorer"
    "</div>",
    unsafe_allow_html=True,
)

numeric_features = get_numeric_feature_columns(
    experiment_df
)

if not numeric_features:

    st.warning(
        "No numeric feature columns available."
    )

else:

    preferred_features = [
        "T701_mean",
        "T702_mean",
        "T703_mean",
        "T704_mean",
        "T705_mean",
        "T706_mean",
        "T707_mean",
        "T708_mean",
        "T709_mean",
        "T710_mean",
        "T701_slope",
        "T704_slope",
        "T705_slope",
        "T706_slope",
    ]

    ordered_features = [
        feature
        for feature in preferred_features
        if feature in numeric_features
    ]

    remaining_features = [
        feature
        for feature in numeric_features
        if feature not in ordered_features
    ]

    ordered_features.extend(
        remaining_features
    )

    selected_feature = st.selectbox(
        "Select a feature",
        ordered_features,
    )

    st.plotly_chart(
        create_feature_plot(
            experiment_df,
            selected_feature,
        ),
        width="stretch",
    )


# ============================================================
# Anomalous windows table
# ============================================================

st.markdown(
    '<div class="section-title">'
    "Detected / Labelled Anomalous Windows"
    "</div>",
    unsafe_allow_html=True,
)

if anomaly_windows == 0:

    st.info(
        "No anomalous windows are labelled "
        "for this experiment."
    )

else:

    anomaly_table = experiment_df[
        experiment_df["anomaly_label"] == 1
    ][
        [
            "window",
            "window_start",
            "window_end",
        ]
    ].sort_values("window")

    st.dataframe(
        anomaly_table,
        width="stretch",
        hide_index=True,
    )


# ============================================================
# Dataset information
# ============================================================

with st.expander(
    "Dataset information",
    expanded=False,
):

    feature_columns = get_feature_columns(df)

    st.write(
        "**Dataset:** `window_features.parquet`"
    )

    st.write(
        f"**Rows:** {len(df):,}"
    )

    st.write(
        f"**Columns:** {len(df.columns):,}"
    )

    st.write(
        f"**Metadata columns:** {len(META_COLUMNS)}"
    )

    st.write(
        f"**Feature columns:** {len(feature_columns)}"
    )