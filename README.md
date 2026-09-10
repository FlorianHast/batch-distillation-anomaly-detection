# From Process Data to Fault Diagnosis

## Anomaly Detection for Batch Distillation

This project investigates the use of machine learning for detecting observable process anomalies in batch distillation data.

The core objective is to compare different representations of 60-second process windows and evaluate how feature engineering affects anomaly detection performance.

The project uses an **Isolation Forest** approach and compares four feature representations:

1. Raw Window Baseline
2. Window Features
3. Temporal Features
4. ML Features

A Streamlit application provides an interactive presentation of the process, feature engineering pipeline, model results, and individual anomaly detections.

---

## Project Overview

Batch distillation processes generate large amounts of multivariate time-series data. Detecting abnormal process behavior from these signals can support earlier identification of process deviations.

The project follows a structured pipeline:

```text
Raw Process Data
       ↓
Data Preprocessing
       ↓
60-second Windows
       ↓
Feature Engineering
       ↓
Experiment-level Train / Validation / Test Split
       ↓
Isolation Forest
       ↓
Model Comparison
       ↓
Anomaly Detection
```

The current project focuses on **observable anomalies** during the Operation phase.

Explainable fault diagnosis and integration of NMR information are planned as future extensions.

---

## Dataset

The dataset contains multivariate time-series measurements from batch distillation experiments and was published by **Arweiler et al. (2026)** in *Scientific Data*.

The dataset provides experimental process data from a laboratory-scale batch distillation plant, including fault-free and anomalous experiments, sensor and actuator measurements, anomaly annotations, and additional data sources such as online NMR spectroscopy.

> Arweiler et al. (2026)
> *Batch Distillation Data for Developing Machine Learning Anomaly Detection Methods*
> Scientific Data, 13, 513.
> DOI: [10.1038/s41597-026-07124-3](https://doi.org/10.1038/s41597-026-07124-3)

### Dataset Characteristics

* 119 experiments
* Approximately 1 Hz sampling frequency
* 30 process variables
* 60-second observation windows
* 11,512 complete windows
* Operation phase only

Startup and shutdown periods are excluded from the current analysis.

### Process Variables

The 30 variables cover several process categories:

| Category        | Number of Variables |
| --------------- | ------------------: |
| Temperature     |                  10 |
| Pressure        |                   5 |
| Flow            |                   3 |
| Level           |                   2 |
| Heating / Power |                   6 |
| Vacuum          |                   2 |
| Nitrogen        |                   1 |
| Cooling         |                   1 |

---

## Anomaly Labels

The original dataset contains several anomaly states.

For the current binary anomaly-detection task:

* `0` → Normal
* `2` → Detectable anomaly
* `3` → Detectable state / observable aftereffects
* `1` → Unobservable fault

Labels `2` and `3` are combined into:

```text
observable_anomaly = 1
```

Normal windows are represented by:

```text
observable_anomaly = 0
```

Experiments containing only unobservable faults are excluded from the primary evaluation.

---

## Data Splitting

The split is performed at the **experiment level**, rather than randomly splitting individual windows.

This prevents windows from the same experiment appearing in different datasets.

The final split contains:

* **Training:** normal experiments only
* **Validation:** experiments containing observable anomalies
* **Final test:** previously unseen experiments containing observable anomalies
* **Unobservable-fault experiments:** excluded

The validation set is used for:

1. Hyperparameter selection
2. Classification threshold selection

The final test set is evaluated only after these decisions have been frozen.

This prevents test-set information from influencing model selection.

---

## Feature Representations

All four models operate on the same 60-second observations.

### 1. Raw Window Baseline

The baseline uses the original process signals directly.

There are:

```text
30 variables × 60 samples = 1,800 features
```

Each 60-second window is flattened into a single feature vector.

Example:

```text
T701_00
T701_01
...
T701_59
T702_00
...
```

This provides a direct baseline without engineered process features.

---

### 2. Window Features

Statistical and process-aware features are calculated for each 60-second window.

The feature set includes:

* mean
* standard deviation
* minimum
* maximum
* slope
* actuator constant indicators
* temperature correlations
* process-variable differences
* difference statistics
* difference slopes

Total:

```text
240 model features
```

---

### 3. Temporal Features

Temporal information is added by comparing the current window with the previous window from the same experiment.

The representation contains:

* current-window features
* previous-window features
* changes between consecutive windows

Conceptually:

```text
Current
   +
Previous
   +
Delta
```

Total:

```text
720 model features
```

This allows the model to capture changes in process behavior rather than only the state of an individual window.

---

### 4. ML Features

The final representation extends the temporal feature set with frequency-domain information.

Frequency features are calculated from the process signals and added to the temporal representation.

Total:

```text
780 model features
```

---

## Modeling

The project uses **Isolation Forest** for unsupervised anomaly detection.

The models are trained using normal training experiments only.

The anomaly score is then evaluated on validation and final-test observations.

### Hyperparameter Search

The following parameters were evaluated:

```text
n_estimators:
    200
    300
    500

max_samples:
    auto
    0.5
    0.75
    1.0

max_features:
    0.5
    0.75
    1.0
```

This results in:

```text
36 configurations per model
4 models
= 144 model fits
```

Contamination is set to:

```text
auto
```

---

## Model Selection

The primary evaluation metric is **AUPRC (Average Precision)**.

AUPRC is particularly useful for anomaly detection because it focuses on the precision-recall relationship and is less dependent on the large number of normal observations.

F1 score is used as a secondary metric.

The anomaly classification threshold is selected on the validation set by maximizing F1.

The final-test threshold is therefore not optimized on the test set.

---

## Results

### Validation Set

Best validation configurations:

| Model               | Validation AUPRC |
| ------------------- | ---------------: |
| Raw Window Baseline |            0.478 |
| Window Features     |            0.598 |
| Temporal Features   |            0.611 |
| ML Features         |            0.609 |

The **Temporal Features** representation achieved the highest validation AUPRC.

---

### Final Test Set

Final evaluation on previously unseen experiments:

| Model               |     AUPRC |        F1 | Precision |    Recall |
| ------------------- | --------: | --------: | --------: | --------: |
| Raw Window Baseline |     0.455 |     0.473 |     0.347 |     0.741 |
| Window Features     |     0.712 |     0.626 |     0.528 |     0.768 |
| Temporal Features   | **0.763** | **0.629** |     0.490 | **0.879** |
| ML Features         |     0.747 |     0.564 |     0.423 |     0.848 |

### Final Ranking by AUPRC

1. **Temporal Features** — 0.763
2. **ML Features** — 0.747
3. **Window Features** — 0.712
4. **Raw Window Baseline** — 0.455

The results show a substantial improvement from engineered representations compared with the raw-window baseline.

The temporal representation achieved the strongest overall final-test performance in the current experiment.

---

## Streamlit Application

The project includes an interactive Streamlit MVP.

The application contains five sections:

### 1. Project Overview

Provides a concise overview of the problem, dataset, modeling strategy, and project scope.

### 2. Process & Data

Introduces the batch distillation process, dataset structure, process variables, and 60-second windowing approach.

### 3. Feature Engineering

Shows how the raw time-series data are transformed into the four model representations.

### 4. Model Comparison

Displays:

* validation AUPRC
* selected hyperparameters
* validation thresholds
* final-test AUPRC
* final-test F1
* precision
* recall
* model comparison charts

### 5. Anomaly Detection Demo

Allows individual final-test windows to be inspected interactively.

The user can select:

* model
* experiment
* 60-second window
* process variables

The application displays:

* anomaly score
* validation-derived threshold
* model prediction
* ground truth
* experiment metadata
* raw process signals
* anomaly-score timeline for the selected experiment

The demo uses the predictions generated by the modeling pipeline and does not retrain models.

---

## Project Structure

```text
Final Project/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_model_development.ipynb
│   └── 03_visualization.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── temporal_features.py
│   ├── ml_features.py
│   ├── create_split.py
│   └── modeling.py
│
├── requirements.txt
└── README.md
```

---

## Reproducible Pipeline

The preprocessing and modeling steps are implemented as separate Python scripts.

### Step 1 — Preprocessing

```bash
python src/preprocessing.py
```

Creates:

```text
data/processed/
├── operation_data.parquet
└── baseline_raw_windows.parquet
```

### Step 2 — Window Features

```bash
python src/feature_engineering.py
```

Creates:

```text
data/processed/window_features.parquet
```

### Step 3 — Temporal Features

```bash
python src/temporal_features.py
```

Creates:

```text
data/processed/temporal_features.parquet
```

### Step 4 — ML Features

```bash
python src/ml_features.py
```

Creates:

```text
data/processed/ml_features.parquet
```

### Step 5 — Experiment Split

```bash
python src/create_split.py
```

Creates:

```text
data/processed/experiment_split.parquet
```

### Step 6 — Model Training and Evaluation

```bash
python src/modeling.py
```

Creates:

```text
data/processed/model_results/
├── best_configurations.parquet
├── default_model_results.parquet
├── final_test_results.parquet
├── hyperparameter_results.parquet
├── threshold_results.parquet
└── window_predictions.parquet
```

### Step 7 — Streamlit Application

From the project root:

```bash
streamlit run app/app.py
```

The application is then available locally at:

```text
http://localhost:8501
```

---

## Requirements

The project uses:

* Python
* pandas
* NumPy
* PyArrow
* scikit-learn
* Matplotlib
* Seaborn
* Plotly
* Streamlit

Exact package versions are specified in `requirements.txt`.

---

## Future Work

The current project focuses on anomaly detection and model comparison.

Potential extensions include:

### Explainable Fault Diagnosis

Move from simply detecting an anomaly toward identifying:

* which process variables contribute to the anomaly
* which process area is affected
* what type of process deviation may have occurred

This would extend the current anomaly detection approach toward **explainable fault diagnosis**.

### NMR Integration

NMR information could be integrated with process data to investigate whether chemical-composition information improves anomaly detection and fault diagnosis.

### Further Model Development

Potential future investigations include:

* additional anomaly-detection algorithms
* supervised fault classification
* more advanced temporal models
* improved thresholding strategies
* process-aware anomaly aggregation

---

## Key Takeaway

The comparison demonstrates that the representation of process data has a substantial effect on anomaly detection performance.

The raw 60-second signal baseline achieves an AUPRC of **0.455** on the final test set.

Adding process-aware features increases performance to:

| Representation      | Final Test AUPRC |
| ------------------- | ---------------: |
| Raw Window Baseline |            0.455 |
| Window Features     |            0.712 |
| Temporal Features   |        **0.763** |
| ML Features         |            0.747 |

The strongest current representation is therefore the **Temporal Features** model.

This supports the central idea of the project:

> Transforming raw process data into meaningful representations can substantially improve the detection of abnormal process behavior.
