# Telecom Customers Churn

A clean, reproducible telecom customer churn project with a Python package, data pipeline, and Docker support.

## Project overview

This repository is structured for production-ready analysis and modeling:

- `data/raw/` - store the original input dataset here.
- `data/raw/` - store the original input dataset here.
- `notebook/` - exploratory Jupyter notebooks and analysis files.
- `src/` - Python source package containing data loading, feature engineering, classification, and clustering logic.
- `run.py` - primary entrypoint for running the classification pipeline.
- `Dockerfile` - container definition for running the project inside Docker.
- `.dockerignore` - files excluded from Docker builds.
- `requirements.txt` - Python dependencies.

## How it works

1. `src/data/load_data.py` loads the raw `Telecom Customers Churn.csv` dataset from `data/raw/`.
2. `src/feature_engineering.py` cleans the dataset, encodes categorical values, scales numeric features, and creates domain features such as contract length and churn risk.
3. `src/modeling.py` balances the dataset using `SMOTEENN`, runs grid search across multiple classifiers, and evaluates performance with confusion matrices and ROC AUC.
4. `src/telecom_churn_workflow.py` glues the full modeling pipeline together for one-call execution.
5. `src/clustering.py` provides optional clustering support for exploratory segmentation.

## Setup

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Place the dataset file in the raw data folder:

```bash
move "Telecom Customers Churn.csv" data\raw\
```

> Use `move` on Windows PowerShell and `mv` on macOS/Linux.

## Run the classification pipeline

Execute the main entrypoint:

```bash
python run.py
```

This runs the end-to-end churn modeling pipeline and prints summary results.

## Use the package from Python

```python
from src import run_classification_pipeline
results = run_classification_pipeline()
print(results)
```

## Run clustering

```python
from src import (
    build_model_dataset,
    select_cluster_features,
    run_kmeans,
    elbow_plot,
)

X, y, df = build_model_dataset()
cluster_features = select_cluster_features(df, [
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
])
elbow_plot(cluster_features)
model = run_kmeans(cluster_features, n_clusters=6)
```

## Docker

Build the Docker image:

```bash
docker build -t telecom-churn:latest .
```

Run the container:

```bash
docker run --rm telecom-churn:latest
```

## Notes

- `run.py` is the project entrypoint.
- `src/telecom_churn_workflow.py` handles the classification pipeline.
- `src/clustering.py` supports customer segmentation analysis.
- `src/config.py` centralizes directory and file defaults.
- Use `data/raw/` for the source dataset.
- `notebook/` contains the notebook file.
