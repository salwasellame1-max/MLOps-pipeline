# ⚙️ Customer Churn Prediction — MLOps Pipeline

Production-style MLOps pipeline that industrializes a churn prediction model: a modular training/inference codebase, a FastAPI service, Docker packaging, automated tests, CI, MLflow experiment tracking and a Model Registry, DVC data versioning, and a data drift monitoring script.

> 📓 The exploratory data science work (EDA, model comparison, SHAP interpretation, decision-threshold calibration, Streamlit demo) lives in a separate repository: **[churn-prediction](LIEN_VERS_TON_AUTRE_REPO)**. This repo takes the model choices validated there (XGBoost, threshold = 0.37) and turns them into a deployable, tested, tracked pipeline.

## 🏗️ Architecture

```
            ┌──────────────┐
  raw data ─▶   src/data.py │
            └──────┬───────┘
                   ▼
            ┌──────────────┐        ┌───────────────────┐
            │ src/train.py │──────▶ │ MLflow: experiment │
            │ (pipeline +  │        │ tracking + Model   │
            │  XGBoost)    │        │ Registry (champion)│
            └──────┬───────┘        └───────────────────┘
                   ▼
            ┌──────────────┐
            │ models/*.joblib (local fallback)
            └──────┬───────┘
                   ▼
        ┌──────────────────┐
        │ src/predict.py    │  loads the "champion" model
        └─────────┬─────────┘
                   ▼
        ┌──────────────────┐        ┌──────────────┐
        │ src/api.py        │◀──────▶│ Docker image │
        │ (FastAPI service) │        └──────────────┘
        └──────────────────┘

  src/monitoring.py  →  Evidently drift report (reference vs current data)
  tests/              →  pytest, run locally and in CI
  .github/workflows/  →  CI: install → train → test, on every push
  data/raw/*.csv.dvc   →  dataset tracked with DVC (hash-based versioning)
```

## 📁 Project structure

```
.
├── .github/workflows/ci.yml   # CI: install deps, train, run tests on every push
├── src/
│   ├── config.py               # Paths, hyperparameters, threshold, MLflow settings
│   ├── data.py                 # Load + clean the dataset
│   ├── train.py                # Build pipeline, train, evaluate, log to MLflow, register model
│   ├── predict.py               # Load the "champion" model, run predictions
│   ├── api.py                   # FastAPI service (/health, /predict)
│   ├── schemas.py                # Pydantic request/response models
│   └── monitoring.py             # Evidently data drift report
├── tests/                        # pytest unit tests
├── data/raw/                      # Dataset, versioned with DVC (*.csv.dvc)
├── models/                        # Trained pipeline + metrics.json (local fallback)
├── mlruns/, mlflow.db             # MLflow tracking store (local, gitignored)
├── app.py                         # Streamlit demo calling the trained pipeline
├── Dockerfile, .dockerignore
├── requirements.txt
└── README.md
```

## ▶️ How to run

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt

# Train: creates models/churn_model.joblib, logs params/metrics to MLflow,
# registers the model and promotes it to "champion" if it beats the current one
python -m src.train

# Run the test suite
python -m pytest

# Serve the API (loads the "champion" model from the MLflow registry)
python -m uvicorn src.api:app --reload
# → http://127.0.0.1:8000/docs

# Build and run the same service in Docker
docker build -t churn-api .
docker run -p 8000:8000 churn-api

# Inspect experiments and the Model Registry
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
# → http://127.0.0.1:5000

# Optional: Streamlit demo
python -m streamlit run app.py

# Optional: data drift report (see limitation below)
python -m src.monitoring
```

## 🧪 Testing & CI

`tests/` covers the data-cleaning logic and the prediction function (output shape, valid probability range, high-risk vs low-risk ranking, clear error on missing features). `.github/workflows/ci.yml` runs on every push: installs dependencies, retrains the model from scratch on a clean machine, and runs the full test suite — so a broken change can't merge silently.

## 📊 MLflow: experiment tracking & Model Registry

Every call to `python -m src.train` is logged as an MLflow **run**: hyperparameters, test metrics (ROC-AUC, recall, precision), and the trained pipeline itself. Runs are grouped under the `churn-prediction` **experiment** and compared side by side in the MLflow UI.

The model is also registered under the name `churn-model`, with one numbered **version** per training call. A version is promoted to the `champion` **alias** only if its ROC-AUC beats the current champion's — so a worse run can never silently replace a better model. `src/predict.py` always loads `models:/churn-model@champion`, so promoting a new version (or rolling back) never requires touching code or copying files.

## 🗃️ DVC: dataset versioning

The raw CSV is tracked with [DVC](https://dvc.org) instead of Git: Git stores only a small `.csv.dvc` file (a content hash), not the dataset itself, so the exact data version behind any model stays reproducible without bloating the repo.

⚠️ **Known limitation**: the DVC remote used here is a local folder, set up to demonstrate the `dvc add` / `dvc push` / `dvc pull` workflow. A team setup would point it to shared cloud storage (S3, GCS, Google Drive) instead. Because of this, GitHub Actions CI cannot `dvc pull` the dataset, so the CI training step currently fails there — a deliberate, documented trade-off rather than an oversight.

## 📡 Monitoring: data drift

`src/monitoring.py` uses [Evidently](https://www.evidentlyai.com/) to compare the training data (reference) against newer data (current) and flag features whose distribution has drifted — a signal that the model may need retraining. In production, "current" would be a recent export of live customers instead of a held-out split of the training set.

⚠️ **Known limitation**: Evidently currently depends on Pydantic's v1-compatibility layer, which is not yet compatible with Python 3.14. On this environment the script fails at import time (`pydantic.v1.errors.ConfigError`). The script is included to demonstrate the monitoring design; running it requires a Python ≤3.12 environment until Evidently updates this dependency.

## 🐳 Docker

The `Dockerfile` builds a self-contained image (Python 3.11-slim + dependencies + `src/` + `models/`) that serves the same FastAPI app via uvicorn, so the API runs identically on any machine.

## ⚠️ Limitations and next steps

- DVC remote and dataset access in CI are local-only for this demo (see above).
- Evidently monitoring isn't runnable on this project's Python version yet (see above).
- The champion-promotion rule compares raw ROC-AUC from a single train/test split; a more robust setup would use cross-validated scores and a statistical significance check before promoting.
- Possible next steps: cloud deployment (e.g. Render) with a public API URL, a real cloud DVC remote + `dvc pull` in CI, hyperparameter tuning logged to MLflow, scheduled retraining triggered by drift detection.

## 🧰 Tech stack

Python, scikit-learn, XGBoost, FastAPI, Pydantic, Docker, pytest, GitHub Actions, MLflow, DVC, Evidently, Streamlit, joblib
