"""Central configuration: every path, constant and hyperparameter lives here."""
from pathlib import Path

# Project root = parent of the src/ folder
ROOT = Path(__file__).resolve().parent.parent

# Paths
DATA_PATH = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

# Data
TARGET = "Churn"
DROP_COLS = ["customerID"]

# Training
RANDOM_STATE = 42
TEST_SIZE = 0.2
THRESHOLD = 0.37  # decision threshold chosen in the sensitivity test (5:1 cost ratio)

# Model hyperparameters (same as the notebook)
XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
}