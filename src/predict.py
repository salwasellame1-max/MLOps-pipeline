"""Load the trained pipeline and make predictions.

Try it from the project root:
    python -m src.predict
"""
from functools import lru_cache

import pandas as pd

import joblib
from src import config

# The exact columns the model was trained on (the "contract" with the caller)
FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


@lru_cache(maxsize=1)
def load_model():
    """Load the pipeline once and keep it in memory (loading from disk is slow)."""
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No model at {config.MODEL_PATH}. Run `python -m src.train` first."
        )
    return joblib.load(config.MODEL_PATH)


def predict_proba(customers: list[dict]) -> list[float]:
    """Churn probability for each customer (a customer = a dict of the 19 features)."""
    df = pd.DataFrame(customers)

    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")

    df = df[FEATURE_COLUMNS]  # fixes column order and drops unexpected extras
    return load_model().predict_proba(df)[:, 1].tolist()


def predict(customer: dict, threshold: float = config.THRESHOLD) -> dict:
    """Predict for ONE customer and return a JSON-friendly result."""
    proba = predict_proba([customer])[0]
    return {
        "churn_probability": round(proba, 4),
        "at_risk": bool(proba >= threshold),
        "threshold": threshold,
    }


if __name__ == "__main__":
    example = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 90.0, "TotalCharges": 180.0,
    }
    print(predict(example))