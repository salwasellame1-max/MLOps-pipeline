"""Train the churn model, evaluate it, and save the pipeline + metrics.

Run from the project root:
    python -m src.train
"""
import json

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src import config
from src.data import load_data, split_features_target


def build_pipeline(X, scale_pos_weight: float) -> Pipeline:
    """Preprocessing (scaling + one-hot) and the XGBoost model in one pipeline."""
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    preprocess = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    model = XGBClassifier(scale_pos_weight=scale_pos_weight, **config.XGB_PARAMS)
    return Pipeline([("prep", preprocess), ("clf", model)])


def evaluate(pipeline, X_test, y_test, threshold: float) -> dict:
    """Compute ROC-AUC (threshold-free) and precision/recall at the chosen threshold."""
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "recall": round(float(recall_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred)), 4),
        "threshold": threshold,
    }


def main():
    df = load_data()
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )

    # Class imbalance: weight the positive class (churners) accordingly
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    pipeline = build_pipeline(X_train, scale_pos_weight)
    pipeline.fit(X_train, y_train)

    metrics = evaluate(pipeline, X_test, y_test, config.THRESHOLD)
    print("Test metrics:", metrics)

    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Model saved to {config.MODEL_PATH}")
    print(f"Metrics saved to {config.METRICS_PATH}")


if __name__ == "__main__":
    main()