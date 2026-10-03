"""Train the churn model, evaluate it, and save the pipeline + metrics.

Run from the project root:
    python -m src.train
"""
import json

import joblib
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src import config
from src.data import load_data, split_features_target

# All runs of this project are grouped under the same "experiment".
# Explicit tracking store, identical to the one used by `mlflow ui`.
mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
mlflow.set_experiment("churn-prediction")


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

    # Everything inside this "with" block belongs to ONE MLflow run
    with mlflow.start_run():
        # 1. Log the hyperparameters used for this run
        mlflow.log_params(config.XGB_PARAMS)
        mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 4))
        mlflow.log_param("threshold", config.THRESHOLD)
        mlflow.log_param("test_size", config.TEST_SIZE)

        pipeline = build_pipeline(X_train, scale_pos_weight)
        pipeline.fit(X_train, y_train)

        metrics = evaluate(pipeline, X_test, y_test, config.THRESHOLD)
        print("Test metrics:", metrics)

        # 2. Log the metrics obtained on the test set
        mlflow.log_metrics({
            "roc_auc": metrics["roc_auc"],
            "recall": metrics["recall"],
            "precision": metrics["precision"],
        })

        # 3. Log the model itself, so it can be reloaded from MLflow later.
        # skops (MLflow's safe serializer) blocks unknown types by default,
        # so we explicitly trust the two XGBoost types that WE trained.
        # registered_model_name turns this into a new VERSION of "churn-model"
        # in the Model Registry (v1, v2, v3... one per call, automatically).
        model_info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name=config.REGISTERED_MODEL_NAME,
            skops_trusted_types=[
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ],
        )
        print(f"Registered as {config.REGISTERED_MODEL_NAME} v{model_info.registered_model_version}")

        # 4. Only promote this version to "champion" if it beats the current
        # champion's ROC-AUC (or if there is no champion yet). This keeps a
        # worse run from silently becoming the one the API serves.
        client = mlflow.MlflowClient()
        try:
            current = client.get_model_version_by_alias(
                config.REGISTERED_MODEL_NAME, config.CHAMPION_ALIAS
            )
            current_roc_auc = float(
                client.get_run(current.run_id).data.metrics["roc_auc"]
            )
        except mlflow.exceptions.MlflowException:
            current_roc_auc = -1  # no champion exists yet: this run wins by default

        if metrics["roc_auc"] > current_roc_auc:
            client.set_registered_model_alias(
                config.REGISTERED_MODEL_NAME,
                config.CHAMPION_ALIAS,
                model_info.registered_model_version,
            )
            print(f"New champion: v{model_info.registered_model_version} "
                  f"(roc_auc {metrics['roc_auc']} > {current_roc_auc})")
        else:
            print(f"Not promoted: v{model_info.registered_model_version} "
                  f"(roc_auc {metrics['roc_auc']} <= current champion {current_roc_auc})")

        # Keep saving to the same fixed path too, as a simple local fallback
        config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, config.MODEL_PATH)
        config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
        print(f"Model saved to {config.MODEL_PATH}")
        print(f"Metrics saved to {config.METRICS_PATH}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()