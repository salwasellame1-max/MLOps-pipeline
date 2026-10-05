# 📉 Customer Churn Prediction — MLOps Pipeline

This repository industrializes a churn prediction model into a production-style pipeline: a modular training/inference codebase, a FastAPI service, Docker packaging, automated tests, CI, MLflow experiment tracking and a Model Registry, DVC data versioning, and a data drift monitoring script.

The exploratory data science work — EDA, model comparison, SHAP interpretation, decision-threshold calibration, and the Streamlit demo — lives in a separate repository: **[churn-prediction](LIEN_VERS_TON_AUTRE_REPO)**. This README focuses on the MLOps side; see that repo for the analysis and reasoning behind the modeling choices reused here (model type, hyperparameters, threshold).

![App demo](screenshots/app_high_risk.png)

## 🎯 Problem

Keeping an existing customer is cheaper than acquiring a new one. The goal is to **identify customers at risk of churning** early enough for a retention team to act, and to decide *how aggressively* to flag customers given the cost of missing a churner versus the cost of a retention offer.

## 📊 Dataset

[Telco Customer Churn (IBM)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

- 7,043 customers, 19 features (demographics, services, contract, billing)
- Target: `Churn` (Yes/No), about 27% of customers churn (imbalanced classes)

**Data versioning**: the raw CSV is tracked with [DVC](https://dvc.org) instead of Git, so the repo stays light and the exact dataset version used for any given model is reproducible (`data/raw/*.csv.dvc` holds a content hash of the file). For this project the DVC remote is a local folder, used to demonstrate the workflow (`dvc add`, `dvc push`/`dvc pull`); a team setup would point it to shared cloud storage (S3, GCS, Google Drive) instead. Because of this, GitHub Actions CI cannot `dvc pull` the dataset, so the CI training step currently relies on a copy of the CSV committed separately for that purpose.

## 📡 Monitoring (data drift)

`src/monitoring.py` uses [Evidently](https://www.evidentlyai.com/) to compare the training data (reference) against newer data (current) and flag features whose distribution has drifted, which signals that the model may need retraining. In production, "current" would be a recent export of live customers instead of a held-out split of the training set.

⚠️ **Known limitation**: Evidently currently depends on Pydantic's v1-compatibility layer, which is not yet compatible with Python 3.14. On this environment the script fails at import time (`pydantic.v1.errors.ConfigError`). The script is included to demonstrate the monitoring design; running it requires a Python ≤3.12 environment until Evidently updates this dependency.

## 🛠️ Approach

1. **Cleaning**: fixed `TotalCharges` (stored as text), dropped `customerID`, encoded the target
2. **EDA**: churn rate by contract, tenure, internet service, payment method
3. **Preprocessing**: `StandardScaler` for numeric features and `OneHotEncoder` for categorical features, inside a scikit-learn `Pipeline` to avoid data leakage
4. **Models compared** with stratified 5-fold cross-validation: Logistic Regression, Random Forest, XGBoost (class imbalance handled with class weights)
5. **Evaluation**: ROC-AUC, precision, recall, F1, confusion matrix on a stratified hold-out test set (20%)
6. **Interpretation**: SHAP values
7. **Threshold tuning + sensitivity test** on out-of-fold predictions, then confirmed on the test set
8. **Deployment**: Streamlit app with an adjustable threshold

## 📈 Model results

| Model | CV ROC-AUC | 
|---|---|
| Logistic Regression | `[fill in]` |
| Random Forest | `[fill in]` |
| XGBoost | `[fill in]` |

Selected model: `[fill in]`. Test ROC-AUC: `[fill in]`.

At the default threshold of 0.5 (test set): precision 0.526, recall 0.781, F1 0.629 (82 churners missed, 263 unneeded offers).

## 🔍 What drives churn (SHAP)

![SHAP summary](screenshots/shap_summary.png)

**Increases churn risk**
- Month-to-month contract (strongest driver)
- Short tenure: new customers leave the most
- No online security, tech support or online backup
- Fiber optic internet and electronic check payment
- Senior citizens and customers without dependents

**Reduces churn risk**
- Two-year contract
- Long tenure
- Mailed check payment, DSL internet

Gender has almost no effect, which is a useful sanity check.

> Note: SHAP shows what the *model* relies on, not causal effects. One-hot encoded columns split one variable across several rows (for example `Contract` appears three times).

## 🎚️ Threshold tuning

The model outputs a probability. The default 0.5 cut-off is a convention, not an optimum: missing a churner (lost revenue) usually costs much more than sending a retention offer to someone who would have stayed.

Thresholds were chosen on **out-of-fold predictions from the training set** and then evaluated once on the test set. Costs are **assumptions** (the wasted-offer cost is fixed at 50 and the missed-churner cost is the ratio × 50).

| Cost ratio (missed churner : wasted offer) | Best threshold | Test recall | Test precision | Missed churners | Unneeded offers |
|---|---|---|---|---|---|
| 2 : 1 | 0.58 | 0.698 | 0.544 | 113 | 219 |
| 5 : 1 | **0.37** | 0.845 | 0.473 | 58 | 352 |
| 10 : 1 | 0.14 | 0.957 | 0.376 | 16 | 594 |
| 20 : 1 | 0.10 | 0.971 | 0.358 | 11 | 651 |

**Takeaways**
- The best threshold ranges from 0.58 to 0.10 depending on the cost ratio, so the conclusion depends heavily on a business assumption that should be validated with real data.
- Thresholds chosen on the training folds transfer well to the test set (no sign of overfitting the threshold).
- The app defaults to **0.37** (5:1): about 85% of churners caught while contacting about 47% of customers. A lower threshold catches more churners but means contacting most of the customer base.
- For the 20:1 ratio the search hit the lower bound of the grid (0.10).

## 🚀 Demo app

The Streamlit app takes a customer profile, returns the churn probability and flags the customer as at risk or not. A sidebar slider lets you move the decision threshold and see the decision change.

| High-risk profile | Low-risk profile |
|---|---|
| ![High risk](screenshots/app_high_risk.png) | ![Low risk](screenshots/app_low_risk.png) |

## ▶️ How to run

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
python -m streamlit run app.py
```

To train the model yourself instead of using the one already in `models/`, run `python -m src.train` first (see below).

## 📁 Project structure

```
.
├── .github/workflows/ci.yml   # GitHub Actions: install, train, test on every push
├── src/                       # Production code: config, data, train, predict, api, schemas, monitoring
├── tests/                     # pytest unit tests
├── data/raw/                  # Dataset, versioned with DVC (*.csv.dvc)
├── models/                    # Trained pipeline + metrics (local fallback)
├── mlruns/, mlflow.db         # MLflow tracking store (local, gitignored)
├── app.py                     # Streamlit demo
├── Dockerfile, .dockerignore
├── requirements.txt
├── screenshots/
└── README.md
```

### Training, serving, and testing

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

pip install -r requirements.txt

# Train (creates models/churn_model.joblib, logs to MLflow, registers "champion")
python -m src.train

# Run the tests
python -m pytest

# Serve the API
python -m uvicorn src.api:app --reload
# → http://127.0.0.1:8000/docs

# Or run the Streamlit demo
python -m streamlit run app.py

# Or build and run the Docker image
docker build -t churn-api .
docker run -p 8000:8000 churn-api

# Inspect experiments and the Model Registry
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
# → http://127.0.0.1:5000
```

## ⚠️ Limitations and next steps

- Cost values are assumptions, not real business data.
- One dataset and one train/test split, with no time dimension, so there is no evidence yet about performance on future customers.
- Retention offers do not always work; a real cost model would include their success rate and offer cost for true churners.
- Possible next steps: hyperparameter tuning (Optuna), probability calibration, a capacity-based view ("top 30% riskiest customers"), SHAP explanations inside the app, and deployment on Streamlit Community Cloud.

## 🧰 Tech stack

Python, pandas, scikit-learn, XGBoost, SHAP, matplotlib, seaborn, Streamlit, joblib
