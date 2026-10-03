# 📉 Customer Churn Prediction

Predicting which telecom customers are likely to leave, explaining why with SHAP, and choosing the decision threshold from a business-cost point of view. Includes an interactive Streamlit demo.

![App demo](screenshots/app_high_risk.png)

## 🎯 Problem

Keeping an existing customer is cheaper than acquiring a new one. The goal is to **identify customers at risk of churning** early enough for a retention team to act, and to decide *how aggressively* to flag customers given the cost of missing a churner versus the cost of a retention offer.

## 📊 Dataset

[Telco Customer Churn (IBM)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

- 7,043 customers, 19 features (demographics, services, contract, billing)
- Target: `Churn` (Yes/No), about 27% of customers churn (imbalanced classes)

**Data versioning**: the raw CSV is tracked with [DVC](https://dvc.org) instead of Git, so the repo stays light and the exact dataset version used for any given model is reproducible (`data/raw/*.csv.dvc` holds a content hash of the file). For this project the DVC remote is a local folder, used to demonstrate the workflow (`dvc add`, `dvc push`/`dvc pull`); a team setup would point it to shared cloud storage (S3, GCS, Google Drive) instead. Because of this, GitHub Actions CI cannot `dvc pull` the dataset, so the CI training step currently relies on a copy of the CSV committed separately for that purpose.

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

The app needs `churn_model.joblib` in the same folder. To regenerate it, run the notebook, whose last cell saves the trained pipeline. The versions of `scikit-learn` and `xgboost` should match the ones used for training.

## 📁 Project structure

```
.
├── churn_customer_prediction.ipynb   # EDA, modeling, SHAP, threshold tuning
├── app.py                            # Streamlit demo
├── churn_model.joblib                # Trained pipeline
├── requirements.txt
├── screenshots/                      # Images used in this README
└── README.md
```

## ⚠️ Limitations and next steps

- Cost values are assumptions, not real business data.
- One dataset and one train/test split, with no time dimension, so there is no evidence yet about performance on future customers.
- Retention offers do not always work; a real cost model would include their success rate and offer cost for true churners.
- Possible next steps: hyperparameter tuning (Optuna), probability calibration, a capacity-based view ("top 30% riskiest customers"), SHAP explanations inside the app, and deployment on Streamlit Community Cloud.

## 🧰 Tech stack

Python, pandas, scikit-learn, XGBoost, SHAP, matplotlib, seaborn, Streamlit, joblib
