from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "churn_model.joblib"
DEFAULT_THRESHOLD = 0.37  # balanced choice from the sensitivity test (5:1 cost ratio)

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

# ---------------- Sidebar: decision threshold ----------------
st.sidebar.header("⚙️ Decision threshold")
threshold = st.sidebar.slider(
    "Flag a customer as 'at risk' if probability ≥",
    min_value=0.05, max_value=0.90, value=DEFAULT_THRESHOLD, step=0.01,
)
st.sidebar.markdown(
    """
**How to choose it** (from the sensitivity test)

| Threshold | Meaning |
|---|---|
| ~0.58 | Few offers, but ~30% of churners missed |
| ~0.37 | Balanced (default) |
| ~0.14 | Catch ~96% of churners, contact ~2/3 of customers |
"""
)
st.sidebar.caption("The threshold is a business decision: it depends on the cost of a missed churner vs a retention offer.")

# ---------------- Main: customer form ----------------
st.title("📉 Customer Churn Predictor")
st.write("Enter a customer's profile to estimate their probability of leaving.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Customer")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)

with col2:
    st.subheader("📡 Services")
    phone_service = st.selectbox("Phone service", ["Yes", "No"])
    multiple_lines = st.selectbox(
        "Multiple lines", ["No", "Yes"], disabled=(phone_service == "No")
    )
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])

    addon_names = {
        "OnlineSecurity": "Online security",
        "OnlineBackup": "Online backup",
        "DeviceProtection": "Device protection",
        "TechSupport": "Tech support",
        "StreamingTV": "Streaming TV",
        "StreamingMovies": "Streaming movies",
    }
    addons = {}
    for key, label in addon_names.items():
        addons[key] = st.selectbox(label, ["No", "Yes"], disabled=(internet == "No"), key=key)

with col3:
    st.subheader("💳 Billing")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    monthly = st.number_input("Monthly charges", 15.0, 130.0, 70.0, step=1.0)
    st.caption("Total charges are estimated as tenure × monthly charges.")

# ---------------- Build the row exactly as the model expects ----------------
# Values the dataset uses when a service does not apply
if phone_service == "No":
    multiple_lines = "No phone service"
if internet == "No":
    addons = {k: "No internet service" for k in addons}

row = {
    "gender": gender,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet,
    **addons,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": tenure * monthly,
}
X_new = pd.DataFrame([row])

# ---------------- Prediction ----------------
proba = float(model.predict_proba(X_new)[0, 1])
at_risk = proba >= threshold

st.divider()
left, right = st.columns([1, 2])

with left:
    st.metric("Churn probability", f"{proba:.1%}")
    st.caption(f"Current threshold: {threshold:.2f}")

with right:
    st.progress(min(max(proba, 0.0), 1.0))
    if at_risk:
        st.error("🔴 **At risk**: probability is above the threshold. Consider a retention action.")
    else:
        st.success("🟢 **Low risk**: probability is below the threshold.")

with st.expander("See the data sent to the model"):
    st.dataframe(X_new.T.rename(columns={0: "value"}))
