"""Tests for src/predict.py.

These need a trained model to exist (run `python -m src.train` first),
since they load the real pipeline rather than a fake one.
Run from the project root:
    pytest
"""
import pytest

from src.predict import predict

HIGH_RISK_CUSTOMER = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 90.0, "TotalCharges": 180.0,
}

LOW_RISK_CUSTOMER = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
    "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
    "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
    "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
    "PaymentMethod": "Mailed check", "MonthlyCharges": 50.0, "TotalCharges": 3000.0,
}


def test_prediction_returns_expected_keys():
    result = predict(HIGH_RISK_CUSTOMER)
    assert set(result.keys()) == {"churn_probability", "at_risk", "threshold"}


def test_probability_is_between_0_and_1():
    result = predict(HIGH_RISK_CUSTOMER)
    assert 0.0 <= result["churn_probability"] <= 1.0


def test_high_risk_profile_scores_higher_than_low_risk():
    """Sanity check: the model should rank a risky profile above a safe one."""
    high = predict(HIGH_RISK_CUSTOMER)["churn_probability"]
    low = predict(LOW_RISK_CUSTOMER)["churn_probability"]
    assert high > low


def test_missing_feature_raises_clear_error():
    incomplete_customer = {"gender": "Female"}  # missing the other 18 features
    with pytest.raises(ValueError):
        predict(incomplete_customer)