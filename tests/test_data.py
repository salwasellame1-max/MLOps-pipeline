"""Tests for src/data.py.

Run from the project root:
    pytest
"""
import pandas as pd

from src.data import split_features_target


def test_split_features_target_separates_correctly():
    """The target column should end up in y, and nowhere in X."""
    df = pd.DataFrame({
        "tenure": [1, 2, 3],
        "MonthlyCharges": [50.0, 60.0, 70.0],
        "Churn": [0, 1, 0],
    })

    X, y = split_features_target(df)

    assert "Churn" not in X.columns
    assert list(y) == [0, 1, 0]
    assert len(X) == len(y) == 3