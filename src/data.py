"""Data loading and cleaning."""
import pandas as pd

from src.config import DATA_PATH, DROP_COLS, TARGET


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV and apply the same cleaning as in the notebook."""
    df = pd.read_csv(path)

    # TotalCharges is stored as text; blanks are customers with tenure 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    df = df.drop(columns=DROP_COLS)
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def split_features_target(df: pd.DataFrame):
    """Separate the feature matrix X from the target y."""
    return df.drop(columns=TARGET), df[TARGET]