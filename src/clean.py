import pandas as pd
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV data."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw Telco churn dataframe.
    - Drops customerID (identifier, not a feature)
    - Fixes TotalCharges (object -> float, fills NaN with 0)
    - Encodes target: Yes -> 1, No -> 0
    """
    df = df.copy()

    # Drop identifier column
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # TotalCharges has whitespace strings for new customers — coerce to float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Encode target
    if df["Churn"].dtype == object:
        df["Churn"] = (df["Churn"] == "Yes").astype(int)

    return df


def get_feature_types(df: pd.DataFrame):
    """
    Return numeric and categorical column name lists.
    Excludes target and engineered bucket column.
    """
    num_cols = [
        c
        for c in [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "AvgMonthlySpend",
            "NumServices",
        ]
        if c in df.columns
    ]
    cat_cols = [c for c in df.columns if c not in num_cols + ["Churn", "TenureBucket"]]
    return num_cols, cat_cols
