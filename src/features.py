import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to the dataframe.

    AvgMonthlySpend : normalises TotalCharges by tenure better signal than raw TotalCharges
    NumServices     : count of active services engaged customers churn less
    TenureBucket    : categorical tenure band new customers churn most
    """
    df = df.copy()

    # Average monthly spend
    df["AvgMonthlySpend"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
    )

    # Number of active services
    service_cols = [
        c
        for c in [
            "PhoneService",
            "InternetService",
            "OnlineSecurity",
            "StreamingTV",
            "StreamingMovies",
            "DeviceProtection",
            "TechSupport",
            "OnlineBackup",
        ]
        if c in df.columns
    ]
    df["NumServices"] = (df[service_cols] != "No").sum(axis=1)

    # Tenure bucket
    df["TenureBucket"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["new", "growing", "established", "loyal"],
        include_lowest=True,
    )

    return df
