import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.clean import clean_data
from src.features import engineer_features


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def raw_df():
    """Minimal mock dataframe that matches Telco schema."""
    return pd.DataFrame(
        {
            "customerID": ["1", "2", "3"],
            "gender": ["Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0],
            "Partner": ["Yes", "No", "Yes"],
            "Dependents": ["No", "No", "Yes"],
            "tenure": [1, 24, 60],
            "PhoneService": ["Yes", "Yes", "No"],
            "MultipleLines": ["No", "Yes", "No phone service"],
            "InternetService": ["Fiber optic", "DSL", "No"],
            "OnlineSecurity": ["No", "Yes", "No internet service"],
            "OnlineBackup": ["Yes", "No", "No internet service"],
            "DeviceProtection": ["No", "Yes", "No internet service"],
            "TechSupport": ["No", "No", "No internet service"],
            "StreamingTV": ["Yes", "No", "No internet service"],
            "StreamingMovies": ["No", "Yes", "No internet service"],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "PaperlessBilling": ["Yes", "No", "Yes"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
            ],
            "MonthlyCharges": [70.0, 55.0, 20.0],
            "TotalCharges": ["70.0", "1320.0", " "],  # whitespace = new customer bug
            "Churn": ["Yes", "No", "No"],
        }
    )


@pytest.fixture
def clean_df(raw_df):
    return clean_data(raw_df)


@pytest.fixture
def featured_df(clean_df):
    return engineer_features(clean_df)


# ── Clean data tests ──────────────────────────────────────────────────────────


def test_customerid_dropped(clean_df):
    assert "customerID" not in clean_df.columns


def test_total_charges_is_numeric(clean_df):
    assert clean_df["TotalCharges"].dtype in ["float64", "float32"]


def test_total_charges_no_nulls(clean_df):
    assert clean_df["TotalCharges"].isnull().sum() == 0


def test_whitespace_total_charges_filled_zero(clean_df):
    # Row 3 had whitespace — should be filled with 0
    assert clean_df["TotalCharges"].iloc[2] == 0.0


def test_churn_is_binary(clean_df):
    assert set(clean_df["Churn"].unique()).issubset({0, 1})


def test_churn_yes_encoded_as_1(clean_df):
    assert clean_df["Churn"].iloc[0] == 1


def test_churn_no_encoded_as_0(clean_df):
    assert clean_df["Churn"].iloc[1] == 0


# ── Feature engineering tests ─────────────────────────────────────────────────


def test_avg_monthly_spend_column_exists(featured_df):
    assert "AvgMonthlySpend" in featured_df.columns


def test_num_services_column_exists(featured_df):
    assert "NumServices" in featured_df.columns


def test_tenure_bucket_column_exists(featured_df):
    assert "TenureBucket" in featured_df.columns


def test_avg_monthly_spend_non_negative(featured_df):
    assert (featured_df["AvgMonthlySpend"] >= 0).all()


def test_num_services_range(featured_df):
    assert featured_df["NumServices"].min() >= 0
    assert featured_df["NumServices"].max() <= 8


def test_tenure_bucket_new_customer(featured_df):
    # tenure=1 should be 'new'
    assert str(featured_df["TenureBucket"].iloc[0]) == "new"


def test_tenure_bucket_loyal_customer(featured_df):
    # tenure=60 should be 'loyal'
    assert str(featured_df["TenureBucket"].iloc[2]) == "loyal"


def test_no_new_nulls_introduced(featured_df):
    # AvgMonthlySpend and NumServices must have no nulls
    assert featured_df["AvgMonthlySpend"].isnull().sum() == 0
    assert featured_df["NumServices"].isnull().sum() == 0
