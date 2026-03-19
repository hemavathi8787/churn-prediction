import pandas as pd
import numpy as np
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric
from src.clean import load_data, clean_data
from src.features import engineer_features


def load_reference_data(path: str) -> pd.DataFrame:
    df = load_data(path)
    df = clean_data(df)
    df = engineer_features(df)
    return df


def simulate_production_drift(df: pd.DataFrame) -> pd.DataFrame:
    drifted = df.copy()
    np.random.seed(99)
    n = len(drifted)
    drifted["MonthlyCharges"] *= np.random.uniform(1.10, 1.20, n)
    new_mask = np.random.rand(n) < 0.3
    drifted.loc[new_mask, "tenure"] = np.random.randint(1, 6, new_mask.sum())
    drifted["AvgMonthlySpend"] = np.where(
        drifted["tenure"] > 0,
        drifted["TotalCharges"] / drifted["tenure"],
        drifted["MonthlyCharges"],
    )
    return drifted


def parse_drift_results(result: dict) -> dict:
    """Works across all Evidently versions by scanning all metrics."""
    n_drifted = 0
    n_total = 0
    drift_share = 0.0

    for metric in result.get("metrics", []):
        r = metric.get("result", {})

        # Method 1 — DatasetDriftMetric format
        if "number_of_drifted_columns" in r:
            n_drifted = r["number_of_drifted_columns"]
            n_total = r.get("number_of_columns", 0)
            if n_total > 0:
                drift_share = n_drifted / n_total
            break

        # Method 2 — drift_share directly available
        if "drift_share" in r and r["drift_share"] > 0:
            drift_share = r["drift_share"]
            n_drifted = r.get("number_of_drifted_columns", 0)
            n_total = r.get("number_of_columns", 0)
            break

        # Method 3 — nested under dataset_drift
        if "dataset_drift" in r:
            inner = r["dataset_drift"]
            n_drifted = inner.get("number_of_drifted_columns", 0)
            n_total = inner.get("number_of_columns", 0)
            if n_total > 0:
                drift_share = n_drifted / n_total
            break

    return {"n_drifted": n_drifted, "n_total": n_total, "drift_share": drift_share}


def run_drift_report(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    output_path: str = "reports/drift_report.html",
):
    os.makedirs("reports", exist_ok=True)

    feature_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "AvgMonthlySpend",
        "NumServices",
    ]

    ref = reference[feature_cols].reset_index(drop=True)
    cur = production[feature_cols].reset_index(drop=True)

    report = Report(
        metrics=[
            DataDriftPreset(),
            DatasetDriftMetric(),
        ]
    )

    report.run(reference_data=ref, current_data=cur)
    report.save_html(output_path)
    print(f"Report saved → {output_path}")

    # Debug: print raw structure to find correct keys
    result = report.as_dict()
    print("\n--- Drift Results ---")
    for i, metric in enumerate(result.get("metrics", [])):
        print(f"Metric {i}: {list(metric.get('result', {}).keys())}")

    # Parse correctly across versions
    parsed = parse_drift_results(result)
    n_drifted = parsed["n_drifted"]
    n_total = parsed["n_total"]
    drift_share = parsed["drift_share"]

    print(f"\nDrifted features : {n_drifted} / {n_total}")
    print(f"Drift share      : {drift_share * 100:.1f}%")

    if n_drifted >= 3:
        print("ALERT: 3+ features drifted — retraining recommended!")
        return True
    print("No significant drift detected.")
    return False


if __name__ == "__main__":
    DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    reference = load_reference_data(DATA_PATH)
    production = simulate_production_drift(reference)
    should_retrain = run_drift_report(reference, production)
