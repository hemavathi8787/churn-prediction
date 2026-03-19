import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.pipeline import build_full_pipeline, get_data_splits

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
EXPERIMENT_NAME = "churn-prediction"
TRACKING_URI = "http://127.0.0.1:5000"


def evaluate_model(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
    }


def train_and_log(
    model_name, classifier, params, X_train, X_test, y_train, y_test, num_cols, cat_cols
):
    """Train one model, log everything to MLflow, register model."""
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_params(params)

        pipeline = build_full_pipeline(classifier, num_cols, cat_cols)
        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_test, y_test)
        mlflow.log_metrics(metrics)

        # MLflow 3.x: use name= instead of artifact_path=
        mlflow.sklearn.log_model(
            pipeline, name="model", registered_model_name=f"churn-{model_name}"
        )

        print(f"\n{model_name}")
        print(f"  ROC-AUC:   {metrics['roc_auc']}")
        print(f"  F1:        {metrics['f1']}")
        print(f"  Precision: {metrics['precision']}")
        print(f"  Recall:    {metrics['recall']}")

    return metrics, pipeline


def register_best_model(model_name: str = "xgboost"):
    """
    Find best run for a given model, register it,
    and set the 'production' alias (MLflow 3.x compatible).
    """
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"params.model_name = '{model_name}'",
        order_by=["metrics.roc_auc DESC"],
        max_results=1,
    )

    if not runs:
        print(f"No runs found for {model_name}")
        return

    best_run = runs[0]
    run_id = best_run.info.run_id
    roc_auc = best_run.data.metrics.get("roc_auc", 0)
    print(f"\nBest {model_name} run: {run_id} (ROC-AUC: {roc_auc})")

    model_uri = f"runs:/{run_id}/model"
    model_name_reg = f"churn-{model_name}"

    mv = mlflow.register_model(model_uri, model_name_reg)
    print(f"Registered: {model_name_reg} version {mv.version}")

    # MLflow 3.x uses aliases instead of stages
    try:
        client.set_registered_model_alias(
            name=model_name_reg, alias="production", version=mv.version
        )
        print(f"Alias 'production' set on version {mv.version}")
    except Exception as e:
        # Fallback to stage transition for older MLflow
        try:
            client.transition_model_version_stage(
                name=model_name_reg, version=mv.version, stage="Production"
            )
            print(f"Stage set to Production (version {mv.version})")
        except Exception:
            print(f"Could not set stage/alias: {e}")

    return mv


def run_all_experiments():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test, num_cols, cat_cols = get_data_splits(DATA_PATH)

    experiments = [
        (
            "logistic-regression",
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            {"max_iter": 1000, "class_weight": "balanced"},
        ),
        (
            "xgboost",
            XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                eval_metric="logloss",
                random_state=42,
            ),
            {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
            },
        ),
        (
            "lightgbm",
            LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=31,
                class_weight="balanced",
                random_state=42,
                verbose=-1,
            ),
            {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.05,
                "num_leaves": 31,
            },
        ),
    ]

    results = {}
    for name, clf, params in experiments:
        metrics, pipeline = train_and_log(
            name, clf, params, X_train, X_test, y_train, y_test, num_cols, cat_cols
        )
        results[name] = (metrics, pipeline)

    # Summary
    print("\n--- Summary ---")
    best = max(results.items(), key=lambda x: x[1][0]["roc_auc"])
    print(f"Best model: {best[0]} (ROC-AUC: {best[1][0]['roc_auc']})")

    # Auto-register best model
    print("\n--- Registering best model ---")
    register_best_model("xgboost")

    return results


if __name__ == "__main__":
    run_all_experiments()
