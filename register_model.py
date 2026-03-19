import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = mlflow.MlflowClient()

experiment = client.get_experiment_by_name("churn-prediction")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="params.model_name = 'xgboost'",
    order_by=["metrics.roc_auc DESC"],
    max_results=1,
)

run_id = runs[0].info.run_id
print(f"Run ID: {run_id}")

model_uri = f"runs:/{run_id}/model"
mv = mlflow.register_model(model_uri, "churn-xgboost")
print(f"Registered version: {mv.version}")

client.set_registered_model_alias(
    name="churn-xgboost", alias="production", version=mv.version
)
print("Alias set: production")
print("DONE - model is ready for serving!")
