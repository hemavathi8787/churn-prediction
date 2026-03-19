import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = mlflow.MlflowClient()

model = client.get_registered_model("churn-xgboost")
print("Model name :", model.name)
print("Aliases    :", model.aliases)
print("Ready to serve!")
