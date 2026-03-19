import os
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
# MLflow 3.x uses aliases: models:/churn-xgboost@production
# MLflow 2.x uses stages:  models:/churn-xgboost/Production
MODEL_URI = os.getenv("MODEL_URI", "models:/churn-xgboost@production")

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    global model
    mlflow.set_tracking_uri(TRACKING_URI)
    try:
        model = mlflow.sklearn.load_model(MODEL_URI)
        print(f"Model loaded successfully: {MODEL_URI}")
    except Exception as e:
        print(f"WARNING: Could not load model: {e}")
        print("API will start but /predict will return 503 until model is available")
    yield
    model = None


app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn probability using XGBoost",
    version="1.0.0",
    lifespan=lifespan,
)


class CustomerFeatures(BaseModel):
    gender: str = Field(example="Male")
    SeniorCitizen: int = Field(example=0)
    Partner: str = Field(example="Yes")
    Dependents: str = Field(example="No")
    tenure: int = Field(example=2)
    PhoneService: str = Field(example="Yes")
    MultipleLines: str = Field(example="No")
    InternetService: str = Field(example="Fiber optic")
    OnlineSecurity: str = Field(example="No")
    OnlineBackup: str = Field(example="No")
    DeviceProtection: str = Field(example="No")
    TechSupport: str = Field(example="No")
    StreamingTV: str = Field(example="Yes")
    StreamingMovies: str = Field(example="Yes")
    Contract: str = Field(example="Month-to-month")
    PaperlessBilling: str = Field(example="Yes")
    PaymentMethod: str = Field(example="Electronic check")
    MonthlyCharges: float = Field(example=85.0)
    TotalCharges: float = Field(example=170.0)


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_level: str
    message: str


@app.get("/")
def root():
    return {"message": "Churn Prediction API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "model_uri": MODEL_URI}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training and registration first.",
        )

    from src.features import engineer_features

    # Convert input to DataFrame
    data = pd.DataFrame([customer.model_dump()])

    # Engineer features (same as training)
    data = engineer_features(data)

    # Drop TenureBucket — not used in pipeline
    data = data.drop(columns=["TenureBucket"], errors="ignore")

    # Predict
    prob = float(model.predict_proba(data)[:, 1][0])
    pred = prob >= 0.5

    # Risk bucketing
    if prob > 0.7:
        risk = "high"
        message = "Customer is very likely to churn. Immediate action recommended."
    elif prob > 0.4:
        risk = "medium"
        message = "Customer shows some churn signals. Consider a retention offer."
    else:
        risk = "low"
        message = "Customer is likely to stay."

    return PredictionResponse(
        churn_probability=round(prob, 3),
        churn_prediction=pred,
        risk_level=risk,
        message=message,
    )
