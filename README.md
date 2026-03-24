# Churn Prediction — End-to-End ML Project

A production-grade customer churn prediction system built with
scikit-learn, XGBoost, MLflow, FastAPI, and Evidently AI.

---

## Results

| Model               | ROC-AUC   | F1    | Precision | Recall |
| ------------------- | --------- | ----- | --------- | ------ |
| XGBoost             | **0.842** | 0.598 | 0.579     | 0.618  |
| LightGBM            | 0.838     | 0.593 | 0.577     | 0.610  |
| Logistic Regression | 0.838     | 0.616 | 0.503     | 0.794  |

Best model: **XGBoost** (ROC-AUC 0.842)

---

## Project Structure

churn-prediction/
├── data/ # Raw dataset
├── src/
│ ├── clean.py # Data cleaning
│ ├── features.py # Feature engineering
│ ├── pipeline.py # Scikit-learn pipeline
│ ├── train.py # Model training + MLflow logging
│ ├── api.py # FastAPI serving
│ └── monitor.py # Evidently drift monitoring
├── notebooks/
│ └── 01_eda.ipynb # Exploratory data analysis
├── tests/
│ └── test_pipeline.py # Unit tests
├── reports/
│ └── drift_report.html # Evidently drift report
├── requirements.txt
└── README.md

---

## Quickstart

### 1. Install dependencies

pip install -r requirements.txt

### 2. Train models

mlflow ui --host 127.0.0.1 --port 5000 # Terminal 1
python -m src.train # Terminal 2

### 3. Register best model

python register_model.py

### 4. Serve predictions

uvicorn src.api:app --reload --port 8000

### 5. Test the API

python test_prediction.py

### 6. Run drift monitoring

python -m src.monitor

### 7. Run tests

pytest tests/ -v

---

## API Endpoints

| Method | Endpoint | Description          |
| ------ | -------- | -------------------- |
| GET    | /health  | Health check         |
| POST   | /predict | Get churn prediction |
| GET    | /docs    | Swagger UI           |

### Example prediction request

POST http://127.0.0.1:8000/predict

{
"gender": "Male",
"SeniorCitizen": 0,
"tenure": 2,
"Contract": "Month-to-month",
"MonthlyCharges": 85.0,
"TotalCharges": 170.0,
...
}

### Example response

{
"churn_probability": 0.862,
"churn_prediction": true,
"risk_level": "high"
}

---

## Architecture

Raw Data → Feature Engineering → sklearn Pipeline
→ MLflow Experiment Tracking
→ FastAPI REST Endpoint
→ Evidently Drift Monitoring
→ Prefect Retraining Pipeline

---

## Key Design Decisions

- Used sklearn Pipeline to prevent data leakage between train/test
- Used ROC-AUC over accuracy because of 73/27 class imbalance
- Used SMOTE inside the pipeline — applied only to training data
- Used MLflow Model Registry with aliases instead of deprecated stages
- Used Evidently DataDriftPreset to monitor 5 key features
- Drift threshold set at 3/5 features — triggers retraining alert

---

## Tech Stack

- Data: pandas, numpy
- ML: scikit-learn, XGBoost, LightGBM, imbalanced-learn
- Tracking: MLflow 3.x
- Serving: FastAPI, Uvicorn, Pydantic
- Monitoring: Evidently AI
- Orchestration: Prefect
- Testing: pytest
