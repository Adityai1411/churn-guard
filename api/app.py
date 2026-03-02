"""
ChurnGuard API - FastAPI Prediction Service

Provides REST endpoints for customer churn prediction with explainability.
Part of the ChurnGuard ML Pipeline system.

Endpoints:
    POST /predict - Predict churn probability for a customer
    GET  /health  - Health check endpoint

Author: Aditya
Version: 1.0.0
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from typing import List, Tuple

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.features import FeatureEngineer
from src.evaluation import ExplainabilityAnalyzer

app = FastAPI(
    title="ChurnGuard API",
    description="Customer Churn Prediction Service with Explainability",
    version="1.0.0"
)

model = None
explainer = None


def load_model_artifacts():
    global model, explainer
    model_path = os.path.join(project_root, 'models', 'xgb_model.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    explainer = ExplainabilityAnalyzer(model)


try:
    load_model_artifacts()
except Exception as e:
    print(f"Warning: Model not loaded - {str(e)}")


class CustomerData(BaseModel):
    customer_id: int
    tenure_months: int
    monthly_charges: float
    contract_type: str
    support_tickets: int
    total_charges: float = 0.0
    payment_method: str = "Credit card"
    tech_support: str = "No"
    internet_service: str = "DSL"


class PredictionResponse(BaseModel):
    customer_id: int
    churn_probability: float
    confidence: float
    top_risk_factors: List[Tuple[str, float]]
    recommendation: str
    timestamp: str


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    df = pd.DataFrame([customer.dict()])
    
    if df['total_charges'].iloc[0] == 0.0:
        df['total_charges'] = df['monthly_charges'] * df['tenure_months']
    
    engineer = FeatureEngineer()
    df = engineer.engineer_features(df)
    
    exclude_cols = ['customer_id', 'customer_segment', 'churn']
    X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
    X = X.select_dtypes(include=[np.number])
    
    if hasattr(model, 'feature_names_in_'):
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)
    
    churn_prob = model.predict_proba(X)[0][1]
    confidence = float(np.max(model.predict_proba(X)[0]))
    
    explanation = explainer.explain_prediction(X, churn_prob)
    recommendation = _get_recommendation(churn_prob)
    
    return PredictionResponse(
        customer_id=customer.customer_id,
        churn_probability=round(float(churn_prob), 4),
        confidence=round(confidence, 4),
        top_risk_factors=explanation['top_risk_factors'],
        recommendation=recommendation,
        timestamp=datetime.now().isoformat()
    )


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "xgb_churn_v1",
        "version": "1.0.0"
    }


def _get_recommendation(churn_probability: float) -> str:
    if churn_probability > 0.7:
        return "HIGH RISK - Send retention offer immediately"
    elif churn_probability > 0.5:
        return "MEDIUM RISK - Monitor closely"
    else:
        return "LOW RISK - Standard engagement"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)