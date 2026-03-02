"""
ChurnGuard Batch Scoring Pipeline

Scores all customers daily and identifies high-risk accounts for retention team.

Author: Aditya
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.features import FeatureEngineer


def batch_score():
    """Score all customers and generate risk assessment reports."""
    
    data_path = os.path.join(project_root, "data", "raw", "churn.csv")
    model_path = os.path.join(project_root, "models", "xgb_model.pkl")
    schema_path = os.path.join(project_root, "models", "feature_schema.pkl")
    results_dir = os.path.join(project_root, "results")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Feature schema not found: {schema_path}")
    
    df = pd.read_csv(data_path)
    
    engineer = FeatureEngineer()
    df = engineer.engineer_features(df)
    
    model = joblib.load(model_path)
    schema = joblib.load(schema_path)
    
    exclude_cols = ["churn", "customer_id", "customer_segment"]
    X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
    X = X.select_dtypes(include=[np.number])
    X = X.reindex(columns=schema, fill_value=0)
    
    df["churn_prob"] = model.predict_proba(X)[:, 1]
    
    df["risk_level"] = pd.cut(
        df["churn_prob"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"]
    )
    
    os.makedirs(results_dir, exist_ok=True)
    
    df.to_csv(os.path.join(results_dir, "all_customers_scored.csv"), index=False)
    
    high_risk = df[df["risk_level"] == "High"].nlargest(100, "churn_prob")
    high_risk.to_csv(os.path.join(results_dir, "high_risk_customers.csv"), index=False)
    
    print(f"Batch scoring completed at {datetime.now().isoformat()}")
    print(f"Total customers scored: {len(df)}")
    print(f"High-risk customers identified: {len(high_risk)}")
    print(f"Results saved to: {results_dir}")
    
    return df, high_risk


if __name__ == "__main__":
    batch_score()