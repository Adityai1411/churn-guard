"""
ChurnGuard MLflow Tracking Module

Logs model experiments, metrics, and artifacts to MLflow for tracking and comparison.

Author: Aditya
Version: 1.0.0
"""

import mlflow
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


class MLflowTracker:
    """MLflow experiment tracking for model training."""
    
    def __init__(self, experiment_name):
        mlflow.set_experiment(experiment_name)

    def log_experiment(self, model_name, model, X_test, y_test, params):
        """Log model training experiment to MLflow."""
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(params)
            
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
            
            mlflow.log_metrics({
                'auc': roc_auc_score(y_test, y_pred_proba),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            })
            
            mlflow.xgboost.log_model(model, "model")
            print(f"Logged {model_name} to MLflow")


if __name__ == "__main__":
    import joblib
    from sklearn.model_selection import train_test_split
    from src.features import FeatureEngineer
    import pandas as pd
    
    df = pd.read_csv('data/raw/churn.csv')
    engineer = FeatureEngineer()
    df_feat = engineer.engineer_features(df)
    
    X = df_feat.drop(columns=['churn', 'customer_id', 'customer_segment'])
    X = X.select_dtypes(include=['float64', 'int64'])
    y = df_feat['churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = joblib.load('models/xgb_model.pkl')
    
    tracker = MLflowTracker("churn-prediction")
    tracker.log_experiment("xgboost_v1", model, X_test, y_test, {'model': 'xgboost', 'version': '1.0'})