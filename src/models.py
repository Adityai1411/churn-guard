"""
ChurnGuard Model Training Pipeline

Trains XGBoost classifier for customer churn prediction.
Saves model and feature schema for deployment.

Author: Aditya
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import os
import sys
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.features import FeatureEngineer


class ChurnModelPipeline:

    def __init__(self):
        self.model = None
        self.feature_schema = None

    def train(self):
        data_path = os.path.join(project_root, "data", "raw", "churn.csv")
        model_path = os.path.join(project_root, "models", "xgb_model.pkl")
        schema_path = os.path.join(project_root, "models", "feature_schema.pkl")

        if not os.path.exists(data_path):
            print(f"Error: Data file not found at {data_path}")
            print("Run: python -m scripts.generate_data")
            return

        print("Loading data...")
        df = pd.read_csv(data_path)

        print("Engineering features...")
        engineer = FeatureEngineer()
        df = engineer.engineer_features(df)

        X = df.drop(columns=["churn", "customer_id", "customer_segment"])
        y = df["churn"]

        X = X.select_dtypes(include=[np.number])

        self.feature_schema = X.columns.tolist()

        print(f"Training with {X.shape[0]} samples and {X.shape[1]} features")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("Training XGBoost model...")
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
            random_state=42,
            n_jobs=-1,
            eval_metric="auc"
        )

        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        print("\n" + "="*50)
        print("MODEL PERFORMANCE")
        print("="*50)
        print(f"AUC:       {roc_auc_score(y_test, y_proba):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred):.4f}")
        print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
        print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
        print("="*50)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, model_path)
        joblib.dump(self.feature_schema, schema_path)

        print(f"\nModel saved to: {model_path}")
        print(f"Feature schema saved to: {schema_path}")
        print(f"Total features: {len(self.feature_schema)}")

        self.model = model


if __name__ == "__main__":
    pipeline = ChurnModelPipeline()
    pipeline.train()