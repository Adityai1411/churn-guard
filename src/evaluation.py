"""
ChurnGuard Explainability Module

SHAP-based model interpretability for churn predictions.

Author: Aditya
Version: 1.0.0
"""

import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os


class ExplainabilityAnalyzer:
    """SHAP explainability wrapper for tree-based models."""
    
    def __init__(self, model):
        self.model = model
        self.explainer = shap.TreeExplainer(model)

    def explain_prediction(self, customer_features, prediction_prob):
        """Generate explanation for a single customer prediction."""
        shap_values = self.explainer.shap_values(customer_features)
        
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]

        return {
            'churn_probability': prediction_prob,
            'top_risk_factors': self._get_top_factors(shap_values, customer_features),
            'top_protective_factors': self._get_protective_factors(shap_values, customer_features)
        }

    def _get_top_factors(self, shap_values, features):
        """Identify features that increase churn risk."""
        feature_importance = np.abs(shap_values).mean(axis=0)
        indices = np.argsort(feature_importance)[-3:]
        return [(features.columns[i], float(shap_values[0, i])) for i in indices]

    def _get_protective_factors(self, shap_values, features):
        """Identify features that decrease churn risk."""
        feature_importance = np.abs(shap_values).mean(axis=0)
        indices = np.argsort(feature_importance)[:3]
        return [(features.columns[i], float(shap_values[0, i])) for i in indices]

    def plot_feature_importance(self, X_test, save_path='reports'):
        """Generate and save SHAP feature importance plot."""
        os.makedirs(save_path, exist_ok=True)
        shap_values = self.explainer.shap_values(X_test)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.savefig(f'{save_path}/shap_importance.png', dpi=100, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    import joblib
    from src.features import FeatureEngineer
    import pandas as pd
    
    model = joblib.load('models/xgb_model.pkl')
    df = pd.read_csv('data/raw/churn.csv')
    
    engineer = FeatureEngineer()
    df_feat = engineer.engineer_features(df)
    
    X = df_feat.drop(columns=['churn', 'customer_id', 'customer_segment'])
    X = X.select_dtypes(include=[np.number])
    
    analyzer = ExplainabilityAnalyzer(model)
    analyzer.plot_feature_importance(X)
    print("SHAP feature importance plot saved to reports/shap_importance.png")