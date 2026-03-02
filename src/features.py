"""
ChurnGuard Feature Engineering Module

Creates 30+ features from raw customer data for churn prediction.
Includes customer value, contract risk, engagement, and interaction features.

Author: Aditya
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """Feature engineering pipeline for customer churn data."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None

    def engineer_features(self, df):
        """Create 30+ features from raw customer data."""
        df = df.copy()
        
        df['clv_estimate'] = df['monthly_charges'] * df['tenure_months']
        df['avg_monthly_charge'] = df['total_charges'] / (df['tenure_months'] + 1)
        
        df['contract_risk'] = (df['contract_type'] == 'Month-to-month').astype(int)
        df['month_to_month_penalty'] = df['contract_risk'] * df['monthly_charges']
        
        df['support_intensity'] = df['support_tickets'] / (df['tenure_months'] + 1)
        df['is_new_customer'] = (df['tenure_months'] < 6).astype(int)
        
        df['customer_segment'] = pd.cut(df['clv_estimate'], bins=3, labels=['Low', 'Medium', 'High'])
        df['segment_high'] = (df['customer_segment'] == 'High').astype(int)
        df['segment_medium'] = (df['customer_segment'] == 'Medium').astype(int)
        df['segment_low'] = (df['customer_segment'] == 'Low').astype(int)
        
        df['price_sensitive'] = (df['monthly_charges'] > df['monthly_charges'].median()).astype(int)
        df['spending_growth'] = df['monthly_charges'] - df['avg_monthly_charge']
        
        df['tech_adoption'] = (df['tech_support'] == 'Yes').astype(int)
        df['advanced_service'] = (df['internet_service'] == 'Fiber optic').astype(int)
        
        df['high_value_at_risk'] = df['segment_high'] * df['contract_risk']
        df['new_customer_risk'] = df['is_new_customer'] * df['monthly_charges']
        df['unsupported_risk'] = (1 - df['tech_adoption']) * df['contract_risk']
        
        df['clv_squared'] = df['clv_estimate'] ** 2
        df['tenure_log'] = np.log1p(df['tenure_months'])
        
        df = pd.get_dummies(df, columns=['contract_type', 'payment_method', 'internet_service'], drop_first=True)
        
        self.feature_columns = [col for col in df.columns if col not in ['churn', 'customer_id', 'customer_segment']]
        
        return df


if __name__ == "__main__":
    df = pd.read_csv('data/raw/churn.csv')
    engine = FeatureEngineer()
    df_feat = engine.engineer_features(df)
    print(f"Features created: {len(engine.feature_columns)}")
    df_feat.to_csv('data/processed/features.csv', index=False)
    print("Features saved to data/processed/features.csv")