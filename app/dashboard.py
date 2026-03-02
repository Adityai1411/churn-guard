"""
ChurnGuard Dashboard - Streamlit Application

Interactive dashboard for customer churn prediction and analysis.
Provides overview metrics, high-risk customer identification, and 
individual prediction capabilities.

Author: Aditya
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
import sys
import numpy as np

# Add project root to Python path for module imports
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))
sys.path.insert(0, project_root)

from src.features import FeatureEngineer

# Page configuration
st.set_page_config(page_title="ChurnGuard Dashboard", layout="wide")
st.title("ChurnGuard - Churn Prediction Dashboard")


@st.cache_resource
def load_model():
    """Load trained XGBoost model from disk."""
    model_path = os.path.join(project_root, 'models', 'xgb_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


model = load_model()


# Load batch scoring results
results_df = None
results_path = os.path.join(project_root, 'results', 'all_customers_scored.csv')

if os.path.exists(results_path):
    results_df = pd.read_csv(results_path)
    
    # Create risk_level column if missing from previous batch runs
    if 'risk_level' not in results_df.columns:
        results_df['risk_level'] = pd.cut(
            results_df['churn_prob'], 
            bins=[0, 0.3, 0.6, 1.0], 
            labels=['Low', 'Medium', 'High']
        )
        results_df.to_csv(results_path, index=False)
else:
    # Fallback mock data for development
    np.random.seed(42)
    results_df = pd.DataFrame({
        'customer_id': range(1, 101),
        'churn_prob': np.random.uniform(0, 1, 100),
        'risk_level': np.random.choice(['Low', 'Medium', 'High'], 100)
    })
    st.warning("Results file not found. Run batch scoring first.")


# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["Overview", "High Risk Customers", "Predict Individual"])


with tab1:
    st.header("Overview")
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(results_df))
    col2.metric("Avg Churn Probability", f"{results_df['churn_prob'].mean():.2%}")
    
    if 'risk_level' in results_df.columns:
        col3.metric("High Risk Count", len(results_df[results_df['risk_level'] == 'High']))
    else:
        col3.metric("High Risk Count", "N/A")
    
    # Churn probability distribution
    fig = px.histogram(
        results_df, 
        x='churn_prob', 
        title='Churn Probability Distribution', 
        nbins=50
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk level breakdown
    if 'risk_level' in results_df.columns:
        risk_counts = results_df['risk_level'].value_counts()
        fig2 = px.pie(
            values=risk_counts.values, 
            names=risk_counts.index,
            title='Customer Distribution by Risk Level'
        )
        st.plotly_chart(fig2, use_container_width=True)


with tab2:
    st.header("High Risk Customers")
    
    if 'risk_level' in results_df.columns:
        high_risk = results_df[results_df['risk_level'] == 'High'].nlargest(50, 'churn_prob')
        st.dataframe(
            high_risk[['customer_id', 'churn_prob', 'monthly_charges', 'tenure_months', 'risk_level']]
        )
        
        # Export functionality
        csv = high_risk.to_csv(index=False)
        st.download_button(
            label="Download High Risk Customers",
            data=csv,
            file_name="high_risk.csv",
            mime="text/csv"
        )
    else:
        st.warning("Risk level data not available.")


with tab3:
    st.header("Predict Individual Customer")
    
    if model is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=20, max_value=150, value=50)
            contract = st.selectbox("Contract Type", ['Month-to-month', '1 year', '2 year'])
        
        with col2:
            support = st.number_input("Support Tickets", min_value=0, max_value=10, value=0)
            tech_support = st.selectbox("Has Tech Support?", ['Yes', 'No'])
        
        if st.button("Predict"):
            # Prepare customer data
            customer_data = pd.DataFrame([{
                'tenure_months': tenure,
                'monthly_charges': monthly_charges,
                'contract_type': contract,
                'support_tickets': support,
                'tech_support': tech_support,
                'total_charges': monthly_charges * tenure,
                'payment_method': 'Credit card',
                'internet_service': 'DSL'
            }])
            
            # Feature engineering
            engineer = FeatureEngineer()
            customer_data = engineer.engineer_features(customer_data)
            
            # Prepare features for prediction
            exclude_cols = ['customer_id', 'customer_segment', 'churn']
            X = customer_data.drop(columns=[col for col in exclude_cols if col in customer_data.columns])
            X = X.select_dtypes(include=[np.number])
            
            # Align columns with training data
            if hasattr(model, 'feature_names_in_'):
                X = X.reindex(columns=model.feature_names_in_, fill_value=0)
            
            # Generate prediction
            prob = model.predict_proba(X)[0][1]
            st.metric("Churn Probability", f"{prob:.2%}")
            
            # Business recommendation based on risk threshold
            if prob > 0.7:
                st.error("HIGH RISK - Send retention offer immediately")
            elif prob > 0.5:
                st.warning("MEDIUM RISK - Monitor closely")
            else:
                st.success("LOW RISK - Standard engagement")
    else:
        st.error("Model not found. Please train the model first.")
        st.write(f"Expected location: {os.path.join(project_root, 'models', 'xgb_model.pkl')}")
        st.code("python -m src.models", language="bash")