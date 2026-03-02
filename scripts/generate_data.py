"""
ChurnGuard Data Generator

Generates synthetic customer churn data with realistic patterns and correlations.
Creates 10,000 customer records with 27% baseline churn rate.

Author: Aditya
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import os


def generate_synthetic_data(n_customers=10000, seed=42):
    """Generate synthetic customer churn dataset."""
    np.random.seed(seed)
    
    data = pd.DataFrame({
        "customer_id": range(1, n_customers + 1),
        "tenure_months": np.random.randint(1, 72, n_customers),
        "monthly_charges": np.random.uniform(20, 120, n_customers),
        "contract_type": np.random.choice(
            ["Month-to-month", "1 year", "2 year"], n_customers
        ),
        "support_tickets": np.random.randint(0, 10, n_customers),
        "tech_support": np.random.choice(["Yes", "No"], n_customers),
        "internet_service": np.random.choice(
            ["Fiber optic", "DSL", "No"], n_customers
        ),
        "payment_method": np.random.choice(
            ["E-check", "Mailed check", "Bank transfer", "Credit card"], n_customers
        ),
    })
    
    data["total_charges"] = data["monthly_charges"] * data["tenure_months"]
    
    risk_score = (
        1.2 * (data["contract_type"] == "Month-to-month").astype(int) +
        1.0 * (data["support_tickets"] > 6).astype(int) +
        0.8 * (data["tenure_months"] < 4).astype(int) +
        0.6 * (data["monthly_charges"] > 95).astype(int) +
        0.5 * (
            (data["contract_type"] == "Month-to-month") &
            (data["support_tickets"] > 5)
        ).astype(int)
    )
    
    logit = -2 + risk_score
    churn_prob = 1 / (1 + np.exp(-logit))
    
    data["churn"] = np.random.binomial(1, churn_prob)
    
    os.makedirs("data/raw", exist_ok=True)
    data.to_csv("data/raw/churn.csv", index=False)
    
    churn_rate = data["churn"].mean()
    print(f"Generated {n_customers} customer records")
    print(f"Churn rate: {churn_rate:.2%}")
    print(f"Saved to: data/raw/churn.csv")


if __name__ == "__main__":
    generate_synthetic_data()