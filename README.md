# ChurnGuard - ML Churn Prediction System

## Problem Statement
SaaS companies lose 25% of customers monthly. We built a system that predicts churn, explains why, and identifies at-risk customers.

## Quick Start
1. `pip install -r requirements.txt`
2. `python scripts/generate_data.py`
3. `python src/models.py`
4. `streamlit run app/dashboard.py`
5. `uvicorn api.app:app --reload`

## Live Demo
[Link to Hugging Face Spaces]