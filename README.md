<div align="center">

# 🛡️ ChurnGuard

### End-to-End Explainable AI Retention Engine

**Predict customer churn 30 days in advance • Explain why • Take action**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[🚀 Live Demo](https://huggingface.co/spaces/adityaingale/churn-guard) • [📖 Documentation](#-documentation) • [🐛 Report Bug](https://github.com/Adityai1411/churn-guard/issues) • [✨ Request Feature](https://github.com/Adityai1411/churn-guard/issues)

</div>

---

## 📋 Table of Contents

- [About](#-about)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Model Performance](#-model-performance)
- [Business Impact](#-business-impact)
- [Deployment](#-deployment)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

---

## 🎯 About

### The Problem
SaaS companies lose approximately **25% of customers annually**, translating to millions in revenue leakage. Traditional churn prediction models operate as **black boxes**—they identify *who* is likely to churn but fail to explain *why*. Without actionable insights, retention teams cannot design targeted interventions, resulting in wasted budget and missed opportunities.

### The Solution
**ChurnGuard** is a production-grade machine learning system that bridges this gap. It combines high-accuracy predictive modeling with **Explainable AI (XAI)** to deliver not just predictions, but *reasons*. By integrating SHAP (SHapley Additive exPlanations), the system translates complex model outputs into business-friendly insights such as:

> *"Customer X has a 78% churn risk, primarily driven by a Month-to-Month contract (+0.32) and high monthly charges (+0.25)."*

This enables retention teams to design **targeted, data-driven interventions**—offering the right incentive to the right customer at the right time.

---

## ✨ Key Features

### 🔮 Predictive Intelligence
- **XGBoost Classifier** achieving **0.82 AUC-ROC** on imbalanced churn data
- Native handling of class imbalance via `scale_pos_weight` parameter
- Processes 10,000+ customer records with 30+ engineered features
- Bayesian hyperparameter optimization using **Optuna**

### 🧠 Explainable AI (XAI)
- **SHAP integration** for both global and local feature attribution
- Business-friendly explanations for every prediction
- Interactive force plots and summary visualizations
- Builds stakeholder trust through transparent decision-making

### ⚡ Production-Ready API
- **FastAPI** microservice with automatic OpenAPI documentation
- **Pydantic** validation for robust request handling
- Sub-200ms inference latency via Singleton pattern optimization
- Health check endpoint for monitoring and load balancing

### 📱 Interactive Dashboard
- **Streamlit** interface for non-technical stakeholders
- Dynamic risk threshold adjustment
- Top 100 at-risk customers table with downloadable CSV
- Individual customer deep-dives with SHAP explanations

### 🔄 Batch Processing Pipeline
- Daily scoring of entire customer base
- Automated export of high-risk segments
- Scalable architecture for enterprise datasets

### 🐳 MLOps & Deployment
- **Docker** containerization with multi-stage builds
- **MLflow** experiment tracking and model registry
- CI/CD deployment via Hugging Face Spaces
- Environment parity guaranteed across dev/prod

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CLIENT LAYER                          │
│  ┌──────────────────┐         ┌──────────────────┐     │
│  │  Streamlit UI    │         │  API Clients     │     │
│  │  (Dashboard)     │         │  (curl/requests) │     │
│  └────────┬─────────┘         └────────┬─────────┘     │
└───────────┼─────────────────────────────┼───────────────┘
            │                             │
            └──────────┬──────────────────┘
                       │ HTTP/JSON
                       ▼
┌─────────────────────────────────────────────────────────┐
│              FASTAPI BACKEND (api/app.py)                │
│  • Pydantic validation                                   │
│  • Feature engineering (src/features.py)                 │
│  • Model inference (XGBoost)                             │
│  • SHAP explanation generation                           │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              ML CORE (src/)                              │
│  • models.py: Training, evaluation, Optuna tuning        │
│  • evaluation.py: SHAP integration                       │
│  • features.py: Feature engineering                      │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              PERSISTENCE LAYER                           │
│  • models/: Serialized artifacts (Joblib)                │
│  • mlruns/: MLflow experiment tracking                   │
│  • .env: Configuration (python-dotenv)                   │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| **Core Language** | Python 3.10+ | Primary programming language |
| **ML Framework** | XGBoost 2.0+ | Gradient boosting classifier |
| **ML Utilities** | Scikit-Learn 1.3+ | Preprocessing, metrics, utilities |
| **Explainability** | SHAP 0.42+ | Model interpretability via game theory |
| **Hyperparameter Tuning** | Optuna 3.4+ | Bayesian optimization |
| **Experiment Tracking** | MLflow 2.8+ | Reproducibility and model registry |
| **Data Processing** | Pandas 2.0+, NumPy 1.24+ | Feature engineering and manipulation |
| **Backend API** | FastAPI 0.104+ | High-performance REST framework |
| **Data Validation** | Pydantic 2.4+ | Request/response schema validation |
| **ASGI Server** | Uvicorn 0.24+ | Production server for FastAPI |
| **Frontend** | Streamlit 1.29+ | Rapid dashboard development |
| **Visualization** | Plotly 5.17+ | Interactive charts and graphs |
| **Serialization** | Joblib 1.3+ | Efficient model persistence |
| **Configuration** | python-dotenv 1.0+ | Environment variable management |
| **Containerization** | Docker | Environment parity and deployment |
| **Deployment** | Hugging Face Spaces | Free hosting with CI/CD |

---

## 📁 Project Structure

```
churn-guard/
├── 📁 src/                       # Core ML pipeline
│   ├── features.py               # Feature engineering (30+ features)
│   ├── models.py                 # XGBoost training & evaluation
│   └── evaluation.py             # SHAP explainability module
│
├── 📁 api/                       # FastAPI backend
│   └── app.py                    # REST endpoints & Pydantic schemas
│
├── 📁 app/                       # Streamlit frontend
│   └── dashboard.py              # Interactive UI
│
├── 📁 batch/                     # Batch processing
│   └── score_all_customers.py    # Daily scoring pipeline
│
├── 📁 scripts/                   # Utilities
│   └── generate_data.py          # Synthetic data generation
│
├── 📁 models/                    # Serialized model artifacts
│   ├── churn_model_v1.pkl        # Trained XGBoost model
│   ├── scaler_v1.pkl             # Feature scaler
│   └── feature_names.pkl         # Feature name registry
│
├── 📁 tests/                     # Unit & integration tests
│   ├── test_features.py          # Feature engineering tests
│   └── test_api.py               # API endpoint tests
│
├── 📁 mlruns/                    # MLflow experiment logs
├── 📁 data/                      # Dataset directory
├── 📄 Dockerfile                 # Container configuration
├── 📄 requirements.txt           # Python dependencies
├── 📄 render.yaml                # Render deployment config
├── 📄 .env.example               # Environment template
├── 📄 .gitignore                 # Git ignore rules
└── 📄 README.md                  # This file
```

---

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- Git
- Docker (optional, for containerized deployment)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Adityai1411/churn-guard.git
cd churn-guard
```

### Step 2: Create Virtual Environment
```bash
# Linux/macOS
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Environment
```bash
cp .env.example .env
# Edit .env with your configuration
```

---

## 📖 Usage

### 1. Generate Sample Data
```bash
python -m scripts.generate_data --n-customers 10000 --output data/customers.csv
```

### 2. Train the Model
```bash
python -m src.models --data-path data/customers.csv --output models/
```

This will:
- Load and preprocess the dataset
- Engineer 30+ features
- Split data with stratification
- Tune hyperparameters via Optuna
- Train XGBoost with class imbalance handling
- Log experiments to MLflow
- Save model artifacts to `models/`

### 3. Launch the API Server
```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

API will be available at: `http://localhost:8000`  
Interactive docs at: `http://localhost:8000/docs`

### 4. Launch the Dashboard
```bash
streamlit run app/dashboard.py --server.port 8501
```

Dashboard will be available at: `http://localhost:8501`

### 5. Run Batch Scoring
```bash
python -m batch.score_all_customers --input data/customers.csv --output results/predictions.csv
```

---

## 📡 API Documentation

### Endpoints

#### `POST /predict`
Predict churn risk for a single customer.

**Request Body:**
```json
{
  "customer_id": "CUST-12345",
  "tenure_months": 12,
  "monthly_charges": 79.99,
  "contract_type": "Month-to-month",
  "payment_method": "Electronic check",
  "total_charges": 959.88
}
```

**Response:**
```json
{
  "customer_id": "CUST-12345",
  "churn_probability": 0.78,
  "risk_level": "High",
  "top_risk_factors": [
    {
      "feature": "ContractType_Month-to-month",
      "value": 1,
      "impact": 0.32,
      "direction": "increases"
    },
    {
      "feature": "MonthlyCharges",
      "value": 79.99,
      "impact": 0.25,
      "direction": "increases"
    }
  ],
  "recommended_actions": [
    "Offer annual contract discount",
    "Schedule retention call within 48 hours"
  ],
  "model_version": "v1.2.0"
}
```

#### `GET /health`
Health check endpoint for monitoring.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "v1.2.0",
  "uptime_seconds": 3600
}
```

#### `GET /docs`
Interactive Swagger UI for API exploration.

---

## 📊 Model Performance

### Metrics on Test Set (20% holdout)

| Metric | Value | Business Relevance |
|--------|-------|--------------------|
| **AUC-ROC** | 0.82 | Strong ranking ability across all thresholds |
| **Accuracy** | 78.5% | Overall correctness (misleading due to imbalance) |
| **Precision** | 0.74 | 74% of flagged customers actually churn |
| **Recall** | 0.71 | Catch 71% of actual churners |
| **F1-Score** | 0.72 | Balanced precision-recall harmonic mean |
| **Log Loss** | 0.48 | Well-calibrated probability estimates |

### Feature Importance (Top 5)
1. `ContractType_Month-to-month` — Short contracts strongly predict churn
2. `MonthlyCharges` — Higher charges correlate with higher churn
3. `TenureMonths` — Newer customers churn more frequently
4. `charge_to_tenure_ratio` — Rate of spending relative to loyalty
5. `PaymentMethod_Electronic check` — Certain payment methods show higher risk

---

## 💼 Business Impact

### ROI Calculation

**Assumptions:**
- Average customer Lifetime Value (LTV): $500
- Retention campaign cost per customer: $15
- Campaign success rate (when targeted correctly): 40%

**Monthly Impact (10,000 customer base):**
- ChurnGuard identifies **~2,700 high-risk customers**
- Targeted outreach cost: 2,700 × $15 = **$40,500**
- Retained LTV: 2,700 × 40% × $500 = **$540,000**
- **Net monthly value: ~$500,000**

Even with conservative estimates (10% success rate), the system delivers **$135K saved vs. $40K spent**, yielding a **3.4x ROI**.

### Key Business Benefits
- ✅ **30-day advance warning** enables proactive intervention
- ✅ **Explainable predictions** build stakeholder trust
- ✅ **Actionable insights** replace guesswork with data
- ✅ **Scalable architecture** supports enterprise growth

---

## 🐳 Deployment

### Docker Deployment
```bash
# Build the image
docker build -t churnguard:latest .

# Run the container
docker run -p 8000:8000 -p 8501:8501 churnguard:latest
```

### Hugging Face Spaces
The project is automatically deployed to Hugging Face Spaces via Git integration:
- **Live Demo:** [huggingface.co/spaces/adityaingale/churn-guard](https://huggingface.co/spaces/adityaingale/churn-guard)
- Push to `main` branch triggers automatic rebuild

### Environment Variables
```bash
MODEL_PATH=models/churn_model_v1.pkl
SCALER_PATH=models/scaler_v1.pkl
LOG_LEVEL=INFO
API_PORT=8000
DASHBOARD_PORT=8501
```

---

## 🗺️ Roadmap

### v1.1 (Current)
- [x] XGBoost classifier with SHAP explainability
- [x] FastAPI backend with Pydantic validation
- [x] Streamlit dashboard with Plotly visualizations
- [x] Docker containerization
- [x] Hugging Face Spaces deployment

### v1.2 (Planned)
- [ ] Feature Store integration (Feast) for online/offline consistency
- [ ] Data drift monitoring with Evidently AI
- [ ] Automated retraining pipeline
- [ ] A/B testing framework for retention campaigns
- [ ] Kubernetes deployment manifests

### v2.0 (Future)
- [ ] Causal inference for uplift modeling
- [ ] Real-time feature pipeline with Kafka/Flink
- [ ] Multi-tenant architecture
- [ ] Federated learning support for privacy-sensitive deployments

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure your code:
- Follows PEP 8 style guidelines
- Includes unit tests for new functionality
- Updates documentation as needed
- Passes all existing tests

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👨💻 Author

**Aditya Ingale**  
AI/ML Engineer

- GitHub: [@Adityai1411](https://github.com/Adityai1411)
- LinkedIn: [Aditya Ingale](https://www.linkedin.com/in/ingaleaditya1411/)
- Portfolio: [aditya-ingale-portfolio.vercel.app](https://aditya-ingale-portfolio.vercel.app)

---

## 🙏 Acknowledgments

- [XGBoost](https://xgboost.readthedocs.io/) team for the powerful gradient boosting library
- [SHAP](https://shap.readthedocs.io/) developers for game-theoretic explainability
- [FastAPI](https://fastapi.tiangolo.com/) creator Sebastián Ramírez for the excellent framework
- [Streamlit](https://streamlit.io/) team for making data apps accessible
- The open-source ML community for continuous inspiration

---

## 📞 Contact

For questions, collaborations, or feedback:
- 📧 Email: ingaleaditya1411@gmail.com
- 💼 LinkedIn: [Connect with me](https://www.linkedin.com/in/ingaleaditya1411/)
- 🐛 Issues: [Open an issue](https://github.com/Adityai1411/churn-guard/issues)

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star!**

Made with ❤️ by Aditya Ingale

</div>
