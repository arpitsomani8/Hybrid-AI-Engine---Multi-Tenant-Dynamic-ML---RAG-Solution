# Hybrid AI Engine: Multi-Tenant Dynamic ML & RAG Solution

A production-grade, zero-to-one multi-tenant enterprise intelligence platform featuring **dynamic runtime schema validation**, **automated tabular ML training (LightGBM + Optuna)**, and **intelligent cold-start fallback to vector RAG (Pinecone Serverless + LLM synthesis)**.

---

## 🚀 Key Highlights & Architecture

- **Multi-Tenant Isolation**: Complete isolation across REST API requests (`X-Tenant-ID`), dynamic schema definitions, model artifact registries, and Pinecone vector namespaces (`namespace=tenant_id`).
- **Pinecone Serverless Free Tier Integration**: Connects natively to live Pinecone Serverless indexes (`cloud="aws"`, `region="us-east-1"`). Employs namespace partitioning to guarantee zero cross-tenant vector leakage within a single free-tier index.
- **Dynamic Schema Engine**: Runtime Pydantic v2 model generation from tenant JSON specifications, enforcing type coercion, numerical bounds, categorical values, and detecting feature drift.
- **Automated Tabular ML (LightGBM + Optuna)**: Automatically evaluates whether a tenant has sufficient historical samples ($\ge 100$ records). Performs automated imputation, categorical encoding, multi-trial hyperparameter tuning, and early stopping.
- **Intelligent Fallback Router**:
  - **Cold-Start Tenants ($< 100$ samples)**: Synthesizes structured attributes into a semantic prompt, queries the tenant's Pinecone namespace, and returns grounded policy guidance.
  - **Trained Tenants ($\ge 100$ samples)**: Validates against the dynamic schema and executes sub-millisecond LightGBM inference.
  - **Unstructured Natural Language Queries**: Automatically routes to Pinecone vector search + LLM synthesis regardless of model status.
- **Interactive Enterprise Control Center**: Sleek glassmorphic dark-theme UI with real-time routing pipeline visualization, Pinecone namespace metrics, and sandbox simulators.

---

## 🏛️ Routing Decision Matrix

| Query Format | Tabular History | Active Model | Executed Engine | Output |
|---|---|---|---|---|
| **Structured JSON** | $< 100$ samples | None (Cold-Start) | **Pinecone Serverless RAG + LLM** | Calibrated institutional policy guidance |
| **Structured JSON** | $\ge 100$ samples | Trained LightGBM | **Classical Tabular ML** | Fast probability score, label, confidence, drift report |
| **Natural Language** | Any | Any | **Pinecone Serverless RAG + LLM** | Grounded answer with source citations |

---

## 🛠️ Tech Stack

- **Python 3.11+ / 3.14**
- **FastAPI & Uvicorn** (REST API & Lifespan Architecture)
- **Pydantic v2** (Dynamic Model Generation & Validation)
- **LightGBM & Scikit-Learn** (Classical Tabular ML)
- **Optuna** (Hyperparameter Optimization)
- **Pinecone Client v10** (Serverless Free-Tier Vector DB with Namespaces)
- **Docker & Docker Compose** (Containerization)

---

## 📦 Quickstart Guide

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/arpitsomani8/Hybrid-AI-Engine---Multi-Tenant-Dynamic-ML---RAG-Solution.git
cd "Hybrid AI Engine"

# Optional: Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
Copy the template and verify your keys:
```bash
cp .env.example .env
```
Ensure your Pinecone API key is set in `.env`:
```env
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=hybrid-ai-engine
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

### 3. Seed Realistic Multi-Tenant Data
Populates sample schemas, Pinecone knowledge policies, and tabular datasets for `fintech_corp` (ready to train) and `ecommerce_inc` (cold-start):
```bash
python scripts/seed_demo_data.py
```

### 4. Run the Test Suite
```bash
python -m pytest tests/
```

### 5. Launch the Platform
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser to:
- **Interactive Dashboard UI**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Endpoint**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 📡 REST API Reference

All requests must provide the tenant partition header: `X-Tenant-ID: <tenant_id>`.

### 1. Hybrid Inference (`POST /api/v1/predict`)
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/predict" \
  -H "X-Tenant-ID: fintech_corp" \
  -H "Content-Type: application/json" \
  -d '{
    "payload": {
      "account_age_days": 15,
      "transaction_velocity_24h": 7200.0,
      "failed_login_attempts": 1,
      "risk_score_tier": "high"
    }
  }'
```

### 2. Register Dynamic Schema (`POST /api/v1/schemas`)
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/schemas" \
  -H "X-Tenant-ID: fintech_corp" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Fintech Risk Schema",
    "target_column": "is_fraud",
    "task_type": "binary",
    "fields": {
      "account_age_days": {"type": "int", "ge": 0},
      "transaction_velocity_24h": {"type": "float", "ge": 0.0},
      "risk_score_tier": {"type": "str", "allowed": ["low", "medium", "high"]}
    }
  }'
```

### 3. Trigger Automated LightGBM Training (`POST /api/v1/models/train`)
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/models/train" \
  -H "X-Tenant-ID: fintech_corp" \
  -H "Content-Type: application/json" \
  -d '{
    "target_column": "is_fraud",
    "task_type": "binary",
    "tune_hyperparameters": true
  }'
```

### 4. Index Document into Pinecone (`POST /api/v1/rag/upsert`)
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/rag/upsert" \
  -H "X-Tenant-ID: fintech_corp" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "rule_velocity_limit",
    "text": "Transactions exceeding $5000 in 24 hours require manager authorization.",
    "metadata": {"category": "compliance"}
  }'
```

---

## 🐳 Docker Deployment

```bash
docker compose up --build -d
```
The application will be running at `http://localhost:8000`.

---

## 📄 License
MIT License. Built for enterprise multi-tenant intelligence systems.
