import pytest
import numpy as np
import pandas as pd
from app.core import HybridAIEngine

def test_hybrid_engine_end_to_end_flow(tmp_path):
    engine = HybridAIEngine(model_dir=tmp_path / "models", schema_dir=tmp_path / "schemas")

    # 1. Register dynamic schema for tenant 'alpha_corp'
    schema_def = {
        "description": "Alpha Corp Credit Evaluation",
        "fields": {
            "credit_score": {"type": "int", "ge": 300, "le": 850},
            "debt_to_income": {"type": "float", "ge": 0.0},
            "employment_status": {"type": "str", "allowed": ["employed", "self_employed", "unemployed"]}
        }
    }
    engine.register_schema("alpha_corp", schema_def)

    # 2. Index policy document into alpha_corp namespace
    engine.index_document(
        tenant_id="alpha_corp",
        doc_id="rule_credit_01",
        text="Applicants with debt-to-income over 0.45 or credit score under 580 require senior approval.",
        metadata={"category": "underwriting"}
    )

    # 3. Query in Cold Start (No trained model yet)
    cold_payload = {
        "credit_score": 520,
        "debt_to_income": 0.48,
        "employment_status": "employed"
    }
    cold_res = engine.process_query("alpha_corp", cold_payload)
    assert cold_res["route"] == "cold_start_rag"
    assert "senior approval" in cold_res["answer"].lower()

    # 4. Train Model with >= 100 rows
    np.random.seed(42)
    N = 120
    df = pd.DataFrame({
        "credit_score": np.random.randint(350, 800, N),
        "debt_to_income": np.random.uniform(0.1, 0.6, N),
        "employment_status": np.random.choice(["employed", "self_employed", "unemployed"], N)
    })
    df["approved"] = ((df["credit_score"] > 600) & (df["debt_to_income"] < 0.4)).astype(int)

    engine.train_model("alpha_corp", df, target_col="approved", task="binary", tune_hyperparameters=False)

    # 5. Query Post-Training with structured payload -> Routes to Classical ML
    ml_res = engine.process_query("alpha_corp", cold_payload)
    assert ml_res["route"] == "classical_ml"
    assert "prediction" in ml_res
    assert "confidence" in ml_res

    # 6. Query with natural language text -> Routes to Unstructured RAG
    nl_res = engine.process_query("alpha_corp", {"natural_language_query": "What is the policy for debt-to-income ratio?"})
    assert nl_res["route"] == "unstructured_rag"
    assert "answer" in nl_res
