import requests
import json

BASE_URL = "http://localhost:8000"

def test_api():
    print("--- 1. Testing Health Endpoint ---")
    r = requests.get(f"{BASE_URL}/health")
    print("Health:", r.json())
    assert r.status_code == 200

    print("\n--- 2. Testing Tenant Status (fintech_corp) ---")
    r = requests.get(f"{BASE_URL}/api/v1/tenants/fintech_corp/status")
    print("Fintech Corp Status:", r.json())
    assert r.status_code == 200

    print("\n--- 3. Triggering LightGBM Training (fintech_corp) ---")
    train_payload = {
        "target_column": "is_fraud",
        "task_type": "binary",
        "tune_hyperparameters": True
    }
    r = requests.post(f"{BASE_URL}/api/v1/models/train", json=train_payload, headers={"X-Tenant-ID": "fintech_corp"})
    print("Train Result:", r.json())
    assert r.status_code == 200

    print("\n--- 4. Testing Classical ML Route (fintech_corp) ---")
    pred_payload = {
        "payload": {
            "account_age_days": 15,
            "transaction_velocity_24h": 7200.0,
            "failed_login_attempts": 2,
            "risk_score_tier": "high"
        }
    }
    r = requests.post(f"{BASE_URL}/api/v1/predict", json=pred_payload, headers={"X-Tenant-ID": "fintech_corp"})
    print("Predict (Trained):", r.json())
    assert r.status_code == 200
    assert r.json()["data"]["route"] == "classical_ml"

    print("\n--- 5. Testing Cold-Start RAG Route (ecommerce_inc) ---")
    ecom_payload = {
        "payload": {
            "order_value": 450.0,
            "customer_tenure_days": 5,
            "shipping_country": "US"
        }
    }
    r = requests.post(f"{BASE_URL}/api/v1/predict", json=ecom_payload, headers={"X-Tenant-ID": "ecommerce_inc"})
    print("Predict (Cold-Start):", r.json())
    assert r.status_code == 200
    assert r.json()["data"]["route"] == "cold_start_rag"

    print("\n--- 6. Testing Unstructured Natural Language Query Route ---")
    nl_payload = {
        "payload": {
            "natural_language_query": "What is the manager authorization rule for high velocity transactions?"
        }
    }
    r = requests.post(f"{BASE_URL}/api/v1/predict", json=nl_payload, headers={"X-Tenant-ID": "fintech_corp"})
    print("Predict (NL Query):", r.json())
    assert r.status_code == 200
    assert r.json()["data"]["route"] == "unstructured_rag"

    print("\n[OK] ALL LIVE API TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_api()
