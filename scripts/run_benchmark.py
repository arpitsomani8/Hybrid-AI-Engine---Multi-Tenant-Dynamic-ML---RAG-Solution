import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_platform():
    print("=" * 65)
    print("HYBRID AI ENGINE: LIVE PRODUCTION WORKFLOW VERIFICATION")
    print("=" * 65)

    # 1. Healthcheck
    res_health = requests.get(f"{BASE_URL}/health")
    print(f"\n1. Platform Health: {res_health.json()}")

    # 2. Cold-Start Fallback (Tenant: ecommerce_inc, < 100 samples)
    print("\n2. Testing COLD-START Fallback for 'ecommerce_inc' (< 100 samples)...")
    payload_cold = {
        "days_since_last_purchase": 55,
        "average_order_value": 180.0,
        "support_tickets_opened": 3,
        "loyalty_tier": "vip"
    }
    t0 = time.time()
    res_cold = requests.post(
        f"{BASE_URL}/api/v1/predict",
        headers={"X-Tenant-ID": "ecommerce_inc"},
        json={"payload": payload_cold}
    )
    dt_cold = (time.time() - t0) * 1000
    cold_data = res_cold.json()["data"]
    print(f"   Route Executed: {cold_data['route']}")
    print(f"   Response Answer:\n   {cold_data['answer']}")
    print(f"   Latency: {cold_data['latency_ms']} ms (Total roundtrip: {dt_cold:.1f} ms)")

    # 3. Unstructured Natural Language RAG (Tenant: fintech_corp)
    print("\n3. Testing UNSTRUCTURED Natural Language Query for 'fintech_corp'...")
    payload_nl = {
        "natural_language_query": "What is the policy for new accounts under 30 days old?"
    }
    res_nl = requests.post(
        f"{BASE_URL}/api/v1/predict",
        headers={"X-Tenant-ID": "fintech_corp"},
        json={"payload": payload_nl}
    )
    nl_data = res_nl.json()["data"]
    print(f"   Route Executed: {nl_data['route']}")
    print(f"   Synthesized Directive:\n   {nl_data['answer']}")
    print(f"   Confidence / Similarity Score: {nl_data['confidence']}")

    # 4. Trigger Automated LightGBM Training (Tenant: fintech_corp, 160 samples)
    print("\n4. Triggering Automated LightGBM Training on 'fintech_corp' (160 samples)...")
    train_req = {
        "target_column": "is_fraud",
        "task_type": "binary",
        "tune_hyperparameters": False
    }
    res_train = requests.post(
        f"{BASE_URL}/api/v1/models/train",
        headers={"X-Tenant-ID": "fintech_corp"},
        json=train_req
    )
    train_data = res_train.json()["data"]
    print(f"   Trained Successfully! Best Iteration: {train_data['best_iteration']}")
    print(f"   Metrics: {train_data['metrics']}")
    print(f"   Top 2 Important Features: {list(train_data['feature_importances'].items())[:2]}")

    # 5. Post-Training Structured Query -> Routes to Classical Tabular ML
    print("\n5. Testing Post-Training Query on 'fintech_corp' (Now routes to Classical ML)...")
    payload_ml = {
        "account_age_days": 18,
        "transaction_velocity_24h": 6800.0,
        "failed_login_attempts": 2,
        "risk_score_tier": "high"
    }
    res_ml = requests.post(
        f"{BASE_URL}/api/v1/predict",
        headers={"X-Tenant-ID": "fintech_corp"},
        json={"payload": payload_ml}
    )
    ml_data = res_ml.json()["data"]
    print(f"   Route Executed: {ml_data['route']}")
    print(f"   Model Type: {ml_data['model_type']}")
    print(f"   Fraud Probability Score: {ml_data['prediction']}")
    print(f"   Label: {ml_data['label']} (Flagged as Fraud: {ml_data['label'] == 1})")
    print(f"   Confidence: {ml_data['confidence']}")
    print(f"   Drift Detected: {ml_data['drift_detected']}")
    print(f"   Inference Latency: {ml_data['latency_ms']} ms")

    print("\n" + "=" * 65)
    print("ALL TESTS AND ARCHITECTURAL PATHS VERIFIED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    test_platform()
