import requests
import json
import numpy as np

RENDER_URL = "https://hybrid-ai-engine-multi-tenant-dynamic-ml.onrender.com"

def seed_all_remote():
    print(f"--- Seeding Both Tenants to Live Render App ({RENDER_URL}) ---")

    # =========================================================================
    # TENANT 1: fintech_corp (160 rows -> Model Trained, Ready)
    # =========================================================================
    tenant_1 = "fintech_corp"
    np.random.seed(42)
    N1 = 160
    account_age = np.random.randint(1, 365, N1)
    velocity = np.random.exponential(scale=1500, size=N1) + 100
    failed_logins = np.random.choice([0, 1, 2, 3, 4], size=N1, p=[0.6, 0.2, 0.1, 0.06, 0.04])
    tiers = np.random.choice(["low", "medium", "high"], size=N1, p=[0.5, 0.35, 0.15])
    is_fraud = ((velocity > 4000) & ((tiers == "high") | (failed_logins >= 2))).astype(int)

    records_1 = []
    for i in range(N1):
        records_1.append({
            "account_age_days": int(account_age[i]),
            "transaction_velocity_24h": float(round(velocity[i], 2)),
            "failed_login_attempts": int(failed_logins[i]),
            "risk_score_tier": str(tiers[i]),
            "is_fraud": int(is_fraud[i])
        })

    print(f"\n[1/2] Ingesting {N1} records for '{tenant_1}'...")
    r1 = requests.post(f"{RENDER_URL}/api/v1/data/ingest", json={"records": records_1}, headers={"X-Tenant-ID": tenant_1})
    print("Ingest Result:", r1.json()["message"])

    print(f"Training LightGBM model for '{tenant_1}'...")
    r1_train = requests.post(
        f"{RENDER_URL}/api/v1/models/train",
        json={"target_column": "is_fraud", "task_type": "binary", "tune_hyperparameters": False},
        headers={"X-Tenant-ID": tenant_1}
    )
    print("Train Result:", r1_train.json()["message"])

    # =========================================================================
    # TENANT 2: ecommerce_inc (25 rows -> COLD-START Active mode)
    # =========================================================================
    tenant_2 = "ecommerce_inc"
    N2 = 25
    days_since = np.random.randint(1, 90, N2)
    avg_order = np.round(np.random.uniform(20.0, 300.0, N2), 2)
    tickets = np.random.randint(0, 4, N2)
    loyalty = np.random.choice(["standard", "silver", "gold"], N2)
    churn = np.random.choice([0, 1], N2)

    records_2 = []
    for i in range(N2):
        records_2.append({
            "days_since_last_purchase": int(days_since[i]),
            "average_order_value": float(avg_order[i]),
            "support_tickets_opened": int(tickets[i]),
            "loyalty_tier": str(loyalty[i]),
            "will_churn": int(churn[i])
        })

    print(f"\n[2/2] Ingesting {N2} records for '{tenant_2}' (Cold-Start mode: < 100 rows)...")
    r2 = requests.post(f"{RENDER_URL}/api/v1/data/ingest", json={"records": records_2}, headers={"X-Tenant-ID": tenant_2})
    print("Ingest Result:", r2.json()["message"])

    print("\n[OK] BOTH TENANTS SUCCESSFULLY SEEDED & SYNCHRONIZED ON RENDER!")

if __name__ == "__main__":
    seed_all_remote()
