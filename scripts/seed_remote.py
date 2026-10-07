import requests
import json
import numpy as np
import pandas as pd

RENDER_URL = "https://hybrid-ai-engine-multi-tenant-dynamic-ml.onrender.com"
TENANT = "fintech_corp"

def seed_remote():
    print(f"Seeding Tabular Data to Live Render App ({RENDER_URL})...")

    # 1. Generate 160 records for fintech_corp
    np.random.seed(42)
    N = 160
    account_age = np.random.randint(1, 365, N)
    velocity = np.random.exponential(scale=1500, size=N) + 100
    failed_logins = np.random.choice([0, 1, 2, 3, 4], size=N, p=[0.6, 0.2, 0.1, 0.06, 0.04])
    tiers = np.random.choice(["low", "medium", "high"], size=N, p=[0.5, 0.35, 0.15])
    is_fraud = ((velocity > 4000) & ((tiers == "high") | (failed_logins >= 2))).astype(int)

    records = []
    for i in range(N):
        records.append({
            "account_age_days": int(account_age[i]),
            "transaction_velocity_24h": float(round(velocity[i], 2)),
            "failed_login_attempts": int(failed_logins[i]),
            "risk_score_tier": str(tiers[i]),
            "is_fraud": int(is_fraud[i])
        })

    # Ingest records into Render app
    print(f"Ingesting 160 records to '{TENANT}' on Render...")
    r = requests.post(f"{RENDER_URL}/api/v1/data/ingest", json={"records": records}, headers={"X-Tenant-ID": TENANT})
    print("Ingest Result:", r.json())

    # Trigger LightGBM Training on Render app
    print(f"Triggering LightGBM Training on Render for '{TENANT}'...")
    train_req = {
        "target_column": "is_fraud",
        "task_type": "binary",
        "tune_hyperparameters": True
    }
    r_train = requests.post(f"{RENDER_URL}/api/v1/models/train", json=train_req, headers={"X-Tenant-ID": TENANT})
    print("Train Result:", r_train.json())

    print("\n[OK] Render container is now seeded and trained! Refresh your browser tab to see 160 samples & LightGBM Ready!")

if __name__ == "__main__":
    seed_remote()
