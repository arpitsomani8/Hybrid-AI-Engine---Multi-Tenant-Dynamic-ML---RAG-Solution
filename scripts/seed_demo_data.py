"""
Seed Script: Populates realistic multi-tenant data into the Hybrid AI Engine:
1. Dynamic Schema definitions
2. Pinecone Serverless vector documents (strictly isolated namespaces)
3. Tabular datasets (cold-start vs ready-to-train)
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.core import HybridAIEngine

def seed():
    print("Initializing Hybrid AI Engine for seeding...")
    engine = HybridAIEngine()

    # =========================================================================
    # TENANT 1: fintech_corp (Financial Risk & Transaction Velocity)
    # =========================================================================
    tenant_1 = "fintech_corp"
    print(f"\n[1/2] Seeding Tenant: {tenant_1}")

    # 1. Register Dynamic Schema
    schema_fintech = {
        "description": "Fintech Fraud & Risk Assessment Schema",
        "target_column": "is_fraud",
        "task_type": "binary",
        "fields": {
            "account_age_days": {"type": "int", "ge": 0, "description": "Age of the account in days"},
            "transaction_velocity_24h": {"type": "float", "ge": 0.0, "description": "Total spend in last 24 hours"},
            "failed_login_attempts": {"type": "int", "ge": 0, "description": "Failed logins in last hour"},
            "risk_score_tier": {"type": "str", "default": "medium", "allowed": ["low", "medium", "high"]}
        }
    }
    engine.register_schema(tenant_1, schema_fintech)
    print(f"  [OK] Dynamic schema registered for {tenant_1}")

    # 2. Index Institutional Knowledge Documents into Pinecone namespace 'fintech_corp'
    fintech_docs = [
        (
            "rule_velocity_threshold",
            "Accounts with 24-hour transaction velocity exceeding $5,000 and high risk score tier require mandatory multi-factor authorization and fraud alert escalation.",
            {"category": "compliance", "priority": "critical"}
        ),
        (
            "rule_new_account_hold",
            "New customer accounts created within the last 30 days are subject to an initial transaction cap of $2,500 unless verified via Tier-3 biometric KYC.",
            {"category": "underwriting", "priority": "high"}
        ),
        (
            "rule_brute_force_mitigation",
            "Any account experiencing 3 or more failed login attempts combined with sudden international transaction velocity must be placed in a provisional lock.",
            {"category": "security", "priority": "critical"}
        ),
        (
            "rule_chargeback_mitigation",
            "Medium tier accounts with repeated disputed charges qualify for manual review by the compliance investigation division within 4 business hours.",
            {"category": "operations", "priority": "medium"}
        )
    ]

    for doc_id, text, meta in fintech_docs:
        engine.index_document(tenant_1, doc_id, text, meta)
    print(f"  [OK] Indexed {len(fintech_docs)} policy documents into Pinecone namespace [{tenant_1}]")

    # 3. Save Tabular Training Dataset (150 rows - eligible for automated training)
    np.random.seed(42)
    N1 = 160
    account_age = np.random.randint(1, 365, N1)
    velocity = np.random.exponential(scale=1500, size=N1) + 100
    failed_logins = np.random.choice([0, 1, 2, 3, 4], size=N1, p=[0.6, 0.2, 0.1, 0.06, 0.04])
    tiers = np.random.choice(["low", "medium", "high"], size=N1, p=[0.5, 0.35, 0.15])

    # Fraud logic: high velocity + (high tier or failed logins >= 2)
    is_fraud = ((velocity > 4000) & ((tiers == "high") | (failed_logins >= 2))).astype(int)

    df_fintech = pd.DataFrame({
        "account_age_days": account_age,
        "transaction_velocity_24h": np.round(velocity, 2),
        "failed_login_attempts": failed_logins,
        "risk_score_tier": tiers,
        "is_fraud": is_fraud
    })
    fintech_csv = settings.DATA_STORAGE_DIR / f"{tenant_1}.csv"
    df_fintech.to_csv(fintech_csv, index=False)
    print(f"  [OK] Saved {N1} tabular records to {fintech_csv}")

    # =========================================================================
    # TENANT 2: ecommerce_inc (E-Commerce Customer Retention & Cold-Start)
    # =========================================================================
    tenant_2 = "ecommerce_inc"
    print(f"\n[2/2] Seeding Tenant: {tenant_2}")

    # 1. Register Dynamic Schema
    schema_ecom = {
        "description": "E-Commerce Customer Churn & Lifetime Value Schema",
        "target_column": "will_churn",
        "task_type": "binary",
        "fields": {
            "days_since_last_purchase": {"type": "int", "ge": 0},
            "average_order_value": {"type": "float", "ge": 0.0},
            "support_tickets_opened": {"type": "int", "ge": 0},
            "loyalty_tier": {"type": "str", "default": "standard", "allowed": ["standard", "silver", "gold", "vip"]}
        }
    }
    engine.register_schema(tenant_2, schema_ecom)
    print(f"  [OK] Dynamic schema registered for {tenant_2}")

    # 2. Index Institutional Knowledge Documents into Pinecone namespace 'ecommerce_inc'
    ecom_docs = [
        (
            "retention_vip_policy",
            "VIP and Gold loyalty tier members with more than 45 days since last purchase receive an automated $50 loyalty credit and concierge outreach.",
            {"category": "retention", "priority": "high"}
        ),
        (
            "support_escalation_policy",
            "Customers who have filed 3 or more support tickets within 30 days must be flagged for VIP supervisor satisfaction follow-up.",
            {"category": "customer_success", "priority": "urgent"}
        ),
        (
            "refund_guarantee_policy",
            "All orders placed by members in good standing qualify for unconditional 30-day hassle-free returns with prepaid return labels.",
            {"category": "support", "priority": "medium"}
        )
    ]

    for doc_id, text, meta in ecom_docs:
        engine.index_document(tenant_2, doc_id, text, meta)
    print(f"  [OK] Indexed {len(ecom_docs)} policy documents into Pinecone namespace [{tenant_2}]")

    # 3. Only 25 tabular records for Tenant 2 to demonstrate COLD-START fallback!
    N2 = 25
    df_ecom = pd.DataFrame({
        "days_since_last_purchase": np.random.randint(1, 90, N2),
        "average_order_value": np.round(np.random.uniform(20.0, 300.0, N2), 2),
        "support_tickets_opened": np.random.randint(0, 4, N2),
        "loyalty_tier": np.random.choice(["standard", "silver", "gold"], N2),
        "will_churn": np.random.choice([0, 1], N2)
    })
    ecom_csv = settings.DATA_STORAGE_DIR / f"{tenant_2}.csv"
    df_ecom.to_csv(ecom_csv, index=False)
    print(f"  [OK] Saved {N2} tabular records for {tenant_2} (COLD-START demonstration mode: < 100 rows)")

    print("\n========================================================")
    print("Multi-tenant seeding completed successfully!")
    print(f"  - Pinecone Index: '{settings.PINECONE_INDEX_NAME}' (Live: {engine.pinecone.is_live})")
    print(f"  - Tenant 'fintech_corp': {len(fintech_docs)} docs in Pinecone, {N1} rows (ready for LightGBM training)")
    print(f"  - Tenant 'ecommerce_inc': {len(ecom_docs)} docs in Pinecone, {N2} rows (cold-start RAG active)")
    print("========================================================")

if __name__ == "__main__":
    seed()
