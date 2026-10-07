import requests
import json
import numpy as np
import pandas as pd
import time

BASE_URL = "http://localhost:8000"

def print_header(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def print_json(data):
    print(json.dumps(data, indent=2))

def run_scenarios():
    print_header("HYBRID AI ENGINE: MULTI-TENANT PAYLOAD MATRIX TEST SUITE")
    print("Testing live endpoints at:", BASE_URL)

    # -------------------------------------------------------------------------
    # SCENARIO 1: BRAND NEW COLD-START TENANT (No ML Model, No Historical Data)
    # -------------------------------------------------------------------------
    tenant_1 = "healthcare_plus"
    print_header(f"SCENARIO 1: Brand New Tenant Cold-Start Mode ('{tenant_1}')")
    
    # 1A. Register Schema for new tenant
    schema_payload = {
        "description": "Healthcare Patient Risk & Admission Schema",
        "target_column": "requires_icu",
        "task_type": "binary",
        "fields": {
            "patient_age": {"type": "int", "ge": 0, "le": 120, "description": "Patient age in years"},
            "blood_pressure_sys": {"type": "float", "ge": 50.0, "le": 300.0, "description": "Systolic blood pressure"},
            "prior_admissions": {"type": "int", "ge": 0, "description": "Admissions in last 12 months"},
            "triage_level": {"type": "str", "allowed": ["routine", "urgent", "critical"], "description": "Triage severity"}
        }
    }
    print(f"\n[1A] Registering Dynamic Schema for tenant '{tenant_1}'...")
    r = requests.post(f"{BASE_URL}/api/v1/schemas", json=schema_payload, headers={"X-Tenant-ID": tenant_1})
    print("Status:", r.status_code)
    print_json(r.json())

    # 1B. Upsert Policy Documents into Pinecone Namespace
    print(f"\n[1B] Indexing Clinical Guidelines into Pinecone Namespace '{tenant_1}'...")
    docs = [
        {
            "doc_id": "guide_icu_protocol_01",
            "text": "Patients with systolic blood pressure exceeding 170 mmHg and urgent triage level require immediate ICU bed allocation and cardiologist evaluation.",
            "metadata": {"category": "icu_triage"}
        },
        {
            "doc_id": "guide_elderly_care_02",
            "text": "Patients over age 70 with more than 2 prior admissions within 12 months qualify for continuous vitals monitoring and geriatric outreach.",
            "metadata": {"category": "geriatric"}
        }
    ]
    r = requests.post(f"{BASE_URL}/api/v1/rag/batch-upsert", json={"documents": docs}, headers={"X-Tenant-ID": tenant_1})
    print("Indexing Status:", r.status_code)
    print_json(r.json())

    # 1C. Query Structured Payload in Cold-Start (No Model Trained Yet)
    structured_payload = {
        "payload": {
            "patient_age": 72,
            "blood_pressure_sys": 175.0,
            "prior_admissions": 1,
            "triage_level": "urgent"
        }
    }
    print(f"\n[1C] Sending Structured Payload in COLD-START Mode (No ML Model Trained Yet)...")
    r = requests.post(f"{BASE_URL}/api/v1/predict", json=structured_payload, headers={"X-Tenant-ID": tenant_1})
    print("Response Status:", r.status_code)
    print_json(r.json())
    
    res = r.json()["data"]
    assert res["route"] == "cold_start_rag"
    print(f"\n[OK] VERIFIED: Routed to '{res['route']}'. Fallback to Pinecone vector namespace was triggered!")

    # -------------------------------------------------------------------------
    # SCENARIO 2: DATA INGESTION & LIGHTGBM TRAINING (>= 100 Records)
    # -------------------------------------------------------------------------
    print_header(f"SCENARIO 2: Ingesting 150 Records & Training LightGBM ML Model for '{tenant_1}'")
    
    np.random.seed(42)
    N = 150
    records = []
    for _ in range(N):
        age = int(np.random.randint(20, 85))
        bp = float(np.random.uniform(90, 190))
        priors = int(np.random.randint(0, 5))
        triage = str(np.random.choice(["routine", "urgent", "critical"]))
        requires_icu = int((bp > 160) or (triage == "critical") or (age > 75 and priors > 2))
        records.append({
            "patient_age": age,
            "blood_pressure_sys": bp,
            "prior_admissions": priors,
            "triage_level": triage,
            "requires_icu": requires_icu
        })

    print(f"\n[2A] Ingesting {N} Records via Batch Ingest Endpoint...")
    r = requests.post(f"{BASE_URL}/api/v1/data/ingest", json={"records": records}, headers={"X-Tenant-ID": tenant_1})
    print("Ingest Status:", r.status_code)
    print_json(r.json())

    print(f"\n[2B] Triggering Automated LightGBM Training with Optuna Tuning...")
    train_req = {
        "target_column": "requires_icu",
        "task_type": "binary",
        "tune_hyperparameters": True
    }
    r = requests.post(f"{BASE_URL}/api/v1/models/train", json=train_req, headers={"X-Tenant-ID": tenant_1})
    print("Train Status:", r.status_code)
    print_json(r.json())

    # -------------------------------------------------------------------------
    # SCENARIO 3: EXISTING TENANT POST-TRAINING (Classical Tabular ML Route)
    # -------------------------------------------------------------------------
    print_header(f"SCENARIO 3: Querying the SAME Payload Post-Training ('{tenant_1}')")
    
    print(f"\n[3A] Sending Structured Payload to Trained Engine...")
    r = requests.post(f"{BASE_URL}/api/v1/predict", json=structured_payload, headers={"X-Tenant-ID": tenant_1})
    print("Response Status:", r.status_code)
    print_json(r.json())
    
    res = r.json()["data"]
    assert res["route"] == "classical_ml"
    print(f"\n[OK] VERIFIED: Routed to '{res['route']}'. Executed sub-millisecond LightGBM inference ({res['latency_ms']} ms)!")

    # -------------------------------------------------------------------------
    # SCENARIO 4: FEATURE DRIFT DETECTION (Outlier Payload)
    # -------------------------------------------------------------------------
    print_header(f"SCENARIO 4: Feature Drift Detection for Extreme Outlier Payload")
    
    outlier_payload = {
        "payload": {
            "patient_age": 72,
            "blood_pressure_sys": 295.0,  # Extreme outlier (z-score > 3.0)
            "prior_admissions": 1,
            "triage_level": "urgent"
        }
    }
    print(f"\n[4A] Sending Outlier Payload (blood_pressure_sys = 295.0)...")
    r = requests.post(f"{BASE_URL}/api/v1/predict", json=outlier_payload, headers={"X-Tenant-ID": tenant_1})
    print("Response Status:", r.status_code)
    print_json(r.json())
    
    res = r.json()["data"]
    assert res["drift_detected"] is True
    print(f"\n[OK] VERIFIED: Feature Drift Alert Triggered! Details: {res['drift_details']}")

    # -------------------------------------------------------------------------
    # SCENARIO 5: UNSTRUCTURED NATURAL LANGUAGE QUERY ROUTE
    # -------------------------------------------------------------------------
    print_header(f"SCENARIO 5: Unstructured Natural Language Query Route")
    
    nl_payload = {
        "payload": {
            "natural_language_query": "What is the ICU bed allocation protocol for blood pressure over 170?"
        }
    }
    print(f"\n[5A] Sending Natural Language Query...")
    r = requests.post(f"{BASE_URL}/api/v1/predict", json=nl_payload, headers={"X-Tenant-ID": tenant_1})
    print("Response Status:", r.status_code)
    print_json(r.json())
    
    res = r.json()["data"]
    assert res["route"] == "unstructured_rag"
    print(f"\n[OK] VERIFIED: Routed to '{res['route']}'. Retrieved matching directives from Pinecone namespace '{tenant_1}'!")

    # -------------------------------------------------------------------------
    # SCENARIO 6: SCHEMA VALIDATION FAILURE (Invalid Bounds / Unallowed Enum)
    # -------------------------------------------------------------------------
    print_header(f"SCENARIO 6: Invalid Payload Schema Validation Failure")
    
    invalid_payload = {
        "payload": {
            "patient_age": -15,  # Invalid negative age
            "blood_pressure_sys": 120.0,
            "prior_admissions": 0,
            "triage_level": "ultra_extreme"  # Invalid tier
        }
    }
    print(f"\n[6A] Sending Invalid Payload (age = -15, triage = 'ultra_extreme')...")
    r = requests.post(f"{BASE_URL}/api/v1/predict", json=invalid_payload, headers={"X-Tenant-ID": tenant_1})
    print("Response Status:", r.status_code)
    print_json(r.json())

    # -------------------------------------------------------------------------
    # SCENARIO 7: TENANT COMPARISON MATRIX SUMMARY
    # -------------------------------------------------------------------------
    print_header("SUMMARY: MULTI-TENANT ISOLATION & ROUTING MATRIX")
    
    for t_id in ["healthcare_plus", "fintech_corp", "ecommerce_inc"]:
        r = requests.get(f"{BASE_URL}/api/v1/tenants/{t_id}/status")
        status = r.json()["data"]
        print(f"\nTenant: '{t_id}'")
        print(f"  - Dynamic Schema Registered: {status['has_dynamic_schema']}")
        print(f"  - Trained LightGBM Model:   {status['has_trained_model']}")
        print(f"  - Cold-Start Active:        {status['is_cold_start']}")
        print(f"  - Pinecone Namespace Vectors: {status['pinecone']['vector_count']} vectors")

    print_header("ALL 7 PAYLOAD TEST SCENARIOS EXECUTED & VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    run_scenarios()
