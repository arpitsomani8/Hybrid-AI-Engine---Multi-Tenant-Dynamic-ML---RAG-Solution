import pytest
from starlette.testclient import TestClient
from app.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "pinecone_index" in data

def test_missing_tenant_header_raises_400(client):
    res = client.get("/api/v1/schemas")
    assert res.status_code == 400
    assert "Missing required header 'X-Tenant-ID'" in res.json()["detail"]

def test_schema_registration_and_get(client):
    tenant = "api_test_tenant"
    schema_payload = {
        "description": "Test dynamic schema",
        "fields": {
            "amount": {"type": "float", "ge": 0.0},
            "status": {"type": "str", "allowed": ["active", "suspended"]}
        }
    }

    # Register
    res = client.post(
        "/api/v1/schemas",
        json=schema_payload,
        headers={"X-Tenant-ID": tenant}
    )
    assert res.status_code == 200
    assert res.json()["success"] is True

    # Retrieve
    res_get = client.get("/api/v1/schemas", headers={"X-Tenant-ID": tenant})
    assert res_get.status_code == 200
    assert res_get.json()["data"]["tenant_id"] == tenant

def test_cold_start_inference_via_api(client):
    tenant = "api_cold_tenant"
    # Upsert a document first
    doc_payload = {
        "doc_id": "rule_99",
        "text": "Transactions over $1000 from suspended status must be declined immediately.",
        "metadata": {"category": "security"}
    }
    client.post("/api/v1/rag/upsert", json=doc_payload, headers={"X-Tenant-ID": tenant})

    # Call predict without trained model -> triggers cold-start fallback
    pred_res = client.post(
        "/api/v1/predict",
        json={"payload": {"amount": 1500.0, "status": "suspended"}},
        headers={"X-Tenant-ID": tenant}
    )
    assert pred_res.status_code == 200
    res_data = pred_res.json()["data"]
    assert res_data["route"] == "cold_start_rag"
    assert "declined immediately" in res_data["answer"].lower()
