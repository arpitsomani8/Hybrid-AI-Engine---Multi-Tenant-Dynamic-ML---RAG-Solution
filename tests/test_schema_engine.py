import pytest
from app.schema_engine import DynamicSchemaRegistry, SchemaValidator

def test_dynamic_schema_registration_and_validation(tmp_path):
    registry = DynamicSchemaRegistry(storage_dir=tmp_path)
    validator = SchemaValidator(registry)

    schema_def = {
        "description": "Fintech Risk Profile",
        "fields": {
            "account_age_days": {"type": "int", "ge": 0},
            "transaction_velocity": {"type": "float", "ge": 0.0},
            "risk_tier": {"type": "str", "allowed": ["low", "medium", "high"]}
        }
    }

    registry.register_tenant_schema("fintech_corp", schema_def)
    assert registry.has_schema("fintech_corp")

    # Valid payload
    valid_payload = {
        "account_age_days": 45,
        "transaction_velocity": 1250.50,
        "risk_tier": "medium"
    }
    res = validator.validate("fintech_corp", valid_payload)
    assert res.is_valid is True
    assert res.data["account_age_days"] == 45

    # Invalid payload (negative age, invalid tier)
    invalid_payload = {
        "account_age_days": -5,
        "transaction_velocity": 100.0,
        "risk_tier": "ultra_extreme"
    }
    res_inv = validator.validate("fintech_corp", invalid_payload)
    assert res_inv.is_valid is False
    assert len(res_inv.errors) >= 1

def test_drift_detection(tmp_path):
    registry = DynamicSchemaRegistry(storage_dir=tmp_path)
    validator = SchemaValidator(registry)

    validator.set_baseline("fintech_corp", {
        "transaction_velocity": {"mean": 1000.0, "std": 100.0}
    })

    # Normal payload
    normal_payload = {"transaction_velocity": 1050.0}
    report = validator.check_drift("fintech_corp", normal_payload)
    assert report.has_drift is False

    # Outlier / Drifted payload (z-score > 3.0)
    drifted_payload = {"transaction_velocity": 8000.0}
    report_drift = validator.check_drift("fintech_corp", drifted_payload)
    assert report_drift.has_drift is True
    assert "transaction_velocity" in report_drift.drifted_features
