import pytest
import numpy as np
import pandas as pd
from app.ml_engine import AutomatedMLPipeline, ModelRegistry

def test_ml_pipeline_training_and_threshold(tmp_path):
    registry = ModelRegistry(base_dir=tmp_path)
    pipeline = AutomatedMLPipeline(min_samples_to_train=100, model_registry=registry)

    # 1. Dataset with only 30 samples (< 100) should raise ValueError
    df_small = pd.DataFrame({
        "age": np.random.randint(20, 60, 30),
        "spend": np.random.uniform(10, 500, 30),
        "is_fraud": np.random.choice([0, 1], 30)
    })

    with pytest.raises(ValueError) as excinfo:
        pipeline.train_tenant_model("test_tenant", df_small, target_col="is_fraud")
    assert "Insufficient training records" in str(excinfo.value)

    # 2. Dataset with 120 samples (>= 100) should train successfully
    np.random.seed(42)
    N = 120
    df_large = pd.DataFrame({
        "age": np.random.randint(20, 60, N),
        "spend": np.random.uniform(10, 500, N),
        "tier": np.random.choice(["bronze", "silver", "gold"], N),
    })
    df_large["is_fraud"] = ((df_large["spend"] > 250) & (df_large["tier"] == "gold")).astype(int)

    result = pipeline.train_tenant_model(
        "test_tenant",
        df_large,
        target_col="is_fraud",
        task="binary",
        tune_hyperparameters=False
    )

    assert result.num_samples == 120
    assert "auc" in result.metrics
    assert pipeline.has_model("test_tenant")

    # 3. Predict with valid features
    pred = pipeline.predict("test_tenant", {"age": 35, "spend": 400.0, "tier": "gold"})
    assert pred["mode"] == "classical_ml"
    assert "prediction" in pred
    assert "confidence" in pred
    assert pred["label"] in [0, 1]
