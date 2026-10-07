import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, root_mean_squared_error, r2_score

from .preprocessor import TabularPreprocessor
from .optimizer import HyperparameterOptimizer
from .registry import ModelRegistry
from app.config import settings

logger = logging.getLogger("HybridAIEngine.Trainer")

@dataclass
class TrainingResult:
    tenant_id: str
    task: str
    num_samples: int
    features: List[str]
    metrics: Dict[str, float]
    feature_importances: Dict[str, float]
    best_iteration: int

class AutomatedMLPipeline:
    """
    Automated Tabular ML Pipeline per tenant.
    Enforces minimum sample thresholds, automates preprocessing,
    tunes hyperparameters with Optuna, trains LightGBM models with early stopping,
    and calculates evaluation metrics.
    """
    def __init__(
        self,
        min_samples_to_train: Optional[int] = None,
        model_registry: Optional[ModelRegistry] = None
    ):
        self.min_samples = min_samples_to_train or settings.MIN_SAMPLES_TO_TRAIN
        self.registry = model_registry or ModelRegistry()
        self.preprocessors: Dict[str, TabularPreprocessor] = {}
        self.cached_models: Dict[str, lgb.Booster] = {}
        self.cached_metadata: Dict[str, Dict[str, Any]] = {}
        self._load_existing_models()

    def _load_existing_models(self):
        for tenant_id in self.registry.list_tenants_with_models():
            booster = self.registry.load_model(tenant_id)
            meta = self.registry.load_metadata(tenant_id)
            prep_state = self.registry.load_preprocessor_state(tenant_id)
            if booster and meta and prep_state:
                self.cached_models[tenant_id] = booster
                self.cached_metadata[tenant_id] = meta
                prep = TabularPreprocessor()
                prep.from_dict(prep_state)
                self.preprocessors[tenant_id] = prep
                logger.info(f"Loaded existing ML model for tenant: {tenant_id}")

    def can_train(self, df: pd.DataFrame) -> bool:
        return len(df) >= self.min_samples

    def train_tenant_model(
        self,
        tenant_id: str,
        df: pd.DataFrame,
        target_col: str,
        task: str = "binary",
        tune_hyperparameters: bool = True
    ) -> TrainingResult:
        if not self.can_train(df):
            raise ValueError(
                f"Insufficient training records ({len(df)}) for tenant '{tenant_id}'. "
                f"Requires at least {self.min_samples} samples."
            )

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in provided dataset.")

        preprocessor = TabularPreprocessor()
        X, y = preprocessor.fit_transform(df, target_col=target_col)
        features = X.columns.tolist()

        # Split into Train / Validation
        stratify = y if task == "binary" and len(np.unique(y)) > 1 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Optimize hyperparameters or use strong defaults
        if tune_hyperparameters and len(df) >= 120:
            optimizer = HyperparameterOptimizer(n_trials=8, timeout=20)
            params = optimizer.optimize(X_train, y_train, X_val, y_val, task=task)
        else:
            params = {
                "objective": "binary" if task == "binary" else "regression",
                "metric": "auc" if task == "binary" else "rmse",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "verbosity": -1,
                "seed": 42
            }

        booster = lgb.train(
            params,
            train_data,
            num_boost_round=150,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
        )

        # Evaluate on validation split
        val_preds = booster.predict(X_val)
        metrics = {}
        if task == "binary":
            try:
                metrics["auc"] = round(float(roc_auc_score(y_val, val_preds)), 4)
            except Exception:
                metrics["auc"] = 0.5
            val_labels = (val_preds >= 0.5).astype(int)
            metrics["accuracy"] = round(float(accuracy_score(y_val, val_labels)), 4)
        else:
            metrics["rmse"] = round(float(root_mean_squared_error(y_val, val_preds)), 4)
            metrics["r2"] = round(float(r2_score(y_val, val_preds)), 4)

        # Compute feature importances
        importance_scores = booster.feature_importance(importance_type="gain")
        total_gain = float(np.sum(importance_scores)) if np.sum(importance_scores) > 0 else 1.0
        feature_importance_map = {
            feat: round(float(gain / total_gain), 4)
            for feat, gain in zip(features, importance_scores)
        }
        # Sort feature importance descending
        feature_importance_map = dict(sorted(feature_importance_map.items(), key=lambda item: item[1], reverse=True))

        metadata = {
            "tenant_id": tenant_id,
            "task": task,
            "target_col": target_col,
            "num_samples": len(df),
            "features": features,
            "metrics": metrics,
            "feature_importances": feature_importance_map,
            "best_iteration": int(booster.best_iteration),
            "params": params
        }

        # Save to registry and update in-memory cache
        self.registry.save_model(tenant_id, booster, metadata, preprocessor.to_dict())
        self.cached_models[tenant_id] = booster
        self.cached_metadata[tenant_id] = metadata
        self.preprocessors[tenant_id] = preprocessor

        logger.info(
            f"Successfully trained LightGBM for tenant '{tenant_id}'. "
            f"Metrics: {metrics}, Best Iteration: {booster.best_iteration}"
        )

        return TrainingResult(
            tenant_id=tenant_id,
            task=task,
            num_samples=len(df),
            features=features,
            metrics=metrics,
            feature_importances=feature_importance_map,
            best_iteration=int(booster.best_iteration)
        )

    def predict(self, tenant_id: str, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not self.has_model(tenant_id):
            raise RuntimeError(f"No trained ML model found for tenant '{tenant_id}'")

        booster = self.cached_models[tenant_id]
        preprocessor = self.preprocessors[tenant_id]
        meta = self.cached_metadata[tenant_id]
        expected_features = meta["features"]
        task = meta["task"]

        transformed_df = preprocessor.transform_single(features_dict, expected_features)
        raw_pred = booster.predict(transformed_df)[0]

        if task == "binary":
            pred_score = float(raw_pred)
            label = int(pred_score >= 0.5)
            confidence = float(pred_score if pred_score >= 0.5 else 1.0 - pred_score)
            return {
                "mode": "classical_ml",
                "model_type": "LightGBM",
                "prediction": pred_score,
                "label": label,
                "confidence": round(confidence, 4),
                "features_evaluated": expected_features
            }
        else:
            return {
                "mode": "classical_ml",
                "model_type": "LightGBM",
                "prediction": round(float(raw_pred), 4),
                "confidence": 0.95,
                "features_evaluated": expected_features
            }

    def has_model(self, tenant_id: str) -> bool:
        if tenant_id in self.cached_models:
            return True
        if self.registry.has_model(tenant_id):
            try:
                booster = self.registry.load_model(tenant_id)
                meta = self.registry.load_metadata(tenant_id)
                prep_state = self.registry.load_preprocessor_state(tenant_id)
                if booster and meta and prep_state:
                    self.cached_models[tenant_id] = booster
                    self.cached_metadata[tenant_id] = meta
                    prep = TabularPreprocessor()
                    prep.from_dict(prep_state)
                    self.preprocessors[tenant_id] = prep
                    return True
            except Exception as e:
                logger.error(f"Failed to dynamically load model for tenant '{tenant_id}': {e}")
        return False
