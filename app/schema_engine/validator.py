import logging
from typing import Dict, Any, List, Optional, Tuple
from pydantic import ValidationError, BaseModel
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger("HybridAIEngine.SchemaValidator")

@dataclass
class ValidationResult:
    is_valid: bool
    data: Optional[Dict[str, Any]] = None
    errors: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class DriftReport:
    has_drift: bool
    drifted_features: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

class SchemaValidator:
    """
    Validates tenant incoming payloads against their registered dynamic schema
    and optionally inspects data for feature drift against baseline statistics.
    """
    def __init__(self, registry):
        self.registry = registry
        self._baselines: Dict[str, Dict[str, Dict[str, float]]] = {}

    def validate(self, tenant_id: str, payload: Dict[str, Any]) -> ValidationResult:
        model_cls = self.registry.get_model(tenant_id)
        if not model_cls:
            return ValidationResult(
                is_valid=False,
                errors=[{"error": f"No registered schema found for tenant '{tenant_id}'."}]
            )

        try:
            validated_obj = model_cls(**payload)
            return ValidationResult(
                is_valid=True,
                data=validated_obj.model_dump()
            )
        except ValidationError as e:
            formatted_errors = []
            for err in e.errors():
                loc = ".".join(str(p) for p in err.get("loc", []))
                formatted_errors.append({
                    "field": loc,
                    "message": err.get("msg"),
                    "type": err.get("type")
                })
            return ValidationResult(is_valid=False, errors=formatted_errors)

    def set_baseline(self, tenant_id: str, baseline_stats: Dict[str, Dict[str, float]]):
        """
        Sets baseline statistics for numerical features (e.g., mean, std, min, max).
        """
        self._baselines[tenant_id] = baseline_stats
        logger.info(f"Recorded feature baseline for tenant: {tenant_id}")

    def check_drift(self, tenant_id: str, payload: Dict[str, Any], z_threshold: float = 3.0) -> DriftReport:
        """
        Performs quick z-score check for numerical fields to detect significant anomaly or drift.
        """
        baseline = self._baselines.get(tenant_id)
        if not baseline:
            return DriftReport(has_drift=False)

        drifted = []
        details = {}

        for feat, stats in baseline.items():
            if feat in payload and isinstance(payload[feat], (int, float)):
                val = float(payload[feat])
                mean = stats.get("mean", 0.0)
                std = stats.get("std", 1.0)
                if std > 0:
                    z_score = abs(val - mean) / std
                    if z_score > z_threshold:
                        drifted.append(feat)
                        details[feat] = {
                            "value": val,
                            "baseline_mean": mean,
                            "baseline_std": std,
                            "z_score": round(z_score, 2),
                            "alert": f"Value {val} deviates by {z_score:.2f} standard deviations"
                        }

        return DriftReport(
            has_drift=len(drifted) > 0,
            drifted_features=drifted,
            details=details
        )
