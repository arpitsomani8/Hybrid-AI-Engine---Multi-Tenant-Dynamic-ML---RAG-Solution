import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import lightgbm as lgb
from app.config import settings

logger = logging.getLogger("HybridAIEngine.ModelRegistry")

class ModelRegistry:
    """
    Manages persistence of trained LightGBM models, preprocessor states,
    and performance metadata strictly isolated by tenant_id.
    """
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or settings.MODEL_ARTIFACTS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _tenant_dir(self, tenant_id: str) -> Path:
        d = self.base_dir / tenant_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_model(
        self,
        tenant_id: str,
        booster: lgb.Booster,
        metadata: Dict[str, Any],
        preprocessor_state: Dict[str, Any]
    ):
        t_dir = self._tenant_dir(tenant_id)
        
        # Save model booster text file
        model_path = t_dir / "model.txt"
        booster.save_model(str(model_path))

        # Save metadata (metrics, feature importance, parameters)
        meta_path = t_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Save preprocessor state
        prep_path = t_dir / "preprocessor.json"
        with open(prep_path, "w", encoding="utf-8") as f:
            json.dump(preprocessor_state, f, indent=2)

        logger.info(f"Successfully saved model artifacts for tenant '{tenant_id}' at {t_dir}")

    def load_model(self, tenant_id: str) -> Optional[lgb.Booster]:
        model_path = self._tenant_dir(tenant_id) / "model.txt"
        if not model_path.exists():
            return None
        return lgb.Booster(model_file=str(model_path))

    def load_metadata(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        meta_path = self._tenant_dir(tenant_id) / "metadata.json"
        if not meta_path.exists():
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_preprocessor_state(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        prep_path = self._tenant_dir(tenant_id) / "preprocessor.json"
        if not prep_path.exists():
            return None
        with open(prep_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def has_model(self, tenant_id: str) -> bool:
        model_path = self._tenant_dir(tenant_id) / "model.txt"
        return model_path.exists()

    def list_tenants_with_models(self) -> List[str]:
        return [p.name for p in self.base_dir.iterdir() if p.is_dir() and (p / "model.txt").exists()]
