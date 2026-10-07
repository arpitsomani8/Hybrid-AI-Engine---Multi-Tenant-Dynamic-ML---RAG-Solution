import json
import logging
from pathlib import Path
from typing import Dict, Any, Type, Optional, Union, List, Literal
from pydantic import BaseModel, create_model, Field

logger = logging.getLogger("HybridAIEngine.SchemaRegistry")

TYPE_MAPPING = {
    "int": int,
    "integer": int,
    "float": float,
    "number": float,
    "str": str,
    "string": str,
    "bool": bool,
    "boolean": bool,
}

class DynamicSchemaRegistry:
    """
    Manages tenant-specific dynamic Pydantic models.
    Supports persistent schema definitions and runtime validation model generation.
    """
    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or Path("artifacts/schemas")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, Type[BaseModel]] = {}
        self._raw_definitions: Dict[str, Dict[str, Any]] = {}
        self._load_persisted_schemas()

    def _load_persisted_schemas(self):
        """Loads any previously registered schemas from the storage directory."""
        for schema_file in self.storage_dir.glob("*.json"):
            try:
                tenant_id = schema_file.stem
                with open(schema_file, "r", encoding="utf-8") as f:
                    schema_def = json.load(f)
                self.register_tenant_schema(tenant_id, schema_def, persist=False)
                logger.info(f"Loaded persisted schema for tenant: {tenant_id}")
            except Exception as e:
                logger.error(f"Failed to load schema from {schema_file}: {e}")

    def register_tenant_schema(
        self,
        tenant_id: str,
        schema_definition: Dict[str, Any],
        persist: bool = True
    ) -> Type[BaseModel]:
        """
        Registers a dynamic schema for a tenant.
        schema_definition format:
        {
            "description": "Fintech Risk Profile Schema",
            "target_column": "is_fraud",
            "task_type": "binary",  # or "regression"
            "fields": {
                "account_age_days": {"type": "int", "ge": 0, "description": "Account age in days"},
                "transaction_velocity_24h": {"type": "float", "ge": 0.0},
                "risk_score_tier": {"type": "str", "default": "medium", "allowed": ["low", "medium", "high"]}
            }
        }
        """
        fields = schema_definition.get("fields", {})
        if not fields:
            raise ValueError(f"Schema definition for tenant '{tenant_id}' must contain a non-empty 'fields' dict.")

        pydantic_field_defs = {}
        for field_name, rules in fields.items():
            py_type = TYPE_MAPPING.get(rules.get("type", "str").lower(), str)
            
            # Check for allowed values
            if "allowed" in rules and isinstance(rules["allowed"], list) and rules["allowed"]:
                # Use Literal or string validation
                allowed_vals = tuple(rules["allowed"])
                py_type = Literal[allowed_vals] # type: ignore

            field_kwargs = {}
            if "default" in rules:
                field_kwargs["default"] = rules["default"]
            elif rules.get("required", True):
                field_kwargs["default"] = ...
            else:
                field_kwargs["default"] = None
                py_type = Optional[py_type]

            for constraint in ["ge", "le", "gt", "lt", "min_length", "max_length"]:
                if constraint in rules:
                    field_kwargs[constraint] = rules[constraint]

            if "description" in rules:
                field_kwargs["description"] = rules["description"]

            pydantic_field_defs[field_name] = (py_type, Field(**field_kwargs))

        model_cls = create_model(f"DynamicTenant_{tenant_id}", **pydantic_field_defs)
        self._models[tenant_id] = model_cls
        self._raw_definitions[tenant_id] = schema_definition

        if persist:
            schema_path = self.storage_dir / f"{tenant_id}.json"
            with open(schema_path, "w", encoding="utf-8") as f:
                json.dump(schema_definition, f, indent=2)
            logger.info(f"Persisted schema for tenant: {tenant_id} at {schema_path}")

        logger.info(f"Successfully registered dynamic model for tenant '{tenant_id}' with {len(fields)} fields")
        return model_cls

    def get_model(self, tenant_id: str) -> Optional[Type[BaseModel]]:
        return self._models.get(tenant_id)

    def get_raw_definition(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        return self._raw_definitions.get(tenant_id)

    def has_schema(self, tenant_id: str) -> bool:
        return tenant_id in self._models

    def list_tenants(self) -> List[str]:
        return list(self._models.keys())

    def delete_schema(self, tenant_id: str):
        if tenant_id in self._models:
            del self._models[tenant_id]
        if tenant_id in self._raw_definitions:
            del self._raw_definitions[tenant_id]
        schema_path = self.storage_dir / f"{tenant_id}.json"
        if schema_path.exists():
            schema_path.unlink()
        logger.info(f"Deleted dynamic schema for tenant '{tenant_id}'")
