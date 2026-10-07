import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

from app.schema_engine import DynamicSchemaRegistry, SchemaValidator
from app.ml_engine import AutomatedMLPipeline, ModelRegistry
from app.rag_engine import VectorEmbedder, PineconeManager, ContextSynthesizer
from .router import IntelligentRouter, ExecutionRoute

logger = logging.getLogger("HybridAIEngine.Orchestrator")

class HybridAIEngine:
    """
    Central Zero-to-One Hybrid Intelligence Orchestrator.
    Seamlessly integrates:
    - Dynamic Runtime Schema Registration & Validation
    - Automated Classical ML Training (LightGBM + Optuna)
    - Pinecone Serverless Multi-Tenant Vector RAG (Strict Namespace Isolation)
    - Intelligent Routing between Tabular ML, Cold-Start Fallback, and Natural Language RAG
    """
    def __init__(self, model_dir: Optional[Path] = None, schema_dir: Optional[Path] = None):
        logger.info("Initializing Hybrid AI Engine components...")
        self.schema_registry = DynamicSchemaRegistry(storage_dir=schema_dir)
        self.schema_validator = SchemaValidator(self.schema_registry)
        self.model_registry = ModelRegistry(base_dir=model_dir)
        self.ml_pipeline = AutomatedMLPipeline(model_registry=self.model_registry)
        self.embedder = VectorEmbedder()
        self.pinecone = PineconeManager()
        self.synthesizer = ContextSynthesizer()
        self.router = IntelligentRouter()
        logger.info("Hybrid AI Engine successfully initialized.")

    def process_query(self, tenant_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unified Hybrid Inference Entrypoint.
        Routes intelligently based on query type, tenant schema, and trained model availability.
        """
        start_time = time.time()
        has_model = self.ml_pipeline.has_model(tenant_id)

        route, route_context = self.router.classify_request(
            tenant_id=tenant_id,
            payload=request_data,
            has_trained_model=has_model
        )

        logger.info(f"Query for tenant '{tenant_id}' resolved to route: {route.value}")

        # -------------------------------------------------------------
        # ROUTE 1: CLASSICAL TABULAR ML
        # -------------------------------------------------------------
        if route == ExecutionRoute.CLASSICAL_ML:
            # Validate input against dynamic schema
            val_res = self.schema_validator.validate(tenant_id, request_data)
            if not val_res.is_valid:
                return {
                    "tenant_id": tenant_id,
                    "route": route.value,
                    "status": "validation_error",
                    "errors": val_res.errors,
                    "latency_ms": round((time.time() - start_time) * 1000, 2)
                }

            # Check feature drift against baseline
            drift_report = self.schema_validator.check_drift(tenant_id, val_res.data)

            # Predict with LightGBM
            prediction_output = self.ml_pipeline.predict(tenant_id, val_res.data)
            prediction_output["tenant_id"] = tenant_id
            prediction_output["route"] = route.value
            prediction_output["drift_detected"] = drift_report.has_drift
            prediction_output["drift_details"] = drift_report.details if drift_report.has_drift else None
            prediction_output["latency_ms"] = round((time.time() - start_time) * 1000, 2)
            return prediction_output

        # -------------------------------------------------------------
        # ROUTE 2: COLD-START RAG FALLBACK
        # -------------------------------------------------------------
        elif route == ExecutionRoute.COLD_START_RAG:
            query_str = route_context["query"]
            query_vec = self.embedder.embed_text(query_str)
            top_matches = self.pinecone.query_similarity(tenant_id, query_vec, top_k=3)

            synthesis = self.synthesizer.synthesize(
                tenant_id=tenant_id,
                query=query_str,
                retrieved_matches=top_matches,
                is_cold_start=True
            )
            synthesis["tenant_id"] = tenant_id
            synthesis["route"] = route.value
            synthesis["evaluated_payload"] = route_context["raw_payload"]
            synthesis["latency_ms"] = round((time.time() - start_time) * 1000, 2)
            return synthesis

        # -------------------------------------------------------------
        # ROUTE 3: UNSTRUCTURED NATURAL LANGUAGE RAG
        # -------------------------------------------------------------
        else:
            query_str = route_context["query"]
            query_vec = self.embedder.embed_text(query_str)
            top_matches = self.pinecone.query_similarity(tenant_id, query_vec, top_k=3)

            synthesis = self.synthesizer.synthesize(
                tenant_id=tenant_id,
                query=query_str,
                retrieved_matches=top_matches,
                is_cold_start=False
            )
            synthesis["tenant_id"] = tenant_id
            synthesis["route"] = route.value
            synthesis["latency_ms"] = round((time.time() - start_time) * 1000, 2)
            return synthesis

    def index_document(self, tenant_id: str, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Embeds and indexes knowledge documents strictly into the tenant's Pinecone namespace.
        """
        meta = metadata or {}
        meta["text"] = text
        vector_values = self.embedder.embed_text(text)
        
        record = {
            "id": doc_id,
            "values": vector_values,
            "metadata": meta
        }
        res = self.pinecone.upsert_vectors(tenant_id, [record])
        logger.info(f"Indexed document '{doc_id}' into Pinecone namespace '{tenant_id}'")
        return res

    def register_schema(self, tenant_id: str, schema_def: Dict[str, Any]):
        return self.schema_registry.register_tenant_schema(tenant_id, schema_def)

    def train_model(
        self,
        tenant_id: str,
        df: pd.DataFrame,
        target_col: str,
        task: str = "binary",
        tune_hyperparameters: bool = True
    ):
        result = self.ml_pipeline.train_tenant_model(
            tenant_id=tenant_id,
            df=df,
            target_col=target_col,
            task=task,
            tune_hyperparameters=tune_hyperparameters
        )
        # Register numerical baseline in validator for drift detection
        prep = self.ml_pipeline.preprocessors.get(tenant_id)
        if prep and prep.baseline_stats:
            self.schema_validator.set_baseline(tenant_id, prep.baseline_stats)
        return result

    def get_tenant_status(self, tenant_id: str) -> Dict[str, Any]:
        """
        Provides comprehensive health and isolation metrics for a tenant.
        """
        has_schema = self.schema_registry.has_schema(tenant_id)
        schema_def = self.schema_registry.get_raw_definition(tenant_id)
        has_model = self.ml_pipeline.has_model(tenant_id)
        model_meta = self.ml_pipeline.cached_metadata.get(tenant_id) if has_model else None
        pinecone_stats = self.pinecone.get_namespace_stats(tenant_id)

        return {
            "tenant_id": tenant_id,
            "has_dynamic_schema": has_dynamic_schema,
            "schema_definition": schema_def,
            "has_trained_model": has_model,
            "model_metadata": model_meta,
            "is_cold_start": not has_model,
            "pinecone": pinecone_stats
        } if (has_dynamic_schema := has_schema) else {
            "tenant_id": tenant_id,
            "has_dynamic_schema": False,
            "has_trained_model": has_model,
            "model_metadata": model_meta,
            "is_cold_start": not has_model,
            "pinecone": pinecone_stats
        }
