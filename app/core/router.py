import logging
import json
from enum import Enum
from typing import Dict, Any, Tuple

logger = logging.getLogger("HybridAIEngine.Router")

class ExecutionRoute(str, Enum):
    CLASSICAL_ML = "classical_ml"
    COLD_START_RAG = "cold_start_rag"
    UNSTRUCTURED_RAG = "unstructured_rag"

class IntelligentRouter:
    """
    Evaluates inbound tenant query payloads and determines whether to route to:
    1. Classical Tabular ML (LightGBM)
    2. Cold-Start RAG Fallback (Pinecone Namespace + LLM)
    3. Unstructured Natural Language RAG (Pinecone Namespace + LLM)
    """
    @staticmethod
    def classify_request(
        tenant_id: str,
        payload: Dict[str, Any],
        has_trained_model: bool
    ) -> Tuple[ExecutionRoute, Dict[str, Any]]:
        # 1. Check for explicit or implicit unstructured natural language query
        if "natural_language_query" in payload:
            query_str = payload["natural_language_query"]
            return ExecutionRoute.UNSTRUCTURED_RAG, {"query": str(query_str)}
        
        if "query" in payload and isinstance(payload["query"], str) and len(payload) == 1:
            return ExecutionRoute.UNSTRUCTURED_RAG, {"query": payload["query"]}

        # 2. Structured Tabular Payload
        if has_trained_model:
            return ExecutionRoute.CLASSICAL_ML, payload

        # 3. Cold Start: No trained model available
        # Synthesize tabular payload into a semantic representation for vector retrieval
        formatted_features = ", ".join([f"{k}: {v}" for k, v in payload.items()])
        query_repr = f"Evaluation criteria for tenant entity with attributes: {formatted_features}"
        
        return ExecutionRoute.COLD_START_RAG, {
            "query": query_repr,
            "raw_payload": payload
        }
