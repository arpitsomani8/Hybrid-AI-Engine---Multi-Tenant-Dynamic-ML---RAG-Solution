import logging
import time
from typing import Dict, Any, List, Optional
from app.config import settings
from .mock_pinecone import MockIndex

logger = logging.getLogger("HybridAIEngine.PineconeClient")

class PineconeManager:
    """
    Production Pinecone Serverless client wrapper.
    Ensures:
    - Auto-provisioning of the free-tier Serverless index (AWS us-east-1)
    - Strict tenant isolation via Pinecone namespaces
    - Graceful fallback to MockIndex if API key is unset or network fails
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        dimension: Optional[int] = None,
        cloud: Optional[str] = None,
        region: Optional[str] = None,
    ):
        self.api_key = api_key or settings.PINECONE_API_KEY
        self.index_name = index_name or settings.PINECONE_INDEX_NAME
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self.cloud = cloud or settings.PINECONE_CLOUD
        self.region = region or settings.PINECONE_REGION
        self.is_live = False
        self.index = None
        self._init_client()

    def _init_client(self):
        if not self.api_key:
            logger.warning("No PINECONE_API_KEY provided. Operating in Mock mode.")
            self.index = MockIndex(name=self.index_name, dimension=self.dimension)
            return

        try:
            from pinecone import Pinecone, ServerlessSpec
            pc = Pinecone(api_key=self.api_key)

            existing_indexes = [idx.name for idx in pc.list_indexes()]
            if self.index_name not in existing_indexes:
                logger.info(
                    f"Creating Pinecone Serverless index '{self.index_name}' "
                    f"(dim={self.dimension}, cloud={self.cloud}, region={self.region})..."
                )
                spec = ServerlessSpec(cloud=self.cloud, region=self.region)
                pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=spec
                )
                # Wait for index readiness
                for _ in range(30):
                    desc = pc.describe_index(self.index_name)
                    if desc.status.get("ready", False):
                        break
                    time.sleep(2)
                logger.info(f"Pinecone Serverless index '{self.index_name}' is ready.")

            self.index = pc.Index(self.index_name)
            self.is_live = True
            logger.info(f"Connected to LIVE Pinecone Serverless Index: '{self.index_name}'")

        except Exception as e:
            logger.error(f"Failed to connect to Pinecone: {e}. Falling back to MockIndex.")
            self.index = MockIndex(name=self.index_name, dimension=self.dimension)
            self.is_live = False

    def upsert_vectors(self, tenant_id: str, vectors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Upserts vector records strictly into the tenant's namespace.
        Each item in vectors: {"id": str, "values": List[float], "metadata": Dict[str, Any]}
        """
        if not self.index:
            raise RuntimeError("Pinecone index is not initialized.")
        
        logger.info(f"Upserting {len(vectors)} vectors into namespace: '{tenant_id}' (is_live={self.is_live})")
        return self.index.upsert(vectors=vectors, namespace=tenant_id)

    def query_similarity(
        self,
        tenant_id: str,
        query_vector: List[float],
        top_k: int = 3,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Queries vectors strictly isolated to the tenant's namespace.
        """
        if not self.index:
            return []

        res = self.index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=tenant_id,
            include_metadata=include_metadata
        )

        matches = []
        raw_matches = res.get("matches", []) if isinstance(res, dict) else res.matches
        for m in raw_matches:
            if isinstance(m, dict):
                matches.append({
                    "id": m.get("id"),
                    "score": round(float(m.get("score", 0.0)), 4),
                    "metadata": m.get("metadata", {})
                })
            else:
                matches.append({
                    "id": m.id,
                    "score": round(float(m.score or 0.0), 4),
                    "metadata": m.metadata or {}
                })
        return matches

    def get_namespace_stats(self, tenant_id: str) -> Dict[str, Any]:
        try:
            stats = self.index.describe_index_stats()
            if hasattr(stats, "to_dict"):
                stats_dict = stats.to_dict()
            elif isinstance(stats, dict):
                stats_dict = stats
            else:
                stats_dict = {}

            namespaces = stats_dict.get("namespaces", {})
            tenant_stats = namespaces.get(tenant_id, {"vector_count": 0})
            return {
                "tenant_id": tenant_id,
                "is_live_pinecone": self.is_live,
                "vector_count": tenant_stats.get("vector_count", 0),
                "total_index_vectors": stats_dict.get("total_vector_count", 0)
            }
        except Exception as e:
            logger.warning(f"Could not retrieve namespace stats for {tenant_id}: {e}")
            return {"tenant_id": tenant_id, "is_live_pinecone": self.is_live, "vector_count": 0}
