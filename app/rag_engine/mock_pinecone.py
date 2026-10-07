import logging
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger("HybridAIEngine.MockPinecone")

class MockIndex:
    """
    Mock Pinecone Index that implements exact namespace partitioning
    and cosine similarity queries in memory.
    """
    def __init__(self, name: str, dimension: int):
        self.name = name
        self.dimension = dimension
        # Dict[namespace, List[dict]]
        self.namespaces: Dict[str, List[Dict[str, Any]]] = {}

    def upsert(self, vectors: List[Dict[str, Any]], namespace: str = "default") -> Dict[str, Any]:
        if namespace not in self.namespaces:
            self.namespaces[namespace] = []

        upserted_count = 0
        existing = {v["id"]: i for i, v in enumerate(self.namespaces[namespace])}

        for vec_record in vectors:
            v_id = vec_record["id"]
            values = np.array(vec_record["values"], dtype=np.float32)
            meta = vec_record.get("metadata", {})

            item = {"id": v_id, "values": values, "metadata": meta}
            if v_id in existing:
                self.namespaces[namespace][existing[v_id]] = item
            else:
                self.namespaces[namespace].append(item)
                existing[v_id] = len(self.namespaces[namespace]) - 1
            upserted_count += 1

        logger.info(f"[MockPinecone] Upserted {upserted_count} vectors into namespace '{namespace}'")
        return {"upserted_count": upserted_count}

    def query(
        self,
        vector: List[float],
        top_k: int = 5,
        namespace: str = "default",
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        records = self.namespaces.get(namespace, [])
        if not records:
            return {"matches": [], "namespace": namespace}

        q_vec = np.array(vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        scored_matches = []
        for r in records:
            r_vec = r["values"]
            r_norm = np.linalg.norm(r_vec)
            if r_norm > 0:
                sim = float(np.dot(q_vec, r_vec) / (q_norm * r_norm))
            else:
                sim = 0.0

            scored_matches.append({
                "id": r["id"],
                "score": round(sim, 4),
                "metadata": r["metadata"] if include_metadata else {}
            })

        scored_matches.sort(key=lambda x: x["score"], reverse=True)
        return {"matches": scored_matches[:top_k], "namespace": namespace}

    def describe_index_stats(self) -> Dict[str, Any]:
        ns_stats = {}
        total = 0
        for ns, docs in self.namespaces.items():
            cnt = len(docs)
            ns_stats[ns] = {"vector_count": cnt}
            total += cnt
        return {
            "namespaces": ns_stats,
            "dimension": self.dimension,
            "total_vector_count": total
        }

    def delete(self, ids: Optional[List[str]] = None, delete_all: bool = False, namespace: str = "default"):
        if namespace not in self.namespaces:
            return
        if delete_all:
            self.namespaces[namespace] = []
        elif ids:
            id_set = set(ids)
            self.namespaces[namespace] = [r for r in self.namespaces[namespace] if r["id"] not in id_set]
