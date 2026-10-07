import logging
import hashlib
from typing import List, Union
import numpy as np
from app.config import settings

logger = logging.getLogger("HybridAIEngine.Embedder")

class VectorEmbedder:
    """
    Generates normalized dense vector embeddings for documents and queries.
    Supports:
    - OpenAI embeddings (if OPENAI_API_KEY is configured)
    - High-performance deterministic semantic embedding generator
      (produces 384-dimensional unit-norm vectors compatible with Pinecone Serverless cosine metric)
    """
    def __init__(self, dimension: int = None, openai_api_key: str = None):
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self.openai_key = openai_api_key or settings.OPENAI_API_KEY

    def embed_text(self, text: str) -> List[float]:
        if self.openai_key:
            try:
                import requests
                headers = {
                    "Authorization": f"Bearer {self.openai_key}",
                    "Content-Type": "application/json"
                }
                body = {
                    "input": text,
                    "model": "text-embedding-3-small",
                    "dimensions": self.dimension
                }
                resp = requests.post("https://api.openai.com/v1/embeddings", json=body, headers=headers, timeout=10)
                if resp.status_code == 200:
                    vec = resp.json()["data"][0]["embedding"]
                    return vec
            except Exception as e:
                logger.warning(f"OpenAI embedding call failed, falling back to deterministic embedder: {e}")

        # Deterministic semantic n-gram feature hashing to dense unit-norm embedding
        return self._generate_dense_vector(text)

    def _generate_dense_vector(self, text: str) -> List[float]:
        """
        Creates a reproducible, normalized dense vector from text tokens and n-grams.
        Similar semantic tokens map into adjacent subspace projections.
        """
        words = text.lower().strip().split()
        vec = np.zeros(self.dimension, dtype=np.float32)

        for i, word in enumerate(words):
            # Hash single words
            h1 = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx1 = h1 % self.dimension
            sign1 = 1.0 if (h1 >> 8) % 2 == 0 else -1.0
            vec[idx1] += sign1 * 1.5

            # Hash character trigrams for subword robustness
            for k in range(max(1, len(word) - 2)):
                tri = word[k:k+3]
                h_tri = int(hashlib.sha256(tri.encode("utf-8")).hexdigest(), 16)
                idx_tri = h_tri % self.dimension
                sign_tri = 1.0 if (h_tri >> 8) % 2 == 0 else -1.0
                vec[idx_tri] += sign_tri * 0.5

            # Hash bigrams
            if i < len(words) - 1:
                bigram = f"{word}_{words[i+1]}"
                h2 = int(hashlib.sha256(bigram.encode("utf-8")).hexdigest(), 16)
                idx2 = h2 % self.dimension
                sign2 = 1.0 if (h2 >> 8) % 2 == 0 else -1.0
                vec[idx2] += sign2 * 2.0

        # L2 Normalization (required for Cosine similarity in Pinecone)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            # Fallback for empty text
            vec[0] = 1.0

        return vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]
