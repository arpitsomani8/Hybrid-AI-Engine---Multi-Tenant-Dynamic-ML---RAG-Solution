import pytest
from app.rag_engine import VectorEmbedder, MockIndex, ContextSynthesizer

def test_vector_embedder_dimension():
    embedder = VectorEmbedder(dimension=384)
    v1 = embedder.embed_text("High transaction volume above 5000 requires verification")
    assert len(v1) == 384
    # Check L2 normalization
    import numpy as np
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-4

def test_pinecone_namespace_isolation():
    mock_idx = MockIndex(name="test-index", dimension=384)
    embedder = VectorEmbedder(dimension=384)

    # Upsert doc into tenant_a namespace
    v_a = embedder.embed_text("Tenant A compliance policy: Accounts under 30 days are limited.")
    mock_idx.upsert(
        vectors=[{"id": "doc_a", "values": v_a, "metadata": {"category": "policy_a", "text": "Policy A"}}],
        namespace="tenant_a"
    )

    # Upsert doc into tenant_b namespace
    v_b = embedder.embed_text("Tenant B refund policy: 90-day return window.")
    mock_idx.upsert(
        vectors=[{"id": "doc_b", "values": v_b, "metadata": {"category": "policy_b", "text": "Policy B"}}],
        namespace="tenant_b"
    )

    # Query tenant_a namespace
    q_vec = embedder.embed_text("refund return window")
    res_a = mock_idx.query(vector=q_vec, top_k=5, namespace="tenant_a")
    assert len(res_a["matches"]) == 1
    assert res_a["matches"][0]["id"] == "doc_a"

    # Query tenant_b namespace
    res_b = mock_idx.query(vector=q_vec, top_k=5, namespace="tenant_b")
    assert len(res_b["matches"]) == 1
    assert res_b["matches"][0]["id"] == "doc_b"

def test_synthesizer_cold_start():
    synthesizer = ContextSynthesizer()
    matches = [{
        "id": "rule_01",
        "score": 0.88,
        "metadata": {"text": "Accounts with velocity > 5000 require manual review.", "category": "fraud"}
    }]
    res = synthesizer.synthesize("fintech_corp", "Check account 123", matches, is_cold_start=True)
    assert res["mode"] == "rag_fallback"
    assert res["is_cold_start"] is True
    assert "manual review" in res["answer"].lower()
