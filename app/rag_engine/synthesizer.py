import logging
import json
from typing import List, Dict, Any, Optional
from app.config import settings

logger = logging.getLogger("HybridAIEngine.Synthesizer")

class ContextSynthesizer:
    """
    Synthesizes retrieved vector search context into an enterprise decision answer.
    Supports:
    - OpenAI ChatCompletion API (if OPENAI_API_KEY is present)
    - Institutional deterministic synthesis engine grounded strictly on retrieved context
    """
    def __init__(self, openai_api_key: Optional[str] = None, model: Optional[str] = None):
        self.openai_key = openai_api_key or settings.OPENAI_API_KEY
        self.model = model or settings.LLM_MODEL

    def synthesize(
        self,
        tenant_id: str,
        query: str,
        retrieved_matches: List[Dict[str, Any]],
        is_cold_start: bool = False
    ) -> Dict[str, Any]:
        if not retrieved_matches:
            return {
                "mode": "rag_fallback" if is_cold_start else "unstructured_rag",
                "answer": f"No institutional knowledge found in tenant '{tenant_id}' vector namespace for this query.",
                "confidence": 0.0,
                "sources": [],
                "is_cold_start": is_cold_start
            }

        # Format retrieved evidence
        evidence_lines = []
        sources = []
        for i, m in enumerate(retrieved_matches, start=1):
            meta = m.get("metadata", {})
            text = meta.get("text") or meta.get("content") or f"Doc #{m.get('id')}"
            score = m.get("score", 0.0)
            category = meta.get("category", "policy")
            evidence_lines.append(f"[{i}] (Score: {score:.2f}, Category: {category}): {text}")
            sources.append({
                "id": m.get("id"),
                "score": score,
                "category": category,
                "snippet": text[:150]
            })

        evidence_str = "\n".join(evidence_lines)

        # Call OpenAI if API key configured
        if self.openai_key:
            try:
                import requests
                sys_prompt = (
                    f"You are the Enterprise AI Decision Assistant for tenant '{tenant_id}'. "
                    f"Answer the query based ONLY on the provided institutional knowledge. "
                    f"Cite evidence numbers [1], [2] when appropriate."
                )
                user_prompt = (
                    f"Tenant Query: {query}\n\n"
                    f"Retrieved Institutional Evidence:\n{evidence_str}\n\n"
                    f"Provide a structured assessment and recommended action."
                )
                headers = {
                    "Authorization": f"Bearer {self.openai_key}",
                    "Content-Type": "application/json"
                }
                body = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.2
                }
                resp = requests.post("https://api.openai.com/v1/chat/completions", json=body, headers=headers, timeout=15)
                if resp.status_code == 200:
                    answer_text = resp.json()["choices"][0]["message"]["content"]
                    return {
                        "mode": "rag_fallback" if is_cold_start else "unstructured_rag",
                        "model": f"Pinecone+{self.model}",
                        "answer": answer_text,
                        "confidence": round(float(retrieved_matches[0].get("score", 0.8)), 4),
                        "sources": sources,
                        "is_cold_start": is_cold_start
                    }
            except Exception as e:
                logger.warning(f"OpenAI completion failed: {e}. Falling back to internal synthesizer.")

        # Grounded institutional deterministic synthesis
        top_match = retrieved_matches[0]
        top_text = top_match.get("metadata", {}).get("text", "Institutional policy guideline")
        top_score = top_match.get("score", 0.85)

        if is_cold_start:
            synthesis = (
                f"[Cold-Start Intelligence Report | Tenant: {tenant_id}]\n"
                f"• Status: Tenant tabular history is below threshold for classical ML. Fallback to Pinecone vector namespace was triggered.\n"
                f"• Evaluated Entity / Query: {query}\n"
                f"• Matched Knowledge Directive [Score: {top_score:.2f}]: \"{top_text}\"\n"
                f"• Recommended Action: Apply institutional compliance rules according to matched policy directive."
            )
        else:
            synthesis = (
                f"[Enterprise Policy Retrieval | Tenant: {tenant_id}]\n"
                f"• Query: \"{query}\"\n"
                f"• Matched Knowledge Directive [Score: {top_score:.2f}]: \"{top_text}\"\n"
                f"• Recommendation: Proceed in alignment with matched organizational guideline [Source: {sources[0]['id']}]."
            )

        return {
            "mode": "rag_fallback" if is_cold_start else "unstructured_rag",
            "model": "Pinecone-Grounded-Synthesizer",
            "answer": synthesis,
            "confidence": round(float(top_score), 4),
            "sources": sources,
            "is_cold_start": is_cold_start
        }
