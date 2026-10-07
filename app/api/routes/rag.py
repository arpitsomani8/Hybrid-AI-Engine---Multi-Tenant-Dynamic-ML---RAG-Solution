from typing import Dict, Any, List
from fastapi import APIRouter, Request, HTTPException, Depends
from app.api.deps import get_tenant_id
from app.api.schemas import ApiResponse, DocumentUpsertRequest, BatchDocumentsUpsertRequest

router = APIRouter(prefix="/rag", tags=["RAG (Pinecone)"])

@router.post("/upsert", response_model=ApiResponse)
def upsert_doc(req: DocumentUpsertRequest, request: Request, tenant_id: str = Depends(get_tenant_id)):
    engine = request.app.state.engine
    try:
        engine.index_document(
            tenant_id=tenant_id,
            doc_id=req.doc_id,
            text=req.text,
            metadata=req.metadata
        )
        return ApiResponse(
            success=True,
            message=f"Document '{req.doc_id}' indexed into Pinecone namespace '{tenant_id}'",
            data={"tenant_id": tenant_id, "doc_id": req.doc_id}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-upsert", response_model=ApiResponse)
def batch_upsert(req: BatchDocumentsUpsertRequest, request: Request, tenant_id: str = Depends(get_tenant_id)):
    engine = request.app.state.engine
    try:
        for doc in req.documents:
            engine.index_document(
                tenant_id=tenant_id,
                doc_id=doc.doc_id,
                text=doc.text,
                metadata=doc.metadata
            )
        return ApiResponse(
            success=True,
            message=f"Successfully indexed {len(req.documents)} documents into namespace '{tenant_id}'",
            data={"tenant_id": tenant_id, "indexed_count": len(req.documents)}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats", response_model=ApiResponse)
def get_stats(request: Request, tenant_id: str = Depends(get_tenant_id)):
    engine = request.app.state.engine
    stats = engine.pinecone.get_namespace_stats(tenant_id)
    return ApiResponse(
        success=True,
        message=f"Pinecone namespace stats for tenant '{tenant_id}'",
        data=stats
    )

@router.post("/search", response_model=ApiResponse)
def search_vectors(query: str, request: Request, tenant_id: str = Depends(get_tenant_id), top_k: int = 3):
    engine = request.app.state.engine
    q_vec = engine.embedder.embed_text(query)
    matches = engine.pinecone.query_similarity(tenant_id, q_vec, top_k=top_k)
    return ApiResponse(
        success=True,
        message=f"Retrieved {len(matches)} vector matches from namespace '{tenant_id}'",
        data={"tenant_id": tenant_id, "query": query, "matches": matches}
    )
