from typing import Dict, Any
from fastapi import APIRouter, Request, HTTPException, Depends
from app.api.deps import get_tenant_id
from app.api.schemas import ApiResponse, PredictRequest

router = APIRouter(prefix="/predict", tags=["Hybrid Inference"])

@router.post("", response_model=ApiResponse)
def predict(req: PredictRequest, request: Request, tenant_id: str = Depends(get_tenant_id)):
    engine = request.app.state.engine
    try:
        result = engine.process_query(tenant_id, req.payload)
        return ApiResponse(
            success=True,
            message=f"Query evaluated via route '{result.get('route', 'unknown')}'",
            data=result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
