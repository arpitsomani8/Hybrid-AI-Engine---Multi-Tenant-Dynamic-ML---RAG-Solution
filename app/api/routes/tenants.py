from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Request
from app.api.deps import get_tenant_id
from app.api.schemas import ApiResponse

router = APIRouter(prefix="/tenants", tags=["Tenants"])

@router.get("", response_model=ApiResponse)
def list_tenants(request: Request):
    engine = request.app.state.engine
    schema_tenants = set(engine.schema_registry.list_tenants())
    model_tenants = set(engine.model_registry.list_tenants_with_models())
    all_tenants = sorted(list(schema_tenants.union(model_tenants)))

    tenants_overview = []
    for t_id in all_tenants:
        tenants_overview.append(engine.get_tenant_status(t_id))

    return ApiResponse(
        success=True,
        message=f"Found {len(tenants_overview)} active tenants",
        data={"tenants": tenants_overview}
    )

@router.get("/{tenant_id}/status", response_model=ApiResponse)
def get_status(tenant_id: str, request: Request):
    engine = request.app.state.engine
    status = engine.get_tenant_status(tenant_id)
    return ApiResponse(
        success=True,
        message=f"Status retrieved for tenant '{tenant_id}'",
        data=status
    )
