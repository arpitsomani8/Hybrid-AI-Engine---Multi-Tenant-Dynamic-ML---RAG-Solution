from fastapi import APIRouter, Request, HTTPException, Depends
from app.api.deps import get_tenant_id
from app.api.schemas import ApiResponse, SchemaRegisterRequest

router = APIRouter(prefix="/schemas", tags=["Schemas"])

@router.post("", response_model=ApiResponse)
def register_schema(req: SchemaRegisterRequest, request: Request, tenant_id: str = Depends(get_tenant_id)):
    engine = request.app.state.engine
    try:
        schema_def = req.model_dump()
        engine.register_schema(tenant_id, schema_def)
        return ApiResponse(
            success=True,
            message=f"Dynamic schema successfully registered for tenant '{tenant_id}'",
            data={"tenant_id": tenant_id, "fields_count": len(req.fields)}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("", response_model=ApiResponse)
def get_schema(request: Request, tenant_id: str = Depends(get_tenant_id)):
    engine = request.app.state.engine
    raw_def = engine.schema_registry.get_raw_definition(tenant_id)
    if not raw_def:
        raise HTTPException(status_code=404, detail=f"No dynamic schema found for tenant '{tenant_id}'")
    return ApiResponse(
        success=True,
        message=f"Schema retrieved for tenant '{tenant_id}'",
        data={"tenant_id": tenant_id, "schema": raw_def}
    )
