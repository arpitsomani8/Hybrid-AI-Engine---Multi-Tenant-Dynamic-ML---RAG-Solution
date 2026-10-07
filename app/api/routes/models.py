import pandas as pd
from fastapi import APIRouter, Request, HTTPException, Depends
from app.api.deps import get_tenant_id
from app.api.schemas import ApiResponse, ModelTrainRequest
from app.api.routes.data import get_tenant_data_path

router = APIRouter(prefix="/models", tags=["Models"])

@router.post("/train", response_model=ApiResponse)
def train_model(req: ModelTrainRequest, request: Request, tenant_id: str = Depends(get_tenant_id)):
    engine = request.app.state.engine
    data_path = get_tenant_data_path(tenant_id)

    if not data_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"No dataset found for tenant '{tenant_id}'. Ingest data first."
        )

    df = pd.read_csv(data_path)
    if len(df) < engine.ml_pipeline.min_samples:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Insufficient dataset size ({len(df)} rows) for tenant '{tenant_id}'. "
                f"Requires at least {engine.ml_pipeline.min_samples} rows to trigger automated classical ML training."
            )
        )

    try:
        result = engine.train_model(
            tenant_id=tenant_id,
            df=df,
            target_col=req.target_column,
            task=req.task_type or "binary",
            tune_hyperparameters=bool(req.tune_hyperparameters)
        )
        return ApiResponse(
            success=True,
            message=f"Model successfully trained for tenant '{tenant_id}'",
            data={
                "tenant_id": tenant_id,
                "task": result.task,
                "num_samples": result.num_samples,
                "metrics": result.metrics,
                "feature_importances": result.feature_importances,
                "best_iteration": result.best_iteration
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/metrics", response_model=ApiResponse)
def get_metrics(request: Request, tenant_id: str = Depends(get_tenant_id)):
    engine = request.app.state.engine
    if not engine.ml_pipeline.has_model(tenant_id):
        raise HTTPException(status_code=404, detail=f"No trained model found for tenant '{tenant_id}'")

    metadata = engine.ml_pipeline.cached_metadata.get(tenant_id)
    return ApiResponse(
        success=True,
        message=f"Metrics retrieved for tenant '{tenant_id}'",
        data=metadata
    )
