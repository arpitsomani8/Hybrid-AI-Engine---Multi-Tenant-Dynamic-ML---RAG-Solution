from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from fastapi import APIRouter, Request, HTTPException, Depends
from app.api.deps import get_tenant_id
from app.api.schemas import ApiResponse, TabularBatchIngestRequest
from app.config import settings

router = APIRouter(prefix="/data", tags=["Data"])

def get_tenant_data_path(tenant_id: str) -> Path:
    return settings.DATA_STORAGE_DIR / f"{tenant_id}.csv"

@router.post("/ingest", response_model=ApiResponse)
def ingest_data(req: TabularBatchIngestRequest, request: Request, tenant_id: str = Depends(get_tenant_id)):
    if not req.records:
        raise HTTPException(status_code=400, detail="No records provided in ingest request.")

    new_df = pd.DataFrame(req.records)
    file_path = get_tenant_data_path(tenant_id)

    if file_path.exists():
        existing_df = pd.read_csv(file_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df.to_csv(file_path, index=False)

    return ApiResponse(
        success=True,
        message=f"Ingested {len(req.records)} records for tenant '{tenant_id}'",
        data={
            "tenant_id": tenant_id,
            "new_records": len(req.records),
            "total_records": len(combined_df),
            "columns": combined_df.columns.tolist()
        }
    )

@router.get("/preview", response_model=ApiResponse)
def preview_data(request: Request, tenant_id: str = Depends(get_tenant_id), limit: int = 10):
    file_path = get_tenant_data_path(tenant_id)
    if not file_path.exists():
        return ApiResponse(
            success=True,
            message=f"No tabular dataset ingested yet for tenant '{tenant_id}'",
            data={"tenant_id": tenant_id, "total_records": 0, "preview": []}
        )

    df = pd.read_csv(file_path)
    preview_records = df.head(limit).to_dict(orient="records")

    return ApiResponse(
        success=True,
        message=f"Retrieved data preview for tenant '{tenant_id}'",
        data={
            "tenant_id": tenant_id,
            "total_records": len(df),
            "columns": df.columns.tolist(),
            "preview": preview_records
        }
    )
