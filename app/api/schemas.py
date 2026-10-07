from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field

# Dynamic Schema Registration
class SchemaRegisterRequest(BaseModel):
    description: Optional[str] = "Tenant dynamic schema"
    target_column: Optional[str] = None
    task_type: Optional[str] = "binary"
    fields: Dict[str, Any]

# Document Ingestion (Pinecone)
class DocumentUpsertRequest(BaseModel):
    doc_id: str
    text: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class BatchDocumentsUpsertRequest(BaseModel):
    documents: List[DocumentUpsertRequest]

# Tabular Data Ingestion
class TabularBatchIngestRequest(BaseModel):
    records: List[Dict[str, Any]]

# Model Training
class ModelTrainRequest(BaseModel):
    target_column: str
    task_type: Optional[str] = "binary"
    tune_hyperparameters: Optional[bool] = False

# Hybrid Prediction
class PredictRequest(BaseModel):
    payload: Dict[str, Any]

# Response models
class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
