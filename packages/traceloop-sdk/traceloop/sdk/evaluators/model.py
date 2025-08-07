import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class InputExtractor(BaseModel):
    source: str


class InputSchemaMapping(BaseModel):
    """Map of field names to input extractors"""
    __root__: Dict[str, InputExtractor]


class ExecuteEvaluatorRequest(BaseModel):
    input_schema_mapping: InputSchemaMapping = Field(..., alias="input_schema_mapping")


class ExecuteEvaluatorResponse(BaseModel):
    """Response from execute API matching actual structure"""
    execution_id: str
    stream_url: str


class EvaluatorResult(BaseModel):
    """Final result from SSE stream"""
    execution_id: str
    status: str  # completed, failed, running
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    completed_at: Optional[datetime.datetime] = None


class StreamEvent(BaseModel):
    """Individual event from SSE stream"""
    event_type: str  # progress, result, error
    data: Dict[str, Any]
    timestamp: datetime.datetime