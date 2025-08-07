import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, RootModel


class InputExtractor(BaseModel):
    source: str


class InputSchemaMapping(RootModel[Dict[str, InputExtractor]]):
    """Map of field names to input extractors"""
    root: Dict[str, InputExtractor]


class ExecuteEvaluatorRequest(BaseModel):
    input_schema_mapping: InputSchemaMapping
    source: str


class ExecuteEvaluatorResponse(BaseModel):
    """Response from execute API matching actual structure"""
    execution_id: str
    stream_url: str


class StreamEvent(BaseModel):
    """Individual event from SSE stream"""
    event_type: str  # progress, result, error
    data: Dict[str, Any]
    timestamp: datetime.datetime


class ExecutionResponse(BaseModel):
    """Complete response structure for evaluator execution"""
    execution_id: str
    result: Dict[str, Any]