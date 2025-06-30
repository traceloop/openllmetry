from typing import Dict, Optional, Any
from pydantic import BaseModel, Field

class InputExtractor(BaseModel):
    source: str  # "input" or "output"
    key: Optional[str] = None  # Key to extract from
    use_regex: bool = False  # Whether to use regex pattern
    regex_pattern: Optional[str] = None  # Regex pattern to apply

class ExecuteEvaluatorRequest(BaseModel):
    input_schema_mapping: Dict[str, InputExtractor] 

class EvaluatorResult(BaseModel):
    reason: str
    success: bool = Field(alias="pass")

class EvaluatorResponse(BaseModel):
    result: EvaluatorResult

class StreamEventData(BaseModel):
    """Data payload in stream events"""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: Optional[float] = None
    status: Optional[str] = None

    class Config:
        extra = "allow"

class StreamEvent(BaseModel):
    """Stream event structure"""
    type: str
    data: StreamEventData