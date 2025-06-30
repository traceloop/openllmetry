from dataclasses import dataclass
from typing import Dict, Optional, Any
from pydantic import BaseModel

@dataclass
class InputExtractor:
    source: str  # "input" or "output"
    key: Optional[str] = None  # Key to extract from
    use_regex: bool = False  # Whether to use regex pattern
    regex_pattern: Optional[str] = None  # Regex pattern to apply

@dataclass
class ExecuteEvaluatorRequest:
    input_schema_mapping: Dict[str, InputExtractor] 

@dataclass
class EvaluatorResponse:
    result: Dict[str, Any]
    score: float
    reason: str
    metadata: Dict[str, Any]
    raw_result: Dict[str, Any]

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