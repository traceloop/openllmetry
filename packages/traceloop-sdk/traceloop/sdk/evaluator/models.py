from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field


class EvaluationConfig(BaseModel):
    """Configuration for evaluation execution"""
    parameters: Dict[str, Any] = Field(default_factory=dict)
    thresholds: Dict[str, float] = Field(default_factory=dict)
    settings: Dict[str, Any] = Field(default_factory=dict)


class EvaluationInputData(BaseModel):
    """Input data for evaluation execution"""
    content: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    context: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "allow"  # Allow additional fields


class EvaluationRequest(BaseModel):
    """Request payload for starting evaluation execution"""
    input_data: EvaluationInputData
    timeout_ms: int
    config: Optional[EvaluationConfig] = None


class EvaluationResponse(BaseModel):
    """Response from starting evaluation execution"""
    execution_id: str
    stream_url: str
    status: str = "started"


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


class EvaluationResult(BaseModel):
    """Result of evaluation execution"""
    result: Optional[Any] = None
    score: Optional[float] = None
    reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    raw_result: Optional[Dict[str, Any]] = None  # Store the raw evaluation result 