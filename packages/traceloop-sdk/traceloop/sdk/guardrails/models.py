from enum import Enum
from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field


class GuardrailAction(Enum):
    PASS = "pass"
    BLOCK = "block"
    RETRY = "retry"


class GuardrailConfig(BaseModel):
    """Configuration for guardrail execution"""
    parameters: Dict[str, Any] = Field(default_factory=dict)
    thresholds: Dict[str, float] = Field(default_factory=dict)
    settings: Dict[str, Any] = Field(default_factory=dict)


class GuardrailInputData(BaseModel):
    """Input data for guardrail execution"""
    content: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    context: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "allow"  # Allow additional fields


# Note: Execution models are now handled by the evaluator layer


class GuardrailResult(BaseModel):
    """Result of guardrail execution"""
    action: GuardrailAction
    result: Optional[Any] = None
    reason: Optional[str] = None
    score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def pass_through(self) -> bool:
        return self.action == GuardrailAction.PASS
        
    @property
    def blocked(self) -> bool:
        return self.action == GuardrailAction.BLOCK
        
    @property
    def retry_required(self) -> bool:
        return self.action == GuardrailAction.RETRY 