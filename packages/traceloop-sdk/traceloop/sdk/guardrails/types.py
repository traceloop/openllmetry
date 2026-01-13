from typing import Dict, Optional, Any
from pydantic import BaseModel


class InputExtractor(BaseModel):
    source: str  # "input" or "output"
    key: Optional[str] = None  # Key to extract from
    use_regex: bool = False  # Whether to use regex pattern
    regex_pattern: Optional[str] = None  # Regex pattern to apply


class ExecuteEvaluatorRequest(BaseModel):
    input_schema_mapping: Dict[str, InputExtractor]
    evaluator_version: Optional[str] = None
    evaluator_config: Optional[Dict[str, Any]] = None


class OutputSchema(BaseModel):
    reason: Optional[str] = None
    success: bool
