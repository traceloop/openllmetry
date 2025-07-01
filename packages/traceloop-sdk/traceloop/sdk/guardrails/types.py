from typing import Dict, Optional
from pydantic import BaseModel, Field


class InputExtractor(BaseModel):
    source: str  # "input" or "output"
    key: Optional[str] = None  # Key to extract from
    use_regex: bool = False  # Whether to use regex pattern
    regex_pattern: Optional[str] = None  # Regex pattern to apply


class ExecuteEvaluatorRequest(BaseModel):
    input_schema_mapping: Dict[str, InputExtractor]


class OutputSchema(BaseModel):
    reason: str
    success: bool = Field(alias="pass")
