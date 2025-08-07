import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel


class ExperimentRequest(BaseModel):
    """Request for running an experiment with single evaluator on multiple inputs"""
    evaluator_slug: str
    inputs: list[Dict[str, str]]  # List of input dictionaries
    timeout_in_sec: int = 120


class ExperimentRunResult(BaseModel):
    """Result from a single evaluator run within an experiment"""
    input_index: int
    input_data: Dict[str, str]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None


class ExperimentResult(BaseModel):
    """Complete experiment results"""
    evaluator_slug: str
    total_runs: int
    successful_runs: int
    failed_runs: int
    results: list[ExperimentRunResult]
    total_execution_time: float