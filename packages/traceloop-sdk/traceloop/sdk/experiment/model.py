from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional


@dataclass
class EvaluatorRun:
    """Single evaluator execution data"""
    evaluator_id: str
    evaluator_name: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    runtime_ms: float
    started_at: datetime
    completed_at: datetime
    success: bool
    error_message: Optional[str] = None


@dataclass
class EvalRun:
    """Single evaluation run (row + evaluators)"""
    eval_run_id: str
    dataset_id: str
    dataset_name: str
    dataset_row_id: str
    variant_name: str
    started_at: datetime
    completed_at: Optional[datetime]
    evaluator_runs: List[EvaluatorRun]


@dataclass
class ExperimentData:
    """Complete experiment data for API logging"""
    experiment_id: str
    experiment_name: str
    created_at: datetime
    completed_at: Optional[datetime]
    duration_ms: float
    eval_runs: List[EvalRun]
    total_evaluator_runs: int
    success_rate: float
    metadata: Dict[str, Any]