from ctypes import Union
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel

EvaluatorVersion = str
EvaluatorSlug = str
EvaluatorDetails = Union[EvaluatorSlug, Tuple[EvaluatorSlug, EvaluatorVersion]]


class TaskResponse(BaseModel):
    """Model for a single task process (row)"""
    task_result: Dict[str, Any]
    evaluations: Dict[str, Any]

class InitExperimentRequest(BaseModel):
    """Model for initializing an experiment"""
    slug: str
    dataset_slug: Optional[str] = None
    dataset_version: Optional[str] = None
    evaluator_slugs: Optional[List[str]] = None
    experiment_metadata: Optional[Dict[str, Any]] = None
    experiment_run_metadata: Optional[Dict[str, Any]] = None

class ExperimentResponse(BaseModel):
    """Model for experiment response"""
    id: str
    slug: str
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

class ExperimentRunResponse(BaseModel):
    """Model for experiment run response"""
    id: str
    metadata: Optional[Dict[str, Any]] = None
    dataset_id: Optional[str] = None
    dataset_version: Optional[str] = None
    evaluator_ids: Optional[List[str]] = None
    created_at: datetime
    updated_at: datetime

class ExperimentInitResponse(BaseModel):
    """Model for experiment and run response"""
    experiment: ExperimentResponse
    run: ExperimentRunResponse

class CreateTaskRequest(BaseModel):
    """Model for create task request"""
    input: Dict[str, Any]
    output: Dict[str, Any]

class CreateTaskResponse(BaseModel):
    """Model for create task response"""
    id: str



