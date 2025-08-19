from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class ExperimentContextData(BaseModel):
    """Pydantic model for experiment context data"""
    experiment_id: str
    experiment_slug: Optional[str] = None
    experiment_run_data: Dict[str, Any]
    run_id: str
    run_name: str

class RunContextData(BaseModel):
    """Pydantic model for run context data"""
    experiment_id: str
    run_id: str
    task_id: str
    task_input: Any
    task_output: Any
    dataset_ids: List[str]
    evaluator_slug: str
    evaluator_version: Optional[str] = None
    
class CreateExperimentRequest(BaseModel):
    """Pydantic model for create experiment request"""
    slug: str
    dataset_slugs: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class ExperimentResponse(BaseModel):
    """Pydantic model for experiment response"""
    id: str
    slug: str
    metadata: Optional[Dict[str, Any]] = None
    dataset_ids: Optional[List[str]] = None
    created_at: datetime
    updated_at: datetime
    

