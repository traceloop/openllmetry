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
    id: str
    name: str
    data: Dict[str, Any]
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: float


