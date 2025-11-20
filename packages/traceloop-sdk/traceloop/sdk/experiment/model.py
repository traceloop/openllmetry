from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field

EvaluatorVersion = str
EvaluatorSlug = str
EvaluatorDetails = Union[EvaluatorSlug, Tuple[EvaluatorSlug, EvaluatorVersion]]


class TaskResponse(BaseModel):
    """Model for a single task process (row)"""

    task_result: Optional[Dict[str, Any]] = None
    evaluations: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


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


class TaskResult(BaseModel):
    """Model for a single task result"""

    task_input: Dict[str, Any]
    task_output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class GithubContext(BaseModel):
    """Model for GitHub context"""

    github_pr_url: Optional[str] = None
    github_commit_hash: Optional[str] = None
    github_actor: Optional[str] = None

class RunInGithubRequest(BaseModel):
    """Model for bulk GitHub experiment execution request"""

    dataset_slug: Optional[str] = None
    dataset_version: Optional[str] = None
    evaluator_slugs: Optional[List[str]] = None
    task_results: List[TaskResult]
    github_context: Dict[str, Any]
    experiment_metadata: Optional[Dict[str, Any]] = None
    related_ref: Optional[Dict[str, Any]] = None
    aux: Optional[Dict[str, Any]] = None
    stop_on_error: bool = False


class RunInGithubResponse(BaseModel):
    """Model for bulk GitHub experiment execution response"""

    experiment_id: str
    experiment_slug: str
    run_id: str
    status: str
    message: Optional[str] = None
