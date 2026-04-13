import datetime
from typing import Dict, Any, Optional, TypeVar, Type
from pydantic import BaseModel, Field, RootModel

T = TypeVar('T', bound=BaseModel)


class InputExtractor(BaseModel):
    source: str


class InputSchemaMapping(RootModel[Dict[str, InputExtractor]]):
    """Map of field names to input extractors"""

    root: Dict[str, InputExtractor]


class ExecuteEvaluatorInExperimentRequest(BaseModel):
    input_schema_mapping: InputSchemaMapping
    evaluator_version: Optional[str] = None
    evaluator_config: Optional[Dict[str, Any]] = None
    evaluator_slug: str
    task_id: str
    experiment_id: str
    experiment_run_id: str

class ExecuteEvaluatorRequest(BaseModel):
    input: Dict[str, Any]
    evaluator_version: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

class ExecuteEvaluatorResponse(BaseModel):
    """Response from execute API matching actual structure"""

    execution_id: str
    stream_url: str


class StreamEvent(BaseModel):
    """Individual event from SSE stream"""

    event_type: str  # progress, result, error
    data: Dict[str, Any]
    timestamp: datetime.datetime

class EvaluatorExecutionResult(BaseModel):
    evaluator_result: Dict[str, Any]

class ExecutionResponse(BaseModel):
    """Complete response structure for evaluator execution"""

    execution_id: str
    result: EvaluatorExecutionResult

    def typed_result(self, model: Type[T]) -> T:
        """Parse result into a typed Pydantic model.

        Args:
            model: The Pydantic model class to parse the result into

        Returns:
            An instance of the provided model class

        Example:
            from traceloop.sdk.generated.evaluators import PIIDetectorResponse
            result = await evaluator.run_experiment_evaluator(...)
            pii = result.typed_result(PIIDetectorResponse)
            print(pii.has_pii)  # IDE autocomplete works!
        """
        return model(**self.result.evaluator_result)


class GuardrailResponse(BaseModel):
    """Response from the guardrails execute route."""

    result: Dict[str, Any]
    passed: bool = Field(alias="pass")
