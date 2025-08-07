from traceloop.sdk.evaluators.evaluator import Evaluator
from traceloop.sdk.evaluators.model import (
    InputExtractor,
    InputSchemaMapping,
    ExecuteEvaluatorRequest,
    ExecuteEvaluatorResponse,
    EvaluatorResult,
    StreamEvent
)
from traceloop.sdk.evaluators.stream_client import SSEResultClient


def run_evaluator(evaluator_slug: str, input_schema_mapping: dict, **kwargs):
    """
    Execute an evaluator with input schema mapping
    
    Args:
        evaluator_slug: Slug of the evaluator to execute
        input_schema_mapping: Dict mapping field names to source fields
        **kwargs: Additional arguments (callback, wait_for_result, timeout)
    
    Returns:
        ExecuteEvaluatorResponse or result data
    """
    return Evaluator.run(
        evaluator_slug=evaluator_slug,
        input_schema_mapping=input_schema_mapping,
        **kwargs
    )


async def run_evaluator_async(evaluator_slug: str, input_schema_mapping: dict, **kwargs):
    """
    Execute an evaluator asynchronously
    
    Args:
        evaluator_slug: Slug of the evaluator to execute
        input_schema_mapping: Dict mapping field names to source fields
        **kwargs: Additional arguments (callback, timeout)
    
    Returns:
        ExecuteEvaluatorResponse
    """
    return await Evaluator.run_async(
        evaluator_slug=evaluator_slug,
        input_schema_mapping=input_schema_mapping,
        **kwargs
    )


__all__ = [
    "Evaluator",
    "InputExtractor", 
    "InputSchemaMapping",
    "ExecuteEvaluatorRequest",
    "ExecuteEvaluatorResponse", 
    "EvaluatorResult",
    "StreamEvent",
    "SSEResultClient",
    "run_evaluator",
    "run_evaluator_async"
]