from traceloop.sdk.evaluators.evaluator import Evaluator
from traceloop.sdk.evaluators.model import (
    InputExtractor,
    InputSchemaMapping,
    ExecuteEvaluatorRequest,
    ExecuteEvaluatorResponse,
    StreamEvent,
)
from traceloop.sdk.evaluators.stream_client import SSEClient


def run_evaluator(evaluator_slug: str, input_schema_mapping: dict, timeout: int = 500):
    """
    Execute an evaluator with input schema mapping and wait for result
    
    Args:
        evaluator_slug: Slug of the evaluator to execute
        input_schema_mapping: Dict mapping field names to source fields
        timeout: Timeout in seconds for execution
    
    Returns:
        Dict[str, Any]: The evaluation result from SSE stream
    """
    return Evaluator.run(
        evaluator_slug=evaluator_slug,
        input_schema_mapping=input_schema_mapping,
        timeout_in_sec=timeout
    )


def create_evaluator(slug: str, evaluator_slug: str, name: str, description: str):
    """
    Create a simple evaluator data structure for testing
    
    Args:
        slug: Evaluator slug identifier
        evaluator_slug: Evaluator type slug
        name: Display name
        description: Description of evaluator
    
    Returns:
        Simple object with evaluator properties
    """
    class SimpleEvaluator:
        def __init__(self, slug, evaluator_slug, name, description):
            self.slug = slug
            self.evaluator_slug = evaluator_slug
            self.name = name
            self.description = description
    
    return SimpleEvaluator(slug, evaluator_slug, name, description)


__all__ = [
    "Evaluator",
    "InputExtractor", 
    "InputSchemaMapping",
    "ExecuteEvaluatorRequest",
    "ExecuteEvaluatorResponse", 
    "StreamEvent",
    "SSEClient",
    "run_evaluator",
    "create_evaluator"
]