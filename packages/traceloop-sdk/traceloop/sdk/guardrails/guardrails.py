from typing import Dict, Any, Optional, Callable, TypeVar, ParamSpec, Union, Awaitable
from traceloop.sdk.evaluator.config import EvaluatorDetails
from traceloop.sdk.evaluator.evaluator import Evaluator
from .types import OutputSchema
import httpx
import asyncio
from functools import wraps


P = ParamSpec('P')
R = TypeVar('R')

# Type alias for evaluator specification - can be either a slug string or EvaluatorDetails
EvaluatorSpec = Union[str, EvaluatorDetails]


def guardrail(
    evaluator: EvaluatorSpec,
    on_evaluation_complete: Optional[Callable[[OutputSchema, Any], Any]] = None
) -> Callable[[Callable[P, Awaitable[Dict[str, Any]]]], Callable[P, Awaitable[Dict[str, Any]]]]:
    """
    Decorator that executes a guardrails evaluator on the decorated function's output.

    Args:
        evaluator: Either a slug string or an EvaluatorDetails object (with slug, version, config)
        on_evaluation_complete: Optional callback function that receives (evaluator_result, original_result)
                                and returns the final result. If not provided, returns original result on
                                success or an error message on failure.

    Returns:
        Result from on_evaluation_complete callback if provided, otherwise original result or error message
    """
    # Extract evaluator details as tuple (slug, version, config, required_fields) - same pattern as experiments
    slug: str
    evaluator_version: Optional[str]
    evaluator_config: Optional[Dict[str, Any]]
    required_input_fields: Optional[list[str]]

    if isinstance(evaluator, str):
        # Simple string slug - use default field mapping
        slug = evaluator
        evaluator_version = None
        evaluator_config = None
        required_input_fields = None
    elif isinstance(evaluator, EvaluatorDetails):
        # EvaluatorDetails object with config
        slug = evaluator.slug
        evaluator_version = evaluator.version
        evaluator_config = evaluator.config
        required_input_fields = evaluator.required_input_fields
    else:
        raise ValueError(f"evaluator must be str or EvaluatorDetails, got {type(evaluator)}")

    def decorator(func: Callable[P, R]) -> Callable[P, Dict[str, Any]]:
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Dict[str, Any]:
            # Execute the original function - should return a dict with fields matching required_input_fields
            result = await func(*args, **kwargs)  # type: ignore[misc]

            # Ensure we have a dict
            if not isinstance(result, dict):
                raise ValueError(
                    f"Function {func.__name__} must return a dict, got {type(result)}. "
                    f"Required fields: {required_input_fields or 'unknown'}"
                )

            original_result: Dict[str, Any] = result

            # Build evaluator_data based on required_input_fields or use all fields from result
            evaluator_data: Dict[str, str] = {}
            if required_input_fields:
                # Use only the required fields from the function result
                for field in required_input_fields:
                    if field not in original_result:
                        raise ValueError(
                            f"Function {func.__name__} must return dict with field '{field}'. "
                            f"Got: {list(original_result.keys())}"
                        )
                    evaluator_data[field] = str(original_result[field])
            else:
                # No required fields specified, use all fields from result
                for field, value in original_result.items():
                    evaluator_data[field] = str(value)

            try:
                from traceloop.sdk import Traceloop
                client_instance = Traceloop.get()
            except Exception as e:
                print(f"Warning: Could not get Traceloop client: {e}")
                return original_result

            evaluator_result = await client_instance.guardrails.execute_evaluator(
                slug, evaluator_data, evaluator_version, evaluator_config
            )

            # Use callback if provided, otherwise use default behavior
            if on_evaluation_complete:
                callback_result = on_evaluation_complete(evaluator_result, original_result)
                # Callback should return a dict, but we can't enforce this at compile time
                return callback_result  # type: ignore[no-any-return]
            else:
                # Default behavior: return original result (dict) regardless of pass/fail
                # Users should use callback for custom behavior
                return original_result

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> Dict[str, Any]:

            # Execute the original function - should return a dict with fields matching required_input_fields
            result = func(*args, **kwargs)

            # Ensure we have a dict
            if not isinstance(result, dict):
                raise ValueError(
                    f"Function {func.__name__} must return a dict, got {type(result)}. "
                    f"Required fields: {required_input_fields or 'unknown'}"
                )

            original_result: Dict[str, Any] = result

            # Build evaluator_data based on required_input_fields or use all fields from result
            evaluator_data: Dict[str, str] = {}
            if required_input_fields:
                # Use only the required fields from the function result
                for field in required_input_fields:
                    if field not in original_result:
                        raise ValueError(
                            f"Function {func.__name__} must return dict with field '{field}'. "
                            f"Got: {list(original_result.keys())}"
                        )
                    evaluator_data[field] = str(original_result[field])
            else:
                # No required fields specified, use all fields from result
                for field, value in original_result.items():
                    evaluator_data[field] = str(value)

            # Get client instance
            try:
                from traceloop.sdk import Traceloop
                client_instance = Traceloop.get()
            except Exception as e:
                print(f"Warning: Could not get Traceloop client: {e}")
                return original_result

            loop = asyncio.get_event_loop()
            evaluator_result = loop.run_until_complete(
                client_instance.guardrails.execute_evaluator(
                    slug, evaluator_data, evaluator_version, evaluator_config
                )
            )

            # Use callback if provided, otherwise use default behavior
            if on_evaluation_complete:
                callback_result = on_evaluation_complete(evaluator_result, original_result)
                # Callback should return a dict, but we can't enforce this at compile time
                return callback_result  # type: ignore[no-any-return]
            else:
                # Default behavior: return original result (dict) regardless of pass/fail
                # Users should use callback for custom behavior
                return original_result

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        else:
            return sync_wrapper

    return decorator  # type: ignore[return-value]


class Guardrails:
    """
    Guardrails class that wraps the Evaluator class for real-time evaluations.
    Unlike experiments, guardrails don't require task/experiment IDs.
    """

    def __init__(self, async_http_client: httpx.AsyncClient):
        self._evaluator = Evaluator(async_http_client)

    async def execute_evaluator(
        self,
        slug: str,
        data: Dict[str, str],
        evaluator_version: Optional[str] = None,
        evaluator_config: Optional[Dict[str, Any]] = None
    ) -> OutputSchema:
        """
        Execute evaluator for guardrails (real-time evaluation without experiment context).

        Args:
            slug: The evaluator slug to execute
            data: Input data mapping (guardrails format with InputExtractor)
            evaluator_version: Optional version of the evaluator
            evaluator_config: Optional configuration for the evaluator

        Returns:
            OutputSchema: The evaluation result with success/reason fields
        """
        try:

            # Use dummy IDs for guardrails (they don't need experiment tracking)
            result = await self._evaluator.run_experiment_evaluator(
                evaluator_slug=slug,
                task_id="guardrail",
                experiment_id="guardrail",
                experiment_run_id="guardrail",
                input=data,
                timeout_in_sec=120,
                evaluator_version=evaluator_version,
                evaluator_config=evaluator_config,
            )

            # Parse the result to OutputSchema format
            # Made by Traceloop evaluators return: {'evaluator_result': 'pass'/'fail', ...}
            raw_result = result.result

            # Check if this is a Made by Traceloop evaluator format
            if "evaluator_result" in raw_result:
                evaluator_result = raw_result.get("evaluator_result", "")
                success = evaluator_result == "pass"
                reason = raw_result.get("reason", None)
                return OutputSchema(reason=reason, success=success)

            # Otherwise, try custom evaluator format with 'pass' and 'reason' fields
            return OutputSchema.model_validate(raw_result)

        except Exception as e:
            print(f"Error executing evaluator {slug}. Error: {str(e)}")
            # Return a failure result on error
            return OutputSchema(reason=f"Evaluation error: {str(e)}", success=False)
