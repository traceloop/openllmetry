from typing import Dict, Any, Optional, Callable, TypeVar, ParamSpec, Union
from traceloop.sdk.evaluator.config import EvaluatorDetails
from traceloop.sdk.evaluator.evaluator import Evaluator
from .types import InputExtractor, OutputSchema
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
):
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
    if isinstance(evaluator, str):
        # Simple string slug - use default field mapping
        evaluator_details = (evaluator, None, None, None)
    elif isinstance(evaluator, EvaluatorDetails):
        # EvaluatorDetails object with config
        evaluator_details = (
            evaluator.slug,
            evaluator.version,
            evaluator.config,
            evaluator.required_input_fields
        )
    else:
        raise ValueError(f"evaluator must be str or EvaluatorDetails, got {type(evaluator)}")

    slug, evaluator_version, evaluator_config, required_input_fields = evaluator_details

    print("NOMI - guardrail - slug:", slug)
    print("NOMI - guardrail - evaluator_version:", evaluator_version)
    print("NOMI - guardrail - evaluator_config:", evaluator_config)
    print("NOMI - guardrail - required_input_fields:", required_input_fields)

    def decorator(func: Callable[P, R]) -> Callable[P, Dict[str, Any]]:
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Dict[str, Any]:
            # Execute the original function - should return a dict with fields matching required_input_fields
            original_result = await func(*args, **kwargs)
            print("NOMI - guardrail - original_result:", original_result)

            # Validate that original_result is a dict
            if not isinstance(original_result, dict):
                raise ValueError(
                    f"Function {func.__name__} must return a dict, got {type(original_result)}. "
                    f"Required fields: {required_input_fields or 'unknown'}"
                )

            try:
                from traceloop.sdk import Traceloop
                client_instance = Traceloop.get()
            except Exception as e:
                print(f"Warning: Could not get Traceloop client: {e}")
                return original_result

            evaluator_result = await client_instance.guardrails.execute_evaluator(
                slug, original_result, evaluator_version, evaluator_config
            )

            print("NOMI - guardrail - evaluator_result:", evaluator_result)

            # Use callback if provided, otherwise use default behavior
            if on_evaluation_complete:
                return on_evaluation_complete(evaluator_result, original_result)
            else:
                # Default behavior: return original result (dict) regardless of pass/fail
                # Users should use callback for custom behavior
                return original_result

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> Dict[str, Any]:

            # Execute the original function - should return a dict with fields matching required_input_fields
            original_result = func(*args, **kwargs)

            # Validate that original_result is a dict
            if not isinstance(original_result, dict):
                raise ValueError(
                    f"Function {func.__name__} must return a dict, got {type(original_result)}. "
                    f"Required fields: {required_input_fields or 'unknown'}"
                )

            # Build evaluator_data based on required_input_fields or use all fields from result
            evaluator_data = {}
            if required_input_fields:
                # Use only the required fields from the function result
                for field in required_input_fields:
                    if field not in original_result:
                        raise ValueError(
                            f"Function {func.__name__} must return dict with field '{field}'. "
                            f"Got: {list(original_result.keys())}"
                        )
                    evaluator_data[field] = InputExtractor(source=original_result[field])
            else:
                # No required fields specified, use all fields from result
                for field, value in original_result.items():
                    evaluator_data[field] = InputExtractor(source=value)

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
                return on_evaluation_complete(evaluator_result, original_result)
            else:
                # Default behavior: return original result (dict) regardless of pass/fail
                # Users should use callback for custom behavior
                return original_result

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


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

            print("NOMI - guardrails - data:", data)
            print("NOMI - guardrails - evaluator_version:", evaluator_version)
            print("NOMI - guardrails - evaluator_config:", evaluator_config)

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
            inner_result = result.result.get("result", {})

            return OutputSchema.model_validate(inner_result)

        except Exception as e:
            print(f"Error executing evaluator {slug}. Error: {str(e)}")
            # Return a failure result on error
            return OutputSchema(reason=f"Evaluation error: {str(e)}", success=False)
