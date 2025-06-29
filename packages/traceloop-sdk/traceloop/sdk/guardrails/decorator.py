import json
import asyncio
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, ParamSpec, Awaitable, cast, Dict

from opentelemetry import trace
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.semconv_ai import TraceloopSpanKindValues

from traceloop.sdk.decorators.base import _setup_span, _handle_span_input, _handle_span_output, _cleanup_span
from traceloop.sdk.tracing import get_tracer
from traceloop.sdk.client import Client
from traceloop.sdk.utils.json_encoder import JSONEncoder

from .context import set_current_score
from .client import GuardrailsClient
from .utils import extract_input_data, extract_output_data
from .types import InputSchemaMapping

P = ParamSpec("P")
R = TypeVar("R")
F = TypeVar("F", bound=Callable[P, R | Awaitable[R]])

def guardrails(
    evaluator_slug: str,
    score_calculator: Callable[[Dict[str, Any]], float],
    input_schema: InputSchemaMapping,
    name: Optional[str] = None,
    version: Optional[int] = None,
) -> Callable[[F], F]:
    """
    Decorator that applies guardrails evaluation to a function.
    
    Args:
        evaluator_slug: The slug of the evaluator to use
        score_calculator: Function that calculates score from event data
        input_schema: Mapping of field names to input extractors
        name: Optional name for the guardrails span
        version: Optional version for the guardrails span
    """
    def decorate(fn: F) -> F:
        is_async = asyncio.iscoroutinefunction(fn)
        entity_name = name or f"{fn.__qualname__}.guardrails"
        
        if is_async:
            @wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    
                    # Execute guardrails evaluation
                    event_data = await _execute_evaluator(evaluator_slug, input_schema)
                    
                    # Calculate score
                    score = score_calculator(event_data)
                    
                    # Set score in context for access within function
                    set_current_score(score)

                    return
                    
                except Exception as e:
                    raise
                finally:
                    pass
            
            return async_wrapper
        else:
            @wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:

                try:
                    # Execute guardrails evaluation in async context
                    event_data = asyncio.run(_execute_evaluator(evaluator_slug, input_schema))
                    
                    # Calculate score
                    score = score_calculator(event_data)
                    
                    # Set score in context for access within function
                    set_current_score(score)

                    return
                    
                except Exception as e:
                    raise
                finally:
                    pass
            
            return sync_wrapper
    
    return decorate

async def _execute_evaluator(evaluator_slug: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute guardrails evaluation and return event data."""
    try:    
        # Create guardrails client
        event_data = await GuardrailsClient().execute_evaluator(evaluator_slug, input_data)
        
        return event_data
    except Exception as e:
        # Log error and return empty data
        print(f"Error executing guardrails: {str(e)}")
        return {} 