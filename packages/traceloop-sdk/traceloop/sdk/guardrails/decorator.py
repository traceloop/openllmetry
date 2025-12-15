import json
import asyncio
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, ParamSpec, Awaitable, cast

from opentelemetry import trace
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.semconv_ai import TraceloopSpanKindValues

from traceloop.sdk.decorators.base import _setup_span, _handle_span_input, _handle_span_output, _cleanup_span
from traceloop.sdk.tracing import get_tracer
from traceloop.sdk.client import Client
from traceloop.sdk.utils.json_encoder import JSONEncoder

from .types import InputSchemaMapping
from .context import set_current_score
from .client import GuardrailsClient
from .utils import extract_input_data, extract_output_data

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
                # Setup span
                span, ctx, ctx_token = _setup_span(entity_name, TraceloopSpanKindValues.TASK, version)
                _handle_span_input(span, args, kwargs, cls=JSONEncoder)
                
                try:
                    # Execute original function first
                    result = await fn(*args, **kwargs)
                    
                    # Extract input data based on schema
                    input_data = extract_input_data(args, kwargs, input_schema)
                    
                    # Also extract from output if needed
                    output_data = extract_output_data(result, input_schema)
                    input_data.update(output_data)
                    
                    # Execute guardrails evaluation
                    event_data = await _execute_guardrails(evaluator_slug, input_data)
                    
                    # Calculate score
                    score = score_calculator(event_data)
                    
                    # Set score in context for access within function
                    set_current_score(score)
                    
                    # Add guardrails attributes to span
                    span.set_attribute("traceloop.guardrails.score", score)
                    span.set_attribute("traceloop.guardrails.evaluator_slug", evaluator_slug)
                    span.set_attribute("traceloop.guardrails.input_schema", json.dumps(input_schema))
                    span.set_attribute("traceloop.guardrails.event_data", json.dumps(event_data))
                    
                    _handle_span_output(span, result, cls=JSONEncoder)
                    return result
                    
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
                finally:
                    _cleanup_span(span, ctx_token)
            
            return cast(F, async_wrapper)
        else:
            @wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Setup span
                span, ctx, ctx_token = _setup_span(entity_name, TraceloopSpanKindValues.TASK, version)
                _handle_span_input(span, args, kwargs, cls=JSONEncoder)
                
                try:
                    # Execute original function first
                    result = fn(*args, **kwargs)
                    
                    # Extract input data based on schema
                    input_data = extract_input_data(args, kwargs, input_schema)
                    
                    # Also extract from output if needed
                    output_data = extract_output_data(result, input_schema)
                    input_data.update(output_data)
                    
                    # Execute guardrails evaluation in async context
                    event_data = asyncio.run(_execute_guardrails(evaluator_slug, input_data))
                    
                    # Calculate score
                    score = score_calculator(event_data)
                    
                    # Set score in context for access within function
                    set_current_score(score)
                    
                    # Add guardrails attributes to span
                    span.set_attribute("traceloop.guardrails.score", score)
                    span.set_attribute("traceloop.guardrails.evaluator_slug", evaluator_slug)
                    span.set_attribute("traceloop.guardrails.input_schema", json.dumps(input_schema))
                    span.set_attribute("traceloop.guardrails.event_data", json.dumps(event_data))
                    
                    _handle_span_output(span, result, cls=JSONEncoder)
                    return result
                    
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
                finally:
                    _cleanup_span(span, ctx_token)
            
            return cast(F, sync_wrapper)
    
    return decorate

async def _execute_guardrails(evaluator_slug: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute guardrails evaluation and return event data."""
    try:
        # Get the current client instance
        from traceloop.sdk import get_client
        client = get_client()
        
        # Create guardrails client
        guardrails_client = GuardrailsClient(client._http)
        
        # Execute evaluator
        event_data = await guardrails_client.execute_evaluator(evaluator_slug, input_data)
        
        return event_data
    except Exception as e:
        # Log error and return empty data
        print(f"Error executing guardrails: {str(e)}")
        return {} 