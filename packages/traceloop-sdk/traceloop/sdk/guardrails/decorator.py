import asyncio
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, ParamSpec, Awaitable, Dict

from traceloop.sdk.guardrails.guardrails import Guardrails
from .context import set_current_score
from .types import InputExtractor

P = ParamSpec("P")
R = TypeVar("R")
F = TypeVar("F", bound=Callable[P, R | Awaitable[R]])

def guardrails(
    evaluator_slug: str,
    score_calculator: Callable[[Dict[str, Any]], float],
    input_schema: Dict[str, InputExtractor],
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
        print("is_async: ", is_async)
        entity_name = name or f"{fn.__qualname__}.guardrails"
        
        if is_async:
            @wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    # Execute guardrails evaluation
                    event_data = await _execute_evaluator(evaluator_slug, input_schema, args, kwargs)
                    
                    # Calculate score
                    score = score_calculator(event_data)
                    
                    # Set score in context for access within function
                    set_current_score(score)

                    # Call the original function
                    return await fn(*args, **kwargs)
                    
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
                    event_data = asyncio.run(_execute_evaluator(evaluator_slug, input_schema, args, kwargs))
                    
                    # Calculate score
                    score = score_calculator(event_data)
                    
                    # Set score in context for access within function
                    set_current_score(score)

                    # Call the original function
                    return fn(*args, **kwargs)
                    
                except Exception as e:
                    raise
                finally:
                    pass
            
            return sync_wrapper
    
    return decorate

async def _execute_evaluator(evaluator_slug: str, input_schema: Dict[str, InputExtractor], args: tuple, kwargs: dict) -> Dict[str, Any]:
    """Execute guardrails evaluation and return event data."""
    print("Executing evaluator")
    try:    
     
        # Get client instance without circular import
        client = _get_client()
        event_data = await client.guardrails.execute_evaluator(evaluator_slug, input_schema)
        
        
        return event_data
    except Exception as e:
        # Log error and return empty data
        print(f"Error executing guardrails: {str(e)}")
        return {}
    
def _get_client():
    """Get the Traceloop client instance without circular import."""
    # Import here to avoid circular import
    from traceloop.sdk import Traceloop
    return Traceloop.get() 
