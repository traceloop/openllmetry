from typing import Any, Optional, TypeVar, Callable, ParamSpec
import asyncio
import concurrent.futures
import warnings
import inspect
from functools import wraps

from opentelemetry.semconv_ai import TraceloopSpanKindValues

from traceloop.sdk.decorators.base import (
    entity_class,
    entity_method,
)
from traceloop.sdk.guardrail.model import Guard, InputMapper
from traceloop.sdk.guardrail.on_failure import OnFailureInput, resolve_on_failure

F = TypeVar("F", bound=Callable[..., Any])
_P = ParamSpec("_P")
_R = TypeVar("_R")

__all__ = [
    "conversation",
    "task",
    "workflow",
    "agent",
    "tool",
    "atask",
    "aworkflow",
    "aagent",
    "atool",
]


def conversation(conversation_id: str) -> Callable[[F], F]:
    """
    Decorator to set the conversation ID for all spans within the decorated function.

    Args:
        conversation_id: Unique identifier for the conversation

    Example:
        @conversation(conversation_id="conv-123")
        def handle_chat(user_message: str):
            response = llm.chat(user_message)
            return response

        # Can be combined with @workflow:
        @workflow(name="chat_session")
        @conversation(conversation_id="conv-456")
        def handle_chat_with_workflow(user_message: str):
            response = llm.chat(user_message)
            return response
    """
    from traceloop.sdk.tracing.tracing import set_conversation_id

    def decorator(fn: F) -> F:
        if inspect.iscoroutinefunction(fn):
            @wraps(fn)
            async def async_wrapper(*args, **kwargs):
                set_conversation_id(conversation_id)
                return await fn(*args, **kwargs)
            return async_wrapper
        else:
            @wraps(fn)
            def sync_wrapper(*args, **kwargs):
                set_conversation_id(conversation_id)
                return fn(*args, **kwargs)
            return sync_wrapper

    return decorator


def task(
    name: Optional[str] = None,
    version: Optional[int] = None,
    method_name: Optional[str] = None,
    tlp_span_kind: Optional[TraceloopSpanKindValues] = TraceloopSpanKindValues.TASK,
) -> Callable[[F], F]:
    if method_name is None:
        return entity_method(name=name, version=version, tlp_span_kind=tlp_span_kind)
    else:
        return entity_class(
            name=name,
            version=version,
            method_name=method_name,
            tlp_span_kind=tlp_span_kind,
        )


def workflow(
    name: Optional[str] = None,
    version: Optional[int] = None,
    method_name: Optional[str] = None,
    tlp_span_kind: Optional[TraceloopSpanKindValues] = TraceloopSpanKindValues.WORKFLOW,
) -> Callable[[F], F]:
    if method_name is None:
        return entity_method(name=name, version=version, tlp_span_kind=tlp_span_kind)
    else:
        return entity_class(
            name=name,
            version=version,
            method_name=method_name,
            tlp_span_kind=tlp_span_kind,
        )


def agent(
    name: Optional[str] = None,
    version: Optional[int] = None,
    method_name: Optional[str] = None,
) -> Callable[[F], F]:
    return workflow(
        name=name,
        version=version,
        method_name=method_name,
        tlp_span_kind=TraceloopSpanKindValues.AGENT,
    )


def tool(
    name: Optional[str] = None,
    version: Optional[int] = None,
    method_name: Optional[str] = None,
) -> Callable[[F], F]:
    return task(
        name=name,
        version=version,
        method_name=method_name,
        tlp_span_kind=TraceloopSpanKindValues.TOOL,
    )


# Async Decorators - Deprecated
def atask(
    name: Optional[str] = None,
    version: Optional[int] = None,
    method_name: Optional[str] = None,
    tlp_span_kind: Optional[TraceloopSpanKindValues] = TraceloopSpanKindValues.TASK,
) -> Callable[[F], F]:
    warnings.warn(
        "DeprecationWarning: The @atask decorator will be removed in a future version. "
        "Please migrate to @task for both sync and async operations.",
        DeprecationWarning,
        stacklevel=2,
    )
    if method_name is None:
        return entity_method(name=name, version=version, tlp_span_kind=tlp_span_kind)
    else:
        return entity_class(
            name=name,
            version=version,
            method_name=method_name,
            tlp_span_kind=tlp_span_kind,
        )


def aworkflow(
    name: Optional[str] = None,
    version: Optional[int] = None,
    method_name: Optional[str] = None,
    tlp_span_kind: Optional[TraceloopSpanKindValues] = TraceloopSpanKindValues.WORKFLOW,
) -> Callable[[F], F]:
    warnings.warn(
        "DeprecationWarning: The @aworkflow decorator will be removed in a future version. "
        "Please migrate to @workflow for both sync and async operations.",
        DeprecationWarning,
        stacklevel=2,
    )
    if method_name is None:
        return entity_method(name=name, version=version, tlp_span_kind=tlp_span_kind)
    else:
        return entity_class(
            name=name,
            version=version,
            method_name=method_name,
            tlp_span_kind=tlp_span_kind,
        )


def aagent(
    name: Optional[str] = None,
    version: Optional[int] = None,
    method_name: Optional[str] = None,
) -> Callable[[F], F]:
    warnings.warn(
        "DeprecationWarning: The @aagent decorator will be removed in a future version. "
        "Please migrate to @agent for both sync and async operations.",
        DeprecationWarning,
        stacklevel=2,
    )
    return atask(
        name=name,
        version=version,
        method_name=method_name,
        tlp_span_kind=TraceloopSpanKindValues.AGENT,
    )


def atool(
    name: Optional[str] = None,
    version: Optional[int] = None,
    method_name: Optional[str] = None,
) -> Callable[[F], F]:
    warnings.warn(
        "DeprecationWarning: The @atool decorator will be removed in a future version. "
        "Please migrate to @tool for both sync and async operations.",
        DeprecationWarning,
        stacklevel=2,
    )
    return atask(
        name=name,
        version=version,
        method_name=method_name,
        tlp_span_kind=TraceloopSpanKindValues.TOOL,
    )


def guardrail(
    *guards: Guard,
    input_mapper: InputMapper | None = None,
    on_failure: OnFailureInput | None = None,
    name: str = "",
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """
    Decorator to protect a function with guardrails.

    Works with both sync and async functions.

    Uses the global Traceloop client (from Traceloop.init()).

    Args:
        *guards: Guard functions to run on the output (positional args).
        input_mapper: Function to convert output to guard inputs.
        on_failure: Handler called when any guard fails. Can be:
            - "raise": Raise GuardValidationError (default)
            - "log": Log warning and return result
            - "ignore": Return result silently (shadow mode)
            - Any other string: Return that string as fallback
            - Callable: Custom OnFailureHandler
            - None (defaults to "raise")
        name: Optional name for the guardrail (defaults to function name).

    Example:
        @guardrail(toxicity_guard())
        async def generate_response(prompt: str) -> str:
            return await llm.complete(prompt)

        @guardrail(pii_guard(), toxicity_guard())
        async def safe_generate(prompt: str) -> str:
            return await llm.complete(prompt)

        @guardrail(json_validator_guard(), on_failure="Invalid JSON")
        def generate_json(prompt: str) -> str:
            return llm.complete_sync(prompt)

        # Call directly - guardrail runs automatically
        result = await generate_response("Hello!")
    """
    # Resolve on_failure: None defaults to "raise", strings/callables resolved
    failure_handler = resolve_on_failure(on_failure if on_failure is not None else "raise")

    guards_list = list(guards)

    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        is_async = asyncio.iscoroutinefunction(func)

        if is_async:
            @wraps(func)
            async def async_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                from traceloop.sdk.guardrail import Guardrails

                g = Guardrails(
                    *guards_list,
                    on_failure=failure_handler,
                    name=name or func.__name__,
                )
                return await g.run(
                    func, *args,
                    input_mapper=input_mapper,
                    **kwargs,
                )

            return async_wrapper  # type: ignore[return-value]
        else:
            @wraps(func)
            def sync_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                from traceloop.sdk.guardrail import Guardrails

                g = Guardrails(
                    *guards_list,
                    on_failure=failure_handler,
                    name=name or func.__name__,
                )
                # Run async guardrail in event loop for sync functions
                to_run = g.run(
                    asyncio.to_thread, func, *args,
                    input_mapper=input_mapper,
                    **kwargs,
                )

                try:
                    asyncio.get_running_loop()
                    # Inside an existing event loop (Jupyter, Django, etc.)
                    # Run in a separate thread with its own event loop
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        return pool.submit(asyncio.run, to_run).result()
                except RuntimeError:
                    # No event loop running — safe to use asyncio.run()
                    return asyncio.run(to_run)

            return sync_wrapper  # type: ignore[return-value]

    return decorator
