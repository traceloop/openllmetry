from typing import Any, Optional, TypeVar, Callable, ParamSpec, Union
import asyncio
import warnings
from functools import wraps

from opentelemetry.semconv_ai import TraceloopSpanKindValues

from traceloop.sdk.decorators.base import (
    entity_class,
    entity_method,
)
from traceloop.sdk.guardrail.model import Guard, OnFailureHandler, InputMapper
from traceloop.sdk.guardrail.on_failure import OnFailure

F = TypeVar("F", bound=Callable[..., Any])
_P = ParamSpec("_P")
_R = TypeVar("_R")


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
    on_failure: Union[OnFailureHandler, str, None] = None,
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
            - OnFailure handler (e.g., OnFailure.raise_exception())
            - String (shorthand for OnFailure.return_value(string))
            - None (defaults to OnFailure.raise_exception())
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
    # Convert string on_failure to OnFailure.return_value
    if isinstance(on_failure, str):
        failure_handler = OnFailure.return_value(on_failure)
    elif on_failure is None:
        failure_handler = OnFailure.raise_exception()
    else:
        failure_handler = on_failure

    guards_list = list(guards)

    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        is_async = asyncio.iscoroutinefunction(func)

        if is_async:
            @wraps(func)
            async def async_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                from traceloop.sdk import Traceloop

                client = Traceloop.get()
                g = client.guardrails.create(
                    guards=guards_list,
                    on_failure=failure_handler,
                    name=name or func.__name__,
                )
                return await g.run(
                    lambda: func(*args, **kwargs),
                    input_mapper=input_mapper,
                )

            return async_wrapper  # type: ignore[return-value]
        else:
            @wraps(func)
            def sync_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                from traceloop.sdk import Traceloop

                client = Traceloop.get()
                g = client.guardrails.create(
                    guards=guards_list,
                    on_failure=failure_handler,
                    name=name or func.__name__,
                )
                # Run async guardrail in event loop for sync functions
                return asyncio.run(
                    g.run(
                        lambda: asyncio.to_thread(func, *args, **kwargs),
                        input_mapper=input_mapper,
                    )
                )

            return sync_wrapper  # type: ignore[return-value]

    return decorator
