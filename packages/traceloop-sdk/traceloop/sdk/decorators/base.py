import json
from functools import wraps
import os
from typing import (
    TypeVar,
    Optional,
    Callable,
    Any,
    cast,
)
import inspect
import warnings
import types

from opentelemetry import trace
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry import context as context_api
from opentelemetry.semconv_ai import SpanAttributes, TraceloopSpanKindValues
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_TOOL_NAME,
)

from traceloop.sdk.tracing import get_tracer, set_workflow_name, set_agent_name
from traceloop.sdk.tracing.tracing import (
    TracerWrapper,
    set_entity_path,
    get_chained_entity_path,
)
from traceloop.sdk.utils import camel_to_snake
from traceloop.sdk.utils.json_encoder import JSONEncoder

F = TypeVar("F", bound=Callable[..., Any])


def _truncate_json_if_needed(json_str: str) -> str:
    """
    Truncate JSON string if it exceeds OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT;
    truncation may yield an invalid JSON string, which is expected for logging purposes.
    """
    limit_str = os.getenv("OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT")
    if limit_str:
        try:
            limit = int(limit_str)
            if limit > 0 and len(json_str) > limit:
                return json_str[:limit]
        except ValueError:
            pass
    return json_str


# Async Decorators - Deprecated


def aentity_method(
    name: Optional[str] = None,
    version: Optional[int] = None,
    tlp_span_kind: Optional[TraceloopSpanKindValues] = TraceloopSpanKindValues.TASK,
):
    warnings.warn(
        "DeprecationWarning: The @aentity_method function will be removed in a future version. "
        "Please migrate to @entity_method for both sync and async operations.",
        DeprecationWarning,
        stacklevel=2,
    )

    return entity_method(
        name=name,
        version=version,
        tlp_span_kind=tlp_span_kind,
    )


def aentity_class(
    name: Optional[str],
    version: Optional[int],
    method_name: str,
    tlp_span_kind: Optional[TraceloopSpanKindValues] = TraceloopSpanKindValues.TASK,
):
    warnings.warn(
        "DeprecationWarning: The @aentity_class function will be removed in a future version. "
        "Please migrate to @entity_class for both sync and async operations.",
        DeprecationWarning,
        stacklevel=2,
    )

    return entity_class(
        name=name,
        version=version,
        method_name=method_name,
        tlp_span_kind=tlp_span_kind,
    )


def _safe_detach(token) -> None:
    """Detach a context token, ignoring the "created in a different Context" error.

    OTel stores context in ``contextvars``, which are copied per asyncio-task /
    thread (PEP 567). When a decorated generator or coroutine resumes on a
    different task/thread than the one that attached the token, ``detach`` raises
    a ``ValueError``. The stale token is harmless to skip, so we swallow that
    failure while still detaching in the common case.
    See https://github.com/open-telemetry/opentelemetry-python/issues/2606
    """
    if token is None:
        return
    try:
        context_api.detach(token)
    except Exception:
        pass


def _detach_tokens(ctx_tokens) -> None:
    """Detach the (name, span, path) tokens in reverse (LIFO) attach order.

    OTel context is a stack: the name token is attached first (bottom) and the
    path token last (top), so we detach path -> span -> name. Detaching out of
    order makes OTel complain the token "was created in a different Context".
    """
    name_token, ctx_token, path_token = ctx_tokens
    _safe_detach(path_token)
    _safe_detach(ctx_token)
    _safe_detach(name_token)


def _handle_generator(span, ctx_tokens, res):
    # for some reason the SPAN_KEY is not being set in the context of the generator, so we re-set it
    # Capture this re-attach's token so it can be detached on cleanup; otherwise
    # this frame is left on the context stack (unbalanced attach).
    gen_span_token = context_api.attach(trace.set_span_in_context(span))
    try:
        for item in res:
            yield item
    except Exception as e:
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.record_exception(e)
        raise
    finally:
        span.end()
        # Best-effort detach: closing a generator on a different task/thread than
        # the one that attached the tokens makes detach fail
        # (https://github.com/open-telemetry/opentelemetry-python/issues/2606).
        # _safe_detach swallows that, so this stops the name/path from leaking in
        # the common case and degrades quietly otherwise (context is reclaimed
        # during garbage collection). Detach the re-attached span first (LIFO:
        # it was attached after the tokens in ctx_tokens).
        _safe_detach(gen_span_token)
        _detach_tokens(ctx_tokens)


async def _ahandle_generator(span, ctx_tokens, res):
    try:
        async for part in res:
            yield part
    except Exception as e:
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.record_exception(e)
        raise
    finally:
        span.end()
        _detach_tokens(ctx_tokens)


def _should_send_prompts():
    return (
        os.getenv("TRACELOOP_TRACE_CONTENT") or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")


# Unified Decorators : handles both sync and async operations


def _is_async_method(fn):
    # check if co-routine function or async generator( example : using async & yield)
    return inspect.iscoroutinefunction(fn) or inspect.isasyncgenfunction(fn)


def _setup_span(entity_name, tlp_span_kind, version):
    """Sets up the OpenTelemetry span and context.

    Returns the span, the span context, and a ``(name_token, ctx_token,
    path_token)`` tuple ordered oldest-first. ``_cleanup_span`` detaches these in
    reverse, so the entity name/path we attach here is scoped to the decorated
    function and does not leak onto sibling or parent spans.
    """
    # Attached first (bottom of the stack), detached last. If we don't detach it
    # later, the agent/workflow name sticks on the context for the rest of the trace.
    name_token = None
    if tlp_span_kind == TraceloopSpanKindValues.WORKFLOW:
        name_token = set_workflow_name(entity_name)
    elif tlp_span_kind == TraceloopSpanKindValues.AGENT:
        name_token = set_agent_name(entity_name)

    span_name = f"{entity_name}.{tlp_span_kind.value}"

    path_token = None
    with get_tracer() as tracer:
        span = tracer.start_span(span_name)
        ctx = trace.set_span_in_context(span)
        ctx_token = context_api.attach(ctx)

        if tlp_span_kind in [
            TraceloopSpanKindValues.TASK,
            TraceloopSpanKindValues.TOOL,
        ]:
            entity_path = get_chained_entity_path(entity_name)
            path_token = set_entity_path(entity_path)

        span.set_attribute(SpanAttributes.TRACELOOP_SPAN_KIND, tlp_span_kind.value)
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, entity_name)
        if tlp_span_kind == TraceloopSpanKindValues.TOOL:
            span.set_attribute(GEN_AI_TOOL_NAME, entity_name)
        if version:
            span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_VERSION, version)

    # Ordered oldest-first (attach order); cleanup detaches in reverse.
    ctx_tokens = (name_token, ctx_token, path_token)
    return span, ctx, ctx_tokens


def _handle_span_input(span, args, kwargs, cls=None):
    """Handles entity input logging in JSON for both sync and async functions"""
    try:
        if _should_send_prompts():
            json_input = json.dumps(
                {"args": args, "kwargs": kwargs}, **({"cls": cls} if cls else {})
            )
            truncated_json = _truncate_json_if_needed(json_input)
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                truncated_json,
            )
    except TypeError:
        pass


def _handle_span_output(span, res, cls=None):
    """Handles entity output logging in JSON for both sync and async functions"""
    try:
        if _should_send_prompts():
            json_output = json.dumps(res, **({"cls": cls} if cls else {}))
            truncated_json = _truncate_json_if_needed(json_output)
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                truncated_json,
            )
    except TypeError:
        pass


def _cleanup_span(span, ctx_tokens):
    """End the span and detach all context tokens (name, span, path) in LIFO order."""
    span.end()
    _detach_tokens(ctx_tokens)


def entity_method(
    name: Optional[str] = None,
    version: Optional[int] = None,
    tlp_span_kind: Optional[TraceloopSpanKindValues] = TraceloopSpanKindValues.TASK,
) -> Callable[[F], F]:
    def decorate(fn: F) -> F:
        is_async = _is_async_method(fn)
        entity_name = name or fn.__qualname__
        if is_async:
            if inspect.isasyncgenfunction(fn):

                @wraps(fn)
                async def async_gen_wrap(*args: Any, **kwargs: Any) -> Any:
                    if not TracerWrapper.verify_initialized():
                        async for item in fn(*args, **kwargs):
                            yield item
                        return

                    span, ctx, ctx_tokens = _setup_span(
                        entity_name, tlp_span_kind, version
                    )
                    _handle_span_input(span, args, kwargs, cls=JSONEncoder)
                    # Delegating with `async for` does not propagate close, so the
                    # inner generator's finally (end span + detach tokens) would
                    # only run whenever it happens to be finalized. Closing it
                    # explicitly ties that cleanup to this wrapper's own close.
                    #
                    # KNOWN LIMITATION -- abandoning an async generator (`break`
                    # without `aclose`) still leaks the entity name. Python defers
                    # finalizing the wrapper to the event loop, so by the time this
                    # finally runs we are on a different context: the detach
                    # succeeds against the finalizer's own contextvars copy and
                    # leaves the caller's untouched. It cannot be fixed from inside
                    # the generator; callers who exit early should use
                    # `contextlib.aclosing()` (or drain the generator) to get
                    # deterministic cleanup, which this try/finally then honours.
                    inner = _ahandle_generator(span, ctx_tokens, fn(*args, **kwargs))
                    try:
                        async for item in inner:
                            yield item
                    finally:
                        await inner.aclose()

                return cast(F, async_gen_wrap)
            else:

                @wraps(fn)
                async def async_wrap(*args: Any, **kwargs: Any) -> Any:
                    if not TracerWrapper.verify_initialized():
                        return await fn(*args, **kwargs)

                    span, ctx, ctx_tokens = _setup_span(
                        entity_name, tlp_span_kind, version
                    )
                    _handle_span_input(span, args, kwargs, cls=JSONEncoder)
                    try:
                        res = await fn(*args, **kwargs)
                        _handle_span_output(span, res, cls=JSONEncoder)
                        return res
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise
                    finally:
                        _cleanup_span(span, ctx_tokens)

                return cast(F, async_wrap)
        else:

            @wraps(fn)
            def sync_wrap(*args: Any, **kwargs: Any) -> Any:
                if not TracerWrapper.verify_initialized():
                    return fn(*args, **kwargs)

                span, ctx, ctx_tokens = _setup_span(entity_name, tlp_span_kind, version)
                _handle_span_input(span, args, kwargs, cls=JSONEncoder)
                try:
                    res = fn(*args, **kwargs)
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    _cleanup_span(span, ctx_tokens)
                    raise

                # span will be ended in the generator
                if isinstance(res, types.GeneratorType):
                    return _handle_generator(span, ctx_tokens, res)

                _handle_span_output(span, res, cls=JSONEncoder)
                _cleanup_span(span, ctx_tokens)
                return res

            return cast(F, sync_wrap)

    return decorate


def entity_class(
    name: Optional[str],
    version: Optional[int],
    method_name: str,
    tlp_span_kind: Optional[TraceloopSpanKindValues] = TraceloopSpanKindValues.TASK,
):
    def decorator(cls):
        task_name = name if name else camel_to_snake(cls.__qualname__)
        method = getattr(cls, method_name)
        setattr(
            cls,
            method_name,
            entity_method(name=task_name, version=version, tlp_span_kind=tlp_span_kind)(
                method
            ),
        )
        return cls

    return decorator
