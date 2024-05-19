import json
from functools import wraps
import os
import types
from typing import Optional

from opentelemetry import trace
from opentelemetry import context as context_api
from opentelemetry.semconv.ai import SpanAttributes, TraceloopSpanKindValues

from traceloop.sdk.telemetry import Telemetry
from traceloop.sdk.tracing import get_tracer, set_workflow_name
from traceloop.sdk.tracing.tracing import (
    TracerWrapper,
    set_entity_name,
    get_chained_entity_name,
)
from traceloop.sdk.utils import camel_to_snake


def entity_method(
    name: Optional[str] = None,
    tlp_span_kind: Optional[TraceloopSpanKindValues] = TraceloopSpanKindValues.TASK,
):
    def decorate(fn):
        @wraps(fn)
        def wrap(*args, **kwargs):
            if not TracerWrapper.verify_initialized():
                return fn(*args, **kwargs)

            entity_name = name or fn.__name__
            if tlp_span_kind in [
                TraceloopSpanKindValues.WORKFLOW,
                TraceloopSpanKindValues.AGENT,
            ]:
                set_workflow_name(entity_name)
            span_name = f"{entity_name}.{tlp_span_kind.value}"

            with get_tracer() as tracer:
                span = tracer.start_span(span_name)
                ctx = trace.set_span_in_context(span)
                context_api.attach(ctx)

                if tlp_span_kind in [
                    TraceloopSpanKindValues.TASK,
                    TraceloopSpanKindValues.TOOL,
                ]:
                    chained_entity_name = get_chained_entity_name(entity_name)
                    set_entity_name(chained_entity_name)
                else:
                    chained_entity_name = entity_name

                span.set_attribute(
                    SpanAttributes.TRACELOOP_SPAN_KIND, tlp_span_kind.value
                )
                span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_NAME, chained_entity_name
                )

                try:
                    if _should_send_prompts():
                        span.set_attribute(
                            SpanAttributes.TRACELOOP_ENTITY_INPUT,
                            json.dumps({"args": args, "kwargs": kwargs}),
                        )
                except TypeError as e:
                    Telemetry().log_exception(e)

                res = fn(*args, **kwargs)

                # span will be ended in the generator
                if isinstance(res, types.GeneratorType):
                    return _handle_generator(span, fn, args, kwargs)

                try:
                    if _should_send_prompts():
                        span.set_attribute(
                            SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json.dumps(res)
                        )
                except TypeError as e:
                    Telemetry().log_exception(e)

                span.end()

                return res

        return wrap

    return decorate


def entity_class(
    name: Optional[str],
    method_name: str,
    tlp_span_kind: Optional[TraceloopSpanKindValues] = TraceloopSpanKindValues.TASK,
):
    def decorator(cls):
        task_name = name if name else camel_to_snake(cls.__name__)
        method = getattr(cls, method_name)
        setattr(
            cls,
            method_name,
            entity_method(name=task_name, tlp_span_kind=tlp_span_kind)(method),
        )
        return cls

    return decorator


# Async Decorators


def aentity_method(
    name: Optional[str] = None,
    tlp_span_kind: Optional[TraceloopSpanKindValues] = TraceloopSpanKindValues.TASK,
):
    def decorate(fn):
        @wraps(fn)
        async def wrap(*args, **kwargs):
            if not TracerWrapper.verify_initialized():
                return await fn(*args, **kwargs)

            span_name = (
                f"{name}.{tlp_span_kind.value}"
                if name
                else f"{fn.__name__}.{tlp_span_kind.value}"
            )
            with get_tracer() as tracer:
                with tracer.start_as_current_span(span_name) as span:
                    span.set_attribute(
                        SpanAttributes.TRACELOOP_SPAN_KIND, tlp_span_kind.value
                    )
                    span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, name)

                    try:
                        if _should_send_prompts():
                            span.set_attribute(
                                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                                json.dumps({"args": args, "kwargs": kwargs}),
                            )
                    except TypeError as e:
                        Telemetry().log_exception(e)

                    res = await fn(*args, **kwargs)

                    # span will be ended in the generator
                    if isinstance(res, types.AsyncGeneratorType):
                        return await _ahandle_generator(span, fn, args, kwargs)

                    try:
                        if _should_send_prompts():
                            span.set_attribute(
                                SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json.dumps(res)
                            )
                    except TypeError as e:
                        Telemetry().log_exception(e)

                    return res

        return wrap

    return decorate


def aentity_class(
    name: Optional[str],
    method_name: str,
    tlp_span_kind: Optional[TraceloopSpanKindValues] = TraceloopSpanKindValues.TASK,
):
    def decorator(cls):
        task_name = name if name else camel_to_snake(cls.__name__)
        method = getattr(cls, method_name)
        setattr(
            cls,
            method_name,
            aentity_method(name=task_name, tlp_span_kind=tlp_span_kind)(method),
        )
        return cls

    return decorator


def _handle_generator(span, fn, args, kwargs):
    for part in fn(*args, **kwargs):
        yield part

    span.end()


async def _ahandle_generator(span, fn, args, kwargs):
    async for part in fn(*args, **kwargs):
        yield part

    span.end()


def _should_send_prompts():
    return (
        os.getenv("TRACELOOP_TRACE_CONTENT") or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")
