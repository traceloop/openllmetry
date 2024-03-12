import json
from functools import wraps
import os
from typing import Optional

from opentelemetry import context as context_api
from opentelemetry.semconv.ai import SpanAttributes, TraceloopSpanKindValues

from traceloop.sdk.tracing import get_tracer, set_workflow_name
from traceloop.sdk.tracing.tracing import TracerWrapper
from traceloop.sdk.utils import camel_to_snake


def task(
    name: Optional[str] = None,
    method_name: Optional[str] = None,
    tlp_span_kind: Optional[TraceloopSpanKindValues] = TraceloopSpanKindValues.TASK,
):
    if method_name is None:
        return task_method(name=name, tlp_span_kind=tlp_span_kind)
    else:
        return task_class(
            name=name, method_name=method_name, tlp_span_kind=tlp_span_kind
        )


def task_method(
    name: Optional[str] = None,
    tlp_span_kind: Optional[TraceloopSpanKindValues] = TraceloopSpanKindValues.TASK,
):
    def decorate(fn):
        @wraps(fn)
        def wrap(*args, **kwargs):
            if not TracerWrapper.verify_initialized():
                return fn(*args, **kwargs)

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
                    except TypeError:
                        pass  # Some args might not be serializable

                    res = fn(*args, **kwargs)

                    try:
                        if _should_send_prompts():
                            span.set_attribute(
                                SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json.dumps(res)
                            )
                    except TypeError:
                        pass  # Some outputs might not be serializable

                    return res

        return wrap

    return decorate


def task_class(
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
            task_method(name=task_name, tlp_span_kind=tlp_span_kind)(method),
        )
        return cls

    return decorator


def workflow(
    name: Optional[str] = None,
    method_name: Optional[str] = None,
    correlation_id: Optional[str] = None,
):
    if method_name is None:
        return workflow_method(name=name, correlation_id=correlation_id)
    else:
        return workflow_class(
            name=name, method_name=method_name, correlation_id=correlation_id
        )


def workflow_method(name: Optional[str] = None, correlation_id: Optional[str] = None):
    def decorate(fn):
        @wraps(fn)
        def wrap(*args, **kwargs):
            if not TracerWrapper.verify_initialized():
                return fn(*args, **kwargs)

            workflow_name = name or fn.__name__
            set_workflow_name(workflow_name)
            span_name = f"{workflow_name}.workflow"

            with get_tracer(flush_on_exit=True) as tracer:
                with tracer.start_as_current_span(span_name) as span:
                    span.set_attribute(
                        SpanAttributes.TRACELOOP_SPAN_KIND,
                        TraceloopSpanKindValues.WORKFLOW.value,
                    )
                    span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, name)

                    try:
                        if _should_send_prompts():
                            span.set_attribute(
                                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                                json.dumps({"args": args, "kwargs": kwargs}),
                            )
                    except TypeError:
                        pass  # Some args might not be serializable

                    res = fn(*args, **kwargs)

                    try:
                        if _should_send_prompts():
                            span.set_attribute(
                                SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json.dumps(res)
                            )
                    except TypeError:
                        pass  # Some outputs might not be serializable

                    return res

        return wrap

    return decorate


def workflow_class(
    name: Optional[str], method_name: str, correlation_id: Optional[str] = None
):
    def decorator(cls):
        workflow_name = name if name else camel_to_snake(cls.__name__)
        method = getattr(cls, method_name)
        setattr(
            cls,
            method_name,
            workflow_method(name=workflow_name, correlation_id=correlation_id)(method),
        )
        return cls

    return decorator


def agent(name: Optional[str] = None, method_name: Optional[str] = None):
    return task(
        name=name, method_name=method_name, tlp_span_kind=TraceloopSpanKindValues.AGENT
    )


def tool(name: Optional[str] = None, method_name: Optional[str] = None):
    return task(
        name=name, method_name=method_name, tlp_span_kind=TraceloopSpanKindValues.TOOL
    )


# Async Decorators
def atask(
    name: Optional[str] = None,
    method_name: Optional[str] = None,
    tlp_span_kind: Optional[TraceloopSpanKindValues] = TraceloopSpanKindValues.TASK,
):
    if method_name is None:
        return atask_method(name=name, tlp_span_kind=tlp_span_kind)
    else:
        return atask_class(
            name=name, method_name=method_name, tlp_span_kind=tlp_span_kind
        )


def atask_method(
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
                    except TypeError:
                        pass  # Some args might not be serializable

                    res = await fn(*args, **kwargs)

                    try:
                        if _should_send_prompts():
                            span.set_attribute(
                                SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json.dumps(res)
                            )
                    except TypeError:
                        pass  # Some args might not be serializable

                    return res

        return wrap

    return decorate


def atask_class(
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
            atask_method(name=task_name, tlp_span_kind=tlp_span_kind)(method),
        )
        return cls

    return decorator


def aworkflow(
    name: Optional[str] = None,
    method_name: Optional[str] = None,
    correlation_id: Optional[str] = None,
):
    if method_name is None:
        return aworkflow_method(name=name, correlation_id=correlation_id)
    else:
        return aworkflow_class(
            name=name, method_name=method_name, correlation_id=correlation_id
        )


def aworkflow_method(name: Optional[str] = None, correlation_id: Optional[str] = None):
    def decorate(fn):
        @wraps(fn)
        async def wrap(*args, **kwargs):
            if not TracerWrapper.verify_initialized():
                return await fn(*args, **kwargs)

            workflow_name = name or fn.__name__
            set_workflow_name(workflow_name)
            span_name = f"{workflow_name}.workflow"

            with get_tracer(flush_on_exit=True) as tracer:
                with tracer.start_as_current_span(span_name) as span:
                    span.set_attribute(
                        SpanAttributes.TRACELOOP_SPAN_KIND,
                        TraceloopSpanKindValues.WORKFLOW.value,
                    )
                    span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, name)

                    if correlation_id:
                        span.set_attribute(
                            SpanAttributes.TRACELOOP_CORRELATION_ID, correlation_id
                        )

                    try:
                        if _should_send_prompts():
                            span.set_attribute(
                                SpanAttributes.TRACELOOP_ENTITY_INPUT,
                                json.dumps({"args": args, "kwargs": kwargs}),
                            )
                    except TypeError:
                        pass  # Some args might not be serializable

                    res = await fn(*args, **kwargs)

                    try:
                        if _should_send_prompts():
                            span.set_attribute(
                                SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json.dumps(res)
                            )
                    except TypeError:
                        pass  # Some args might not be serializable

                    return res

        return wrap

    return decorate


def aworkflow_class(
    name: Optional[str], method_name: str, correlation_id: Optional[str] = None
):
    def decorator(cls):
        workflow_name = name if name else camel_to_snake(cls.__name__)
        method = getattr(cls, method_name)
        setattr(
            cls,
            method_name,
            aworkflow_method(name=workflow_name, correlation_id=correlation_id)(method),
        )
        return cls

    return decorator


def aagent(name: Optional[str] = None, method_name: Optional[str] = None):
    return atask(
        name=name, method_name=method_name, tlp_span_kind=TraceloopSpanKindValues.AGENT
    )


def atool(name: Optional[str] = None, method_name: Optional[str] = None):
    return atask(
        name=name, method_name=method_name, tlp_span_kind=TraceloopSpanKindValues.TOOL
    )


def _should_send_prompts():
    return (
        os.getenv("TRACELOOP_TRACE_CONTENT") or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")
