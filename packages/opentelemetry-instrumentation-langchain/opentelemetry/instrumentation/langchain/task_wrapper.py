from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY

from opentelemetry.semconv.ai import SpanAttributes, TraceloopSpanKindValues

from opentelemetry.instrumentation.langchain.utils import (
    _with_tracer_wrapper,
    process_request,
    process_response,
)


@_with_tracer_wrapper
def task_wrapper(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    # Some Langchain objects are wrapped elsewhere, so we ignore them here
    if instance.__class__.__name__ in ("AgentExecutor"):
        return wrapped(*args, **kwargs)

    name, kind = _handle_request(instance, args, to_wrap)

    with tracer.start_as_current_span(name) as span:
        span.set_attribute(
            SpanAttributes.TRACELOOP_SPAN_KIND,
            kind,
        )
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, name)

        process_request(span, args, kwargs)
        return_value = wrapped(*args, **kwargs)
        process_response(span, return_value)

    return return_value


@_with_tracer_wrapper
async def atask_wrapper(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    # Some Langchain objects are wrapped elsewhere, so we ignore them here
    if instance.__class__.__name__ in ("AgentExecutor"):
        return wrapped(*args, **kwargs)

    name, kind = _handle_request(instance, args, to_wrap)

    with tracer.start_as_current_span(name) as span:
        span.set_attribute(
            SpanAttributes.TRACELOOP_SPAN_KIND,
            kind,
        )
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, name)

        process_request(span, args, kwargs)
        return_value = await wrapped(*args, **kwargs)
        process_response(span, return_value)

    return return_value


def _handle_request(instance, args, to_wrap):
    config = args[1] if len(args) > 1 else {}
    run_name = config.get("run_name") or instance.get_name()
    name = f"{run_name}.langchain.task" if run_name else to_wrap.get("span_name")

    kind = to_wrap.get("kind") or TraceloopSpanKindValues.TASK.value

    return name, kind
