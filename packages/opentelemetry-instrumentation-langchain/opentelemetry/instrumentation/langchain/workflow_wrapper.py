from opentelemetry import context as context_api
from opentelemetry.context import attach, set_value

from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
)

from opentelemetry.semconv.ai import SpanAttributes, TraceloopSpanKindValues

from opentelemetry.instrumentation.langchain.utils import _with_tracer_wrapper


@_with_tracer_wrapper
def workflow_wrapper(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    config = args[1] if len(args) > 1 else {}
    run_name = config.get("run_name") or instance.get_name()
    name = f"{run_name}.langchain.workflow" if run_name else to_wrap.get("span_name")
    kind = to_wrap.get("kind") or TraceloopSpanKindValues.WORKFLOW.value

    attach(set_value("workflow_name", name))

    with tracer.start_as_current_span(name) as span:
        span.set_attribute(
            SpanAttributes.TRACELOOP_SPAN_KIND,
            kind,
        )
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, name)

        return_value = wrapped(*args, **kwargs)

    return return_value


@_with_tracer_wrapper
async def aworkflow_wrapper(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    config = args[1] if len(args) > 1 else {}
    run_name = config.get("run_name") or instance.get_name()
    name = f"{run_name}.langchain.workflow" if run_name else to_wrap.get("span_name")
    kind = to_wrap.get("kind") or TraceloopSpanKindValues.WORKFLOW.value

    attach(set_value("workflow_name", name))

    with tracer.start_as_current_span(name) as span:
        span.set_attribute(
            SpanAttributes.TRACELOOP_SPAN_KIND,
            kind,
        )
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, name)

        return_value = await wrapped(*args, **kwargs)

    return return_value
