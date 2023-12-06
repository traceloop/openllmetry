from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.llamaindex.utils import _with_tracer_wrapper, start_as_current_span_async
from opentelemetry.semconv.ai import SpanAttributes, TraceloopSpanKindValues

MODULE_NAME = "llama_index.response_synthesizers"
CLASS_NAME = "BaseSynthesizer"
TASK_NAME = "synthesize"


class BaseSynthesizerInstrumentor:
    def __init__(self, tracer):
        self._tracer = tracer

    def instrument(self):
        wrap_function_wrapper(MODULE_NAME, f"{CLASS_NAME}.synthesize", synthesize_wrapper(self._tracer))
        wrap_function_wrapper(MODULE_NAME, f"{CLASS_NAME}.asynthesize", asynthesize_wrapper(self._tracer))


@_with_tracer_wrapper
def synthesize_wrapper(tracer, wrapped, instance, args, kwargs):
    with tracer.start_as_current_span(f"{TASK_NAME}.task") as span:
        span.set_attribute(
            SpanAttributes.TRACELOOP_SPAN_KIND,
            TraceloopSpanKindValues.TASK.value,
        )

        return wrapped(*args, **kwargs)


@_with_tracer_wrapper
async def asynthesize_wrapper(tracer, wrapped, instance, args, kwargs):
    async with start_as_current_span_async(tracer=tracer, name=f"{TASK_NAME}.task") as span:
        span.set_attribute(
            SpanAttributes.TRACELOOP_SPAN_KIND,
            TraceloopSpanKindValues.TASK.value,
        )

        return await wrapped(*args, **kwargs)
