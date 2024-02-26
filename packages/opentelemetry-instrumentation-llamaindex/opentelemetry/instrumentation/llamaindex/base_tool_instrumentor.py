from importlib.metadata import version as package_version, PackageNotFoundError

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.llamaindex.utils import (
    _with_tracer_wrapper,
    start_as_current_span_async,
)
from opentelemetry.semconv.ai import SpanAttributes, TraceloopSpanKindValues


V9_MODULE_NAME = "llama_index.tools.types"
V10_MODULE_NAME = "llama_index.core.tools.types"
V10_LEGACY_MODULE_NAME = "llama_index.legacy.tools.types"

CLASS_NAME = "AsyncBaseTool"


class BaseToolInstrumentor:
    def __init__(self, tracer):
        self._tracer = tracer

    def instrument(self):
        try:
            package_version("llama-index-core")
            self._instrument_module(V10_MODULE_NAME)
            self._instrument_module(V10_LEGACY_MODULE_NAME)

        except PackageNotFoundError:
            self._instrument_module(V9_MODULE_NAME)

    def _instrument_module(self, module_name):
        wrap_function_wrapper(
            module_name, f"{CLASS_NAME}.call", query_wrapper(self._tracer)
        )
        wrap_function_wrapper(
            module_name, f"{CLASS_NAME}.acall", aquery_wrapper(self._tracer)
        )


@_with_tracer_wrapper
def query_wrapper(tracer, wrapped, instance, args, kwargs):
    with tracer.start_as_current_span(f"{instance.__class__.__name__}.tool") as span:
        span.set_attribute(
            SpanAttributes.TRACELOOP_SPAN_KIND,
            TraceloopSpanKindValues.TOOL.value,
        )

        return wrapped(*args, **kwargs)


@_with_tracer_wrapper
async def aquery_wrapper(tracer, wrapped, instance, args, kwargs):
    async with start_as_current_span_async(
        tracer=tracer, name=f"{instance.__class__.__name__}.tool"
    ) as span:
        span.set_attribute(
            SpanAttributes.TRACELOOP_SPAN_KIND,
            TraceloopSpanKindValues.TOOL.value,
        )

        return await wrapped(*args, **kwargs)
