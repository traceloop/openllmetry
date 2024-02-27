from importlib.metadata import version as package_version, PackageNotFoundError

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.llamaindex.utils import (
    _with_tracer_wrapper,
    start_as_current_span_async,
)
from opentelemetry.semconv.ai import SpanAttributes, TraceloopSpanKindValues


TO_INSTRUMENT = [
    {
        "class": "FunctionTool",
        "v9_module": "llama_index.tools.function_tool",
        "v10_module": "llama_index.core.tools.function_tool",
        "v10_legacy_module": "llama_index.legacy.tools.function_tool",
    },
    {
        "class": "QueryEngineTool",
        "v9_module": "llama_index.tools.query_engine",
        "v10_module": "llama_index.core.tools.query_engine",
        "v10_legacy_module": "llama_index.legacy.tools.query_engine",
    },
]


class BaseToolInstrumentor:
    def __init__(self, tracer):
        self._tracer = tracer

    def instrument(self):
        for module in TO_INSTRUMENT:
            try:
                package_version("llama-index-core")
                self._instrument_module(module["v10_module"], module["class"])
                self._instrument_module(module["v10_legacy_module"], module["class"])

            except PackageNotFoundError:
                self._instrument_module(module["v9_module"], module["class"])

    def _instrument_module(self, module_name, class_name):
        wrap_function_wrapper(
            module_name, f"{class_name}.call", query_wrapper(self._tracer)
        )
        wrap_function_wrapper(
            module_name, f"{class_name}.acall", aquery_wrapper(self._tracer)
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
