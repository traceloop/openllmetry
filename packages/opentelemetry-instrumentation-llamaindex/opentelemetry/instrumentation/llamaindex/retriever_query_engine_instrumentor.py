from wrapt import wrap_function_wrapper
from opentelemetry.context import attach, set_value

from opentelemetry.instrumentation.llamaindex.utils import _with_tracer_wrapper, start_as_current_span_async
from opentelemetry.semconv.ai import SpanAttributes, TraceloopSpanKindValues

MODULE_NAME = "llama_index.query_engine.retriever_query_engine"
CLASS_NAME = "RetrieverQueryEngine"
WORKFLOW_NAME = "llama_index_retriever_query"


class RetrieverQueryEngineInstrumentor:
    def __init__(self, tracer):
        self._tracer = tracer

    def instrument(self):
        wrap_function_wrapper(MODULE_NAME, f"{CLASS_NAME}.query", query_wrapper(self._tracer))
        wrap_function_wrapper(MODULE_NAME, f"{CLASS_NAME}.aquery", aquery_wrapper(self._tracer))


def set_workflow_context():
    attach(set_value("workflow_name", WORKFLOW_NAME))


@_with_tracer_wrapper
def query_wrapper(tracer, wrapped, instance, args, kwargs):
    set_workflow_context()

    with tracer.start_as_current_span(f"{WORKFLOW_NAME}.workflow") as span:
        span.set_attribute(
            SpanAttributes.TRACELOOP_SPAN_KIND,
            TraceloopSpanKindValues.WORKFLOW.value,
        )

        return wrapped(*args, **kwargs)


@_with_tracer_wrapper
async def aquery_wrapper(tracer, wrapped, instance, args, kwargs):
    set_workflow_context()

    async with start_as_current_span_async(tracer=tracer, name=f"{WORKFLOW_NAME}.workflow") as span:
        span.set_attribute(
            SpanAttributes.TRACELOOP_SPAN_KIND,
            TraceloopSpanKindValues.WORKFLOW.value,
        )

        return await wrapped(*args, **kwargs)
