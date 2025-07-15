from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.llamaindex.utils import (
    _with_tracer_wrapper,
    process_request,
    process_response,
    start_as_current_span_async,
)
from opentelemetry.semconv_ai import SpanAttributes, TraceloopSpanKindValues

MODULE_NAME = "llama_parse"
CLASS_NAME = "LlamaParse"
TASK_NAME = "llamaparse"


class LlamaParseInstrumentor:
    def __init__(self, tracer):
        self._tracer = tracer

    def instrument(self):
        methods_to_wrap = [
            ("load_data", load_data_wrapper),
            ("aload_data", aload_data_wrapper),
            ("get_json_result", get_json_wrapper),
            ("aget_json", aget_json_wrapper),
            ("get_images", get_images_wrapper),
            ("aget_images", aget_images_wrapper),
            ("get_charts", get_charts_wrapper),
            ("aget_charts", aget_charts_wrapper),
        ]

        for method_name, wrapper_func in methods_to_wrap:
            try:
                wrap_function_wrapper(
                    MODULE_NAME,
                    f"{CLASS_NAME}.{method_name}",
                    wrapper_func(self._tracer),
                )
            except AttributeError:
                # Method doesn't exist, skip it
                continue


@_with_tracer_wrapper
def get_json_wrapper(tracer, wrapped, instance, args, kwargs):
    with tracer.start_as_current_span(f"{TASK_NAME}.task") as span:
        span.set_attribute(
            SpanAttributes.TRACELOOP_SPAN_KIND,
            TraceloopSpanKindValues.TASK.value,
        )
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, TASK_NAME)

        process_request(span, args, kwargs)
        result = wrapped(*args, **kwargs)
        process_response(span, result)
        return result


@_with_tracer_wrapper
async def aget_json_wrapper(tracer, wrapped, instance, args, kwargs):
    async with start_as_current_span_async(
        tracer=tracer, name=f"{TASK_NAME}.task"
    ) as span:
        span.set_attribute(
            SpanAttributes.TRACELOOP_SPAN_KIND,
            TraceloopSpanKindValues.TASK.value,
        )
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, TASK_NAME)

        process_request(span, args, kwargs)
        result = await wrapped(*args, **kwargs)
        process_response(span, result)
        return result


@_with_tracer_wrapper
def get_images_wrapper(tracer, wrapped, instance, args, kwargs):
    with tracer.start_as_current_span(f"{TASK_NAME}.task") as span:
        span.set_attribute(
            SpanAttributes.TRACELOOP_SPAN_KIND,
            TraceloopSpanKindValues.TASK.value,
        )
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, TASK_NAME)

        process_request(span, args, kwargs)
        result = wrapped(*args, **kwargs)
        process_response(span, result)
        return result


@_with_tracer_wrapper
async def aget_images_wrapper(tracer, wrapped, instance, args, kwargs):
    async with start_as_current_span_async(
        tracer=tracer, name=f"{TASK_NAME}.task"
    ) as span:
        span.set_attribute(
            SpanAttributes.TRACELOOP_SPAN_KIND,
            TraceloopSpanKindValues.TASK.value,
        )
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, TASK_NAME)

        process_request(span, args, kwargs)
        result = await wrapped(*args, **kwargs)
        process_response(span, result)
        return result


@_with_tracer_wrapper
def get_charts_wrapper(tracer, wrapped, instance, args, kwargs):
    with tracer.start_as_current_span(f"{TASK_NAME}.task") as span:
        span.set_attribute(
            SpanAttributes.TRACELOOP_SPAN_KIND,
            TraceloopSpanKindValues.TASK.value,
        )
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, TASK_NAME)

        process_request(span, args, kwargs)
        result = wrapped(*args, **kwargs)
        process_response(span, result)
        return result


@_with_tracer_wrapper
async def aget_charts_wrapper(tracer, wrapped, instance, args, kwargs):
    async with start_as_current_span_async(
        tracer=tracer, name=f"{TASK_NAME}.task"
    ) as span:
        span.set_attribute(
            SpanAttributes.TRACELOOP_SPAN_KIND,
            TraceloopSpanKindValues.TASK.value,
        )
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, TASK_NAME)

        process_request(span, args, kwargs)
        result = await wrapped(*args, **kwargs)
        process_response(span, result)
        return result


@_with_tracer_wrapper
def load_data_wrapper(tracer, wrapped, instance, args, kwargs):
    with tracer.start_as_current_span(f"{TASK_NAME}.task") as span:
        span.set_attribute(
            SpanAttributes.TRACELOOP_SPAN_KIND,
            TraceloopSpanKindValues.TASK.value,
        )
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, TASK_NAME)

        process_request(span, args, kwargs)
        result = wrapped(*args, **kwargs)
        process_response(span, result)
        return result


@_with_tracer_wrapper
async def aload_data_wrapper(tracer, wrapped, instance, args, kwargs):
    async with start_as_current_span_async(
        tracer=tracer, name=f"{TASK_NAME}.task"
    ) as span:
        span.set_attribute(
            SpanAttributes.TRACELOOP_SPAN_KIND,
            TraceloopSpanKindValues.TASK.value,
        )
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, TASK_NAME)

        process_request(span, args, kwargs)
        result = await wrapped(*args, **kwargs)
        process_response(span, result)
        return result
