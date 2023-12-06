from opentelemetry.semconv.ai import SpanAttributes, TraceloopSpanKindValues, LLMRequestTypeValues

from opentelemetry.instrumentation.openai.utils import _with_tracer_wrapper, start_as_current_span_async

SPAN_NAME = "openai.chat"
LLM_REQUEST_TYPE = LLMRequestTypeValues.CHAT.value


@_with_tracer_wrapper
def chat_wrapper(tracer, wrapped, instance, args, kwargs):
    with tracer.start_as_current_span(SPAN_NAME) as span:
        return wrapped(*args, **kwargs)


@_with_tracer_wrapper
async def achat_wrapper(tracer, wrapped, instance, args, kwargs):
    async with start_as_current_span_async(tracer=tracer, name=SPAN_NAME) as span:
        return await wrapped(*args, **kwargs)
