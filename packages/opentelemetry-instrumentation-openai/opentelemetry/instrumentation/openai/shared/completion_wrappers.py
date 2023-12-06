from opentelemetry.semconv.ai import LLMRequestTypeValues

from opentelemetry.instrumentation.openai.utils import _with_tracer_wrapper, start_as_current_span_async

SPAN_NAME = "openai.completion"
LLM_REQUEST_TYPE = LLMRequestTypeValues.COMPLETION.value


@_with_tracer_wrapper
def completion_wrapper(tracer, wrapped, instance, args, kwargs):
    with tracer.start_as_current_span(SPAN_NAME):
        return wrapped(*args, **kwargs)


@_with_tracer_wrapper
async def acompletion_wrapper(tracer, wrapped, instance, args, kwargs):
    async with start_as_current_span_async(tracer=tracer, name=SPAN_NAME):
        return await wrapped(*args, **kwargs)
