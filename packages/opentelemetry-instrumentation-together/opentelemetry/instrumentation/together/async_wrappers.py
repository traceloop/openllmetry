from opentelemetry import context as context_api
from opentelemetry.instrumentation.together.safety import (
    _apply_completion_safety,
    _apply_prompt_safety,
)
from opentelemetry.instrumentation.together.streaming_safety import (
    build_async_streaming_response,
)
from opentelemetry.instrumentation.together.utils import dont_throw
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    LLMRequestTypeValues,
    SpanAttributes,
)
from opentelemetry.trace import SpanKind
from opentelemetry.trace.status import Status, StatusCode


WRAPPED_AMETHODS = [
    {
        "object": "resources.chat.completions",
        "method": "AsyncChatCompletions.create",
        "span_name": "together.chat",
    },
    {
        "object": "resources.completions",
        "method": "AsyncCompletions.create",
        "span_name": "together.completion",
    },
]


def _with_async_tracer_wrapper(func):
    def _with_tracer(tracer, event_logger, to_wrap):
        async def wrapper(wrapped, instance, args, kwargs):
            return await func(
                tracer, event_logger, to_wrap, wrapped, instance, args, kwargs
            )

        return wrapper

    return _with_tracer


def _llm_request_type_by_method(method_name):
    if method_name == "AsyncChatCompletions.create":
        return LLMRequestTypeValues.CHAT
    if method_name == "AsyncCompletions.create":
        return LLMRequestTypeValues.COMPLETION
    return LLMRequestTypeValues.UNKNOWN


@_with_async_tracer_wrapper
@dont_throw
async def _awrap(
    tracer,
    event_logger,
    to_wrap,
    wrapped,
    instance,
    args,
    kwargs,
):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return await wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    llm_request_type = _llm_request_type_by_method(to_wrap.get("method"))
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            GenAIAttributes.GEN_AI_SYSTEM: "TogetherAI",
            SpanAttributes.LLM_REQUEST_TYPE: llm_request_type.value,
        },
    )
    kwargs = _apply_prompt_safety(span, kwargs, llm_request_type, name)
    from opentelemetry.instrumentation.together import _handle_input, _handle_response

    _handle_input(span, event_logger, llm_request_type, kwargs)

    response = await wrapped(*args, **kwargs)

    if kwargs.get("stream") and response:
        return build_async_streaming_response(
            response,
            span=span,
            event_logger=event_logger,
            llm_request_type=llm_request_type,
            span_name=name,
            handle_response=_handle_response,
        )

    if response:
        _apply_completion_safety(span, response, llm_request_type, name)
        _handle_response(span, event_logger, llm_request_type, response)
        if span.is_recording():
            span.set_status(Status(StatusCode.OK))

    span.end()
    return response
