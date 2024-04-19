from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY

from opentelemetry.semconv.ai import SpanAttributes, LLMRequestTypeValues

from opentelemetry.instrumentation.langchain.utils import (
    _with_tracer_wrapper,
    dont_throw,
)
from opentelemetry.instrumentation.langchain.utils import should_send_prompts


@_with_tracer_wrapper
def llm_wrapper(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    name = f"{instance.__class__.__name__}.chat"
    with tracer.start_as_current_span(name) as span:
        _handle_request(span, args, kwargs, instance)
        return_value = wrapped(*args, **kwargs)
        _handle_response(span, return_value)

    return return_value


@_with_tracer_wrapper
async def allm_wrapper(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    name = f"{instance.__class__.__name__}.chat"
    with tracer.start_as_current_span(name) as span:
        _handle_request(span, args, kwargs, instance)
        return_value = await wrapped(*args, **kwargs)
        _handle_response(span, return_value)

    return return_value


@dont_throw
def _handle_request(span, args, kwargs, instance):
    span.set_attribute(
        SpanAttributes.LLM_REQUEST_TYPE, LLMRequestTypeValues.COMPLETION.value
    )
    span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, instance.__class__.__name__)

    if should_send_prompts():
        for idx, prompt in enumerate(args[0]):
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{idx}.user", prompt)


@dont_throw
def _handle_response(span, return_value):
    if should_send_prompts():
        for idx, generation in enumerate(return_value.generations):
            span.set_attribute(
                f"{SpanAttributes.LLM_COMPLETIONS}.{idx}.content",
                generation[0].text,
            )
