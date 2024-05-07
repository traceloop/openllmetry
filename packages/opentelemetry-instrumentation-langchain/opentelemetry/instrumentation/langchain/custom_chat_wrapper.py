import json
from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY

from opentelemetry.semconv.ai import SpanAttributes, LLMRequestTypeValues

from opentelemetry.instrumentation.langchain.utils import (
    _with_tracer_wrapper,
    dont_throw,
)
from opentelemetry.instrumentation.langchain.utils import should_send_prompts


@_with_tracer_wrapper
def chat_wrapper(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    name = f"{instance.__class__.__name__}.langchain.task"
    with tracer.start_as_current_span(name) as span:
        _handle_request(span, args, kwargs, instance)
        return_value = wrapped(*args, **kwargs)
        _handle_response(span, return_value)

    return return_value


@_with_tracer_wrapper
async def achat_wrapper(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    name = f"{instance.__class__.__name__}.langchain.task"
    with tracer.start_as_current_span(name) as span:
        _handle_request(span, args, kwargs, instance)
        return_value = await wrapped(*args, **kwargs)
        _handle_response(span, return_value)

    return return_value


@dont_throw
def _handle_request(span, args, kwargs, instance):
    if hasattr(instance, "model"):
        model = instance.model
    elif hasattr(instance, "model_name"):
        model = instance.model_name
    elif hasattr(instance, "model_id"):
        model = instance.model_id
    else:
        model = "unknown"
    span.set_attribute(SpanAttributes.LLM_REQUEST_TYPE, LLMRequestTypeValues.CHAT.value)
    span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, model)
    span.set_attribute(SpanAttributes.LLM_RESPONSE_MODEL, model)

    if should_send_prompts():
        messages = args[0] if len(args) > 0 else kwargs.get("messages", [])
        for idx, prompt in enumerate(messages[0]):
            span.set_attribute(
                f"{SpanAttributes.LLM_PROMPTS}.{idx}.role",
                "user" if prompt.type == "human" else prompt.type,
            )
            if isinstance(prompt.content, list):
                span.set_attribute(
                    f"{SpanAttributes.LLM_PROMPTS}.{idx}.content",
                    json.dumps(prompt.content),
                )
            else:
                span.set_attribute(
                    f"{SpanAttributes.LLM_PROMPTS}.{idx}.content", prompt.content
                )


@dont_throw
def _handle_response(span, return_value):
    if should_send_prompts():
        for idx, generation in enumerate(return_value.generations):
            span.set_attribute(
                f"{SpanAttributes.LLM_COMPLETIONS}.{idx}.content",
                generation[0].text,
            )
