import os

from opentelemetry import context as context_api

from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv.ai import SpanAttributes, TraceloopSpanKindValues, LLMRequestTypeValues
from opentelemetry.instrumentation.llamaindex.utils import _with_tracer_wrapper


def should_send_prompts():
    return (
        os.getenv("TRACELOOP_TRACE_CONTENT") or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


def _set_input_attributes(span, llm_request_type, args, kwargs, instance):
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, instance.metadata.model_name)
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, instance.metadata.context_window
    )
    _set_span_attribute(span, SpanAttributes.LLM_TEMPERATURE, instance.temperature)
    _set_span_attribute(span, SpanAttributes.LLM_TOP_P, instance.metadata.num_output)

    if should_send_prompts():
        if llm_request_type == LLMRequestTypeValues.COMPLETION:
            if len(args) > 0:
                prompt = args[0]
                _set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.0.user",
                    prompt[0] if isinstance(prompt, list) else prompt,
                )

    return


def _set_response_attributes(span, llm_request_type, response):
    if should_send_prompts():
        if llm_request_type == LLMRequestTypeValues.COMPLETION:
            _set_span_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content", response.text)

    return


@_with_tracer_wrapper
def custom_llm_wrapper(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    kind = to_wrap.get("kind") or TraceloopSpanKindValues.TASK.value
    with tracer.start_as_current_span(name) as span:
        span.set_attribute(SpanAttributes.LLM_VENDOR, "Ollama")
        span.set_attribute(SpanAttributes.TRACELOOP_SPAN_KIND, kind)
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, name)
        llm_request_type = LLMRequestTypeValues.CHAT
        if to_wrap.get("method") == "complete" or to_wrap.get("method") == "acomplete":
            llm_request_type = LLMRequestTypeValues.COMPLETION
        span.set_attribute(SpanAttributes.LLM_REQUEST_TYPE, llm_request_type.value)

        _set_input_attributes(span, llm_request_type, args, kwargs, instance)

        return_value = wrapped(*args, **kwargs)

        _set_response_attributes(span, llm_request_type, return_value)

    return return_value
