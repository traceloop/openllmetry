import logging

from opentelemetry.semconv.ai import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY
)
from opentelemetry.instrumentation.transformers.utils import _with_tracer_wrapper
from transformers import TextGenerationPipeline

logger = logging.getLogger(__name__)


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


def _set_span_prompts(span, messages):
    if messages is None:
        return

    if isinstance(messages, str):
        messages = [messages]

    for i, msg in enumerate(messages):
        prefix = f"{SpanAttributes.LLM_PROMPTS}.{i}"
        _set_span_attribute(span, f"{prefix}.content", msg)


def _set_input_attributes(span, instance, args, kwargs):
    forward_params = instance._forward_params
    if args and len(args) > 0:
        prompts_list = args[0]
    else:
        prompts_list = kwargs.get("args")

    _set_span_prompts(span, prompts_list)
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, instance.model.config.name_or_path)
    _set_span_attribute(span, SpanAttributes.LLM_VENDOR, instance.model.config.model_type)
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_TYPE, "completion")
    _set_span_attribute(span, SpanAttributes.LLM_TEMPERATURE, forward_params.get("temperature"))
    _set_span_attribute(span, SpanAttributes.LLM_TOP_P, forward_params.get("top_p"))
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, forward_params.get("max_length"))
    _set_span_attribute(span, "llm.request.repetition_penalty", forward_params.get("repetition_penalty"))

    return


def _set_span_completions(span, completions):
    if completions is None:
        return

    for i, completion in enumerate(completions):
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{i}"
        _set_span_attribute(span, f"{prefix}.content", completion.get("generated_text"))


def _set_response_attributes(span, response):
    if span.is_recording() and response:
        if len(response) > 0:
            _set_span_completions(span, response[0])

    return


@_with_tracer_wrapper
def text_generation_pipeline_wrapper(tracer, to_wrap, wrapped, instance, args, kwargs):
    if not isinstance(instance, TextGenerationPipeline):
        return wrapped(*args, **kwargs)

    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    with tracer.start_as_current_span(name) as span:
        if span.is_recording():
            try:
                _set_input_attributes(span, instance, args, kwargs)

            except Exception as ex:  # pylint: disable=broad-except
                logger.warning(
                    "Failed to set input attributes for transformers span, error: %s", str(ex)
                )

        response = wrapped(*args, **kwargs)

        if response:
            try:
                if span.is_recording():
                    _set_response_attributes(span, response)

            except Exception as ex:  # pylint: disable=broad-except
                logger.warning(
                    "Failed to set response attributes for transformers span, error: %s",
                    str(ex),
                )
            if span.is_recording():
                span.set_status(Status(StatusCode.OK))

        return response
