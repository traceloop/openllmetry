import logging

from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)
from opentelemetry.trace import Status, StatusCode
from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.instrumentation.transformers.utils import (
    _with_tracer_wrapper,
    dont_throw,
)
from opentelemetry.instrumentation.transformers.events import (
    prompt_to_event,
    completion_to_event,
)
import transformers

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


@dont_throw
def _set_input_attributes(span, instance, args, kwargs):
    forward_params = instance._forward_params
    if args and len(args) > 0:
        prompts_list = args[0]
    else:
        prompts_list = kwargs.get("args")

    _set_span_prompts(span, prompts_list)
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_MODEL, instance.model.config.name_or_path
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_SYSTEM, instance.model.config.model_type
    )
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_TYPE, "completion")
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_TEMPERATURE, forward_params.get("temperature")
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_TOP_P, forward_params.get("top_p")
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, forward_params.get("max_length")
    )
    _set_span_attribute(
        span,
        SpanAttributes.LLM_REQUEST_REPETITION_PENALTY,
        forward_params.get("repetition_penalty"),
    )


def _set_span_completions(span, completions):
    if completions is None:
        return

    for i, completion in enumerate(completions):
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{i}"
        _set_span_attribute(span, f"{prefix}.content", completion.get("generated_text"))


@dont_throw
def _set_response_attributes(span, response):
    if response:
        if len(response) > 0:
            _set_span_completions(span, response[0])


@_with_tracer_wrapper
def text_generation_pipeline_wrapper(tracer, event_logger, config, to_wrap):
    """Wrap the text generation pipeline to add instrumentation."""

    def wrapper(wrapped, instance, args, kwargs):
        if "TextGenerationPipeline" not in dir(transformers) or not isinstance(
            instance, transformers.TextGenerationPipeline
        ):
            return wrapped(*args, **kwargs)

        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
            SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
        ):
            return wrapped(*args, **kwargs)

        name = to_wrap.get("span_name")
        with tracer.start_as_current_span(name) as span:
            # Get prompt from args or kwargs
            if args and len(args) > 0:
                prompt = args[0]
            else:
                prompt = kwargs.get("args")

            model_name = instance.model.config.name_or_path
            
            # Always set basic model info on span
            if span.is_recording():
                _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, model_name)
                _set_span_attribute(span, SpanAttributes.LLM_SYSTEM, instance.model.config.model_type)
                _set_span_attribute(span, SpanAttributes.LLM_REQUEST_TYPE, "completion")

            if config.use_legacy_attributes:
                if span.is_recording():
                    _set_input_attributes(span, instance, args, kwargs)
            else:
                # Emit prompt event
                event_logger.emit(
                    prompt_to_event(
                        prompt=prompt,
                        model_name=model_name,
                        capture_content=config.capture_content,
                    )
                )

            try:
                response = wrapped(*args, **kwargs)

                if response:
                    if span.is_recording():
                        if config.use_legacy_attributes:
                            _set_response_attributes(span, response)
                        else:
                            # Emit completion event
                            event_logger.emit(
                                completion_to_event(
                                    completion=response[0] if response else {},
                                    model_name=model_name,
                                    capture_content=config.capture_content,
                                )
                            )
                        span.set_status(Status(StatusCode.OK))

                return response

            except Exception as error:
                if config.exception_logger:
                    config.exception_logger(error)
                if span.is_recording():
                    span.set_status(Status(StatusCode.ERROR))
                    span.record_exception(error)
                raise

    return wrapper
