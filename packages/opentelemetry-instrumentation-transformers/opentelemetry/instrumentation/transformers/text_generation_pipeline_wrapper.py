import logging

from opentelemetry import context as context_api
from opentelemetry.instrumentation.transformers.event_emitter import (
    emit_prompt_events,
    emit_response_events,
)
from opentelemetry.instrumentation.transformers.span_utils import (
    set_input_attributes,
    set_model_input_attributes,
    set_response_attributes,
)
from opentelemetry.instrumentation.transformers.utils import (
    dont_throw,
    should_emit_events,
    with_tracer_wrapper,
)
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
)
from opentelemetry.trace import Status, StatusCode

import transformers

logger = logging.getLogger(__name__)


@dont_throw
def _handle_input(span, event_logger, instance, args, kwargs):
    set_model_input_attributes(span, instance)
    if should_emit_events():
        emit_prompt_events(args, kwargs, event_logger)
    else:
        set_input_attributes(span, instance, args, kwargs)


@dont_throw
def _handle_response(span, event_logger, response):
    if should_emit_events():
        emit_response_events(response, event_logger)
    else:
        set_response_attributes(span, response)  # noqa: F821


@with_tracer_wrapper
def text_generation_pipeline_wrapper(
    tracer, event_logger, to_wrap, wrapped, instance, args, kwargs
):
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
        _handle_input(span, event_logger, instance, args, kwargs)

        response = wrapped(*args, **kwargs)

        if response:
            _handle_response(span, event_logger, response)
            if span.is_recording():
                span.set_status(Status(StatusCode.OK))

        return response
