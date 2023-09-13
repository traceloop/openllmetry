import logging
from typing import Collection
from wrapt import wrap_function_wrapper
import openai

from opentelemetry import context as context_api
from opentelemetry.trace import get_tracer, SpanKind
from opentelemetry.trace.status import Status, StatusCode

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)
from packages.opentelemetry-instrumentation-haystack.opentelemetry.instrumentation.haystack import _with_tracer_wrapper


@_with_tracer_wrapper
def _wrap(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    with tracer.start_as_current_span(
        name,
        kind=SpanAttributes.TRACELOOP_SPAN_KIND,
        attributes={
            SpanAttributes.LLM_VENDOR: "OpenAI",
            SpanAttributes.LLM_REQUEST_TYPE: llm_request_type.value,
        },
    ) as span:
        try:
            if span.is_recording():
                _set_input_attributes(span, llm_request_type, kwargs)

        except Exception as ex:  # pylint: disable=broad-except
            logger.warning(
                "Failed to set input attributes for openai span, error: %s", str(ex)
            )

        response = wrapped(*args, **kwargs)

        if response:
            try:
                if span.is_recording():
                    _set_response_attributes(span, llm_request_type, response)

            except Exception as ex:  # pylint: disable=broad-except
                logger.warning(
                    "Failed to set response attributes for openai span, error: %s",
                    str(ex),
                )
            if span.is_recording():
                span.set_status(Status(StatusCode.OK))

        return response
