import logging
import os
from typing import Collection
from opentelemetry.instrumentation.alephalpha.config import Config
from opentelemetry.instrumentation.alephalpha.utils import dont_throw
from wrapt import wrap_function_wrapper

from opentelemetry import context as context_api
from opentelemetry.trace import get_tracer, SpanKind
from opentelemetry.trace.status import Status, StatusCode

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)

from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
    LLMRequestTypeValues,
)
from opentelemetry.instrumentation.alephalpha.version import __version__

logger = logging.getLogger(__name__)

# Constants to avoid string literal typos
ALEPH_ALPHA_CLIENT = "aleph_alpha_client"
ALEPH_ALPHA_CLIENT_VERSION = ">= 7.1.0, <8"
ALEPH_ALPHA_COMPLETION_METHOD = "complete"

_instruments = (f"{ALEPH_ALPHA_CLIENT} {ALEPH_ALPHA_CLIENT_VERSION}",)

WRAPPED_METHODS = [
    {
        "method": ALEPH_ALPHA_COMPLETION_METHOD,
        "span_name": "alephalpha.completion",
    },
]

def should_send_prompts():
    # Using a default "false" to avoid always-True evaluation due to the fallback
    return os.getenv("TRACELOOP_TRACE_CONTENT", "false").lower() == "true" or \
           context_api.get_value("override_enable_content_tracing")

def _set_span_attribute(span, name, value):
    if value:
        span.set_attribute(name, value)

@dont_throw
def _set_input_attributes(span, llm_request_type, args, kwargs):
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("model"))

    if should_send_prompts() and llm_request_type == LLMRequestTypeValues.COMPLETION:
        _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")
        _set_span_attribute(
            span,
            f"{SpanAttributes.LLM_PROMPTS}.0.content",
            getattr(args[0].prompt.items[0], 'text', '')
        )

@dont_throw
def _set_response_attributes(span, llm_request_type, response):
    if should_send_prompts() and llm_request_type == LLMRequestTypeValues.COMPLETION:
        _set_span_attribute(
            span,
            f"{SpanAttributes.LLM_COMPLETIONS}.0.content",
            getattr(response.completions[0], 'completion', '')
        )
        _set_span_attribute(
            span, f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant"
        )

    input_tokens = getattr(response, "num_tokens_prompt_total", 0)
    output_tokens = getattr(response, "num_tokens_generated", 0)

    _set_span_attribute(span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, input_tokens + output_tokens)
    _set_span_attribute(span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, output_tokens)
    _set_span_attribute(span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, input_tokens)

def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer

def _llm_request_type_by_method(method_name):
    if method_name == ALEPH_ALPHA_COMPLETION_METHOD:
        return LLMRequestTypeValues.COMPLETION
    return LLMRequestTypeValues.UNKNOWN

@_with_tracer_wrapper
def _wrap(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    llm_request_type = _llm_request_type_by_method(to_wrap.get("method"))
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "AlephAlpha",
            SpanAttributes.LLM_REQUEST_TYPE: llm_request_type.value,
        },
    )
    if span.is_recording():
        _set_input_attributes(span, llm_request_type, args, kwargs)

    response = wrapped(*args, **kwargs)

    if response and span.is_recording():
        _set_response_attributes(span, llm_request_type, response)
        span.set_status(Status(StatusCode.OK))

    span.end()
    return response

class AlephAlphaInstrumentor(BaseInstrumentor):
    """An instrumentor for Aleph Alpha's client library."""

    def __init__(self, exception_logger=None):
        super().__init__()
        Config.exception_logger = exception_logger

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)
        for wrapped_method in WRAPPED_METHODS:
            wrap_method = wrapped_method.get("method")
            wrap_function_wrapper(
                ALEPH_ALPHA_CLIENT,
                f"Client.{wrap_method}",
                _wrap(tracer, wrapped_method),
            )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            unwrap(
                f"{ALEPH_ALPHA_CLIENT}.Client",
                wrapped_method.get("method"),
            )
