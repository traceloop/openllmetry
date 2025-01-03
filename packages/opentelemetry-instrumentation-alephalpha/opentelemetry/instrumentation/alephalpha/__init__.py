"""OpenTelemetry Aleph Alpha instrumentation"""

import logging
import os
from typing import Collection
from opentelemetry.instrumentation.alephalpha.config import Config
from opentelemetry.instrumentation.alephalpha.utils import (
    dont_throw,
    get_llm_request_attributes,
    message_to_event,
    completion_to_event,
    set_span_attribute,
    handle_span_exception,
    CompletionBuffer,
)
from wrapt import wrap_function_wrapper

from opentelemetry import context as context_api
from opentelemetry.trace import get_tracer, SpanKind, Status, StatusCode
from opentelemetry._events import EventLogger

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)

from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    GenAIAttributes,
    LLMRequestTypeValues,
)
from opentelemetry.instrumentation.alephalpha.version import __version__

logger = logging.getLogger(__name__)

_instruments = ("aleph_alpha_client >= 7.1.0, <8",)

WRAPPED_METHODS = [
    {
        "method": "complete",
        "span_name": "alephalpha.completion",
    },
]

def should_send_prompts():
    return (
        os.getenv("TRACELOOP_TRACE_CONTENT") or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")

class StreamWrapper:
    """Wrapper for streaming responses."""
    def __init__(self, stream, span, event_logger: EventLogger, capture_content: bool):
        self.stream = stream
        self.span = span
        self.event_logger = event_logger
        self.capture_content = capture_content
        self.completion_buffer = CompletionBuffer(0)
        self._span_started = False
        self.setup()

    def setup(self):
        if not self._span_started:
            self._span_started = True

    def cleanup(self):
        if self._span_started:
            if self.completion_buffer.text_content:
                # Emit completion event with buffered content
                self.event_logger.emit(
                    completion_to_event(
                        self.completion_buffer.get_content(),
                        self.capture_content
                    )
                )
            self.span.end()
            self._span_started = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self.stream)
            if chunk.completions:
                self.completion_buffer.append_content(chunk.completions[0].completion)
            return chunk
        except StopIteration:
            self.cleanup()
            raise
        except Exception as error:
            handle_span_exception(self.span, error)
            raise

def _wrap(tracer, event_logger: EventLogger, capture_content: bool):
    """Instruments and calls every function defined in TO_WRAP."""
    def wrapper(wrapped, instance, args, kwargs):
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
            SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
        ):
            return wrapped(*args, **kwargs)

        span_attributes = get_llm_request_attributes(kwargs)
        with tracer.start_as_current_span(
            name="alephalpha.completion",
            kind=SpanKind.CLIENT,
            attributes=span_attributes,
            end_on_exit=False,
        ) as span:
            try:
                if span.is_recording():
                    # Emit prompt event
                    if should_send_prompts():
                        prompt_text = args[0].prompt.items[0].text
                        event_logger.emit(
                            message_to_event(prompt_text, capture_content)
                        )

                result = wrapped(*args, **kwargs)

                # Handle streaming responses
                if kwargs.get("stream", False):
                    return StreamWrapper(result, span, event_logger, capture_content)

                if span.is_recording() and should_send_prompts():
                    # Emit completion event
                    completion_text = result.completions[0].completion
                    event_logger.emit(
                        completion_to_event(completion_text, capture_content)
                    )

                    # Set usage attributes
                    input_tokens = getattr(result, "num_tokens_prompt_total", 0)
                    output_tokens = getattr(result, "num_tokens_generated", 0)
                    set_span_attribute(
                        span,
                        GenAIAttributes.GEN_AI_USAGE_TOTAL_TOKENS,
                        input_tokens + output_tokens,
                    )
                    set_span_attribute(
                        span,
                        GenAIAttributes.GEN_AI_USAGE_COMPLETION_TOKENS,
                        output_tokens,
                    )
                    set_span_attribute(
                        span,
                        GenAIAttributes.GEN_AI_USAGE_PROMPT_TOKENS,
                        input_tokens,
                    )

                span.set_status(Status(StatusCode.OK))
                span.end()
                return result

            except Exception as error:
                handle_span_exception(span, error)
                raise

    return wrapper

class AlephAlphaInstrumentor(BaseInstrumentor):
    """An instrumentor for Aleph Alpha's client library."""

    def __init__(self, exception_logger=None):
        super().__init__()
        Config.exception_logger = exception_logger
        Config.use_legacy_attributes = True

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        Config.use_legacy_attributes = kwargs.get("use_legacy_attributes", True)
        
        tracer = get_tracer(__name__, __version__, tracer_provider)
        event_logger = EventLogger(
            __name__,
            __version__,
            tracer_provider=tracer_provider,
        )
        
        capture_content = kwargs.get("capture_content", True)
        
        for wrapped_method in WRAPPED_METHODS:
            wrap_method = wrapped_method.get("method")
            wrap_function_wrapper(
                "aleph_alpha_client",
                f"Client.{wrap_method}",
                _wrap(tracer, event_logger, capture_content),
            )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_object = wrapped_method.get("object")
            unwrap(
                f"aleph_alpha_client.Client.{wrap_object}",
                wrapped_method.get("method"),
            )
