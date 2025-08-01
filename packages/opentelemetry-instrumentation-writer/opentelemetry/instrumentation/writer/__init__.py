"""OpenTelemetry Writer instrumentation"""

import logging
import os
from typing import Collection

from opentelemetry._events import get_event_logger
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.metrics import Meter, get_meter
from opentelemetry.semconv._incubating.metrics import \
    gen_ai_metrics as GenAIMetrics
from opentelemetry.semconv_ai import Meters
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.writer.config import Config
from opentelemetry.instrumentation.writer.version import __version__

logger = logging.getLogger(__name__)

_instruments = ("writer-sdk >= 2.2.1, < 3",)

WRAPPED_METHODS = [
    {
        "method": "creations.create",
        "span_name": "writer.completions",
    },
    {
        "method": "chat.chat",
        "span_name": "writer.chat",
    },
]


def _build_metrics(meter: Meter):
    token_histogram = meter.create_histogram(
        name=Meters.LLM_TOKEN_USAGE,
        unit="token",
        description="Measures number of input and output tokens used",
    )

    duration_histogram = meter.create_histogram(
        name=Meters.LLM_OPERATION_DURATION,
        unit="s",
        description="Generation operation duration",
    )

    streaming_time_to_first_token = meter.create_histogram(
        name=GenAIMetrics.GEN_AI_SERVER_TIME_TO_FIRST_TOKEN,
        unit="s",
        description="Time to first token in streaming chat completions",
    )

    streaming_time_to_generate = meter.create_histogram(
        name=Meters.LLM_STREAMING_TIME_TO_GENERATE,
        unit="s",
        description="Time from first token to completion in streaming responses",
    )

    return (
        token_histogram,
        duration_histogram,
        streaming_time_to_first_token,
        streaming_time_to_generate,
    )


def is_metrics_collection_enabled() -> bool:
    return (os.getenv("TRACELOOP_METRICS_ENABLED") or "true").lower() == "true"


def _dispatch_wrap(
    tracer,
    token_histogram,
    duration_histogram,
    event_logger,
    streaming_time_to_first_token,
    streaming_time_to_generate,
):
    def wrapper(wrapped, instance, args, kwargs):
        to_wrap = None
        if len(args) > 2 and isinstance(args[2], str):
            path = args[2]
            op = path.rstrip("/").split("/")[-1]
            to_wrap = next((m for m in WRAPPED_METHODS if m.get("method") == op), None)
        if to_wrap:
            return _wrap(
                tracer,
                token_histogram,
                duration_histogram,
                event_logger,
                streaming_time_to_first_token,
                streaming_time_to_generate,
                to_wrap,
            )(wrapped, instance, args, kwargs)
        return wrapped(*args, **kwargs)

    return wrapper


def _dispatch_awrap(
    tracer,
    token_histogram,
    duration_histogram,
    event_logger,
    streaming_time_to_first_token,
    streaming_time_to_generate,
):
    async def wrapper(wrapped, instance, args, kwargs):
        to_wrap = None
        if len(args) > 2 and isinstance(args[2], str):
            path = args[2]
            op = path.rstrip("/").split("/")[-1]
            to_wrap = next((m for m in WRAPPED_METHODS if m.get("method") == op), None)
        if to_wrap:
            return await _awrap(
                tracer,
                token_histogram,
                duration_histogram,
                event_logger,
                streaming_time_to_first_token,
                streaming_time_to_generate,
                to_wrap,
            )(wrapped, instance, args, kwargs)
        return await wrapped(*args, **kwargs)

    return wrapper


class WriterInstrumentor(BaseInstrumentor):
    """An instrumentor for Writer's client library."""

    def __init__(self, exception_logger=None, use_legacy_attributes=True):
        super().__init__()
        Config.exception_logger = exception_logger
        Config.use_legacy_attributes = use_legacy_attributes

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        meter_provider = kwargs.get("meter_provider")
        meter = get_meter(__name__, __version__, meter_provider)

        if is_metrics_collection_enabled():
            (
                token_histogram,
                duration_histogram,
                streaming_time_to_first_token,
                streaming_time_to_generate,
            ) = _build_metrics(meter)
        else:
            (
                token_histogram,
                duration_histogram,
                streaming_time_to_first_token,
                streaming_time_to_generate,
            ) = (None, None, None, None)

        event_logger = None
        if not Config.use_legacy_attributes:
            event_logger_provider = kwargs.get("event_logger_provider")
            event_logger = get_event_logger(
                __name__, __version__, event_logger_provider=event_logger_provider
            )

        wrap_function_wrapper(
            "writerai",
            "_copy_messages",  # TODO fix path
            _sanitize_copy_messages,
        )

        wrap_function_wrapper(
            "writerai",
            "Writer",  # TODO fix path
            _dispatch_wrap(
                tracer,
                token_histogram,
                duration_histogram,
                event_logger,
                streaming_time_to_first_token,
                streaming_time_to_generate,
            ),
        )
        wrap_function_wrapper(
            "writerai",
            "AsyncWriter",  # TODO fix path
            _dispatch_awrap(
                tracer,
                token_histogram,
                duration_histogram,
                event_logger,
                streaming_time_to_first_token,
                streaming_time_to_generate,
            ),
        )

    def _uninstrument(self, **kwargs):
        try:
            import writerai
            from writerai import AsyncWriter, Writer

            for wrapped_method in WRAPPED_METHODS:
                method_name = wrapped_method.get("method")
                unwrap(Writer, method_name)
                unwrap(AsyncWriter, method_name)
                unwrap(writerai, method_name)
        except ImportError:
            logger.warning("Failed to import writer modules for uninstrumentation.")
