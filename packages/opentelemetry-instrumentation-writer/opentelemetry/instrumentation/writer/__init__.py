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
        "package": "writerai.resources.chat",
        "object": "ChatResource",
        "method": "chat",
        "span_name": "writerai.chat",
    },
    {
        "package": "writerai.resources.completions",
        "object": "CompletionsResource",
        "method": "create",
        "span_name": "writerai.completions",
    },
]
WRAPPED_AMETHODS = [
    {
        "package": "writerai.resources.chat",
        "object": "AsyncChatResource",
        "method": "chat",
        "span_name": "writerai.chat",
    },
    {
        "package": "writerai.resources.completions",
        "object": "AsyncCompletionsResource",
        "method": "create",
        "span_name": "writerai.completions",
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

    choice_counter = meter.create_counter(
        name=Meters.LLM_GENERATION_CHOICES,
        unit="choice",
        description="Number of choices returned by chat completions call",
    )

    return (
        token_histogram,
        duration_histogram,
        streaming_time_to_first_token,
        streaming_time_to_generate,
        choice_counter,
    )


def is_metrics_collection_enabled() -> bool:
    return (os.getenv("TRACELOOP_METRICS_ENABLED") or "true").lower() == "true"


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
                choice_counter,
            ) = _build_metrics(meter)
        else:
            (
                token_histogram,
                duration_histogram,
                streaming_time_to_first_token,
                streaming_time_to_generate,
                choice_counter,
            ) = (None, None, None, None, None)

        event_logger = None
        if not Config.use_legacy_attributes:
            event_logger_provider = kwargs.get("event_logger_provider")
            event_logger = get_event_logger(
                __name__, __version__, event_logger_provider=event_logger_provider
            )

        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")

            try:
                wrap_function_wrapper(
                    wrap_package,
                    f"{wrap_object}.{wrap_method}",
                    _wrap(
                        tracer,
                        token_histogram,
                        duration_histogram,
                        streaming_time_to_first_token,
                        streaming_time_to_generate,
                        choice_counter,
                        event_logger,
                        wrapped_method,
                    ),
                )
            except ModuleNotFoundError:
                pass

        for wrapped_method in WRAPPED_AMETHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            try:
                wrap_function_wrapper(
                    wrap_package,
                    f"{wrap_object}.{wrap_method}",
                    _awrap(
                        tracer,
                        token_histogram,
                        duration_histogram,
                        streaming_time_to_first_token,
                        streaming_time_to_generate,
                        choice_counter,
                        event_logger,
                        wrapped_method,
                    ),
                )
            except ModuleNotFoundError:
                pass

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            unwrap(
                f"{wrap_package}.{wrap_object}",
                wrapped_method.get("method"),
            )
        for wrapped_method in WRAPPED_AMETHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            unwrap(
                f"{wrap_package}.{wrap_object}",
                wrapped_method.get("method"),
            )
