"""OpenTelemetry OpenAI Agents instrumentation"""

import os
from typing import Collection
from opentelemetry.trace import get_tracer
from opentelemetry.metrics import Meter, get_meter
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.openai_agents.version import __version__
from opentelemetry.semconv_ai import Meters
from opentelemetry._events import get_event_logger
from .config import Config


_instruments = ("openai-agents >= 0.2.0",)


class OpenAIAgentsInstrumentor(BaseInstrumentor):
    """An instrumentor for OpenAI Agents SDK."""

    def __init__(
        self,
        exception_logger=None,
        use_legacy_attributes: bool = True,
    ):
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

        if not Config.use_legacy_attributes:
            event_logger_provider = kwargs.get("event_logger_provider")
            Config.event_logger = get_event_logger(
                __name__, __version__, event_logger_provider=event_logger_provider
            )

        if is_metrics_enabled():
            _create_metrics(meter)

        # Use hook-based approach with OpenAI Agents SDK callbacks
        try:
            from agents import add_trace_processor
            from ._hooks import OpenTelemetryTracingProcessor

            # Create and add our OpenTelemetry processor
            otel_processor = OpenTelemetryTracingProcessor(tracer)
            add_trace_processor(otel_processor)

        except Exception:
            # Silently handle import errors - OpenAI Agents SDK may not be available
            pass

        try:
            from ._realtime_wrappers import wrap_realtime_session

            wrap_realtime_session(tracer)
        except Exception:
            pass

    def _uninstrument(self, **kwargs):
        try:
            from ._realtime_wrappers import unwrap_realtime_session

            unwrap_realtime_session()
        except Exception:
            pass


def is_metrics_enabled() -> bool:
    return (os.getenv("TRACELOOP_METRICS_ENABLED") or "true").lower() == "true"


def _create_metrics(meter: Meter):
    token_histogram = meter.create_histogram(
        name=Meters.LLM_TOKEN_USAGE,
        unit="token",
        description="Measures number of input and output tokens used",
    )

    duration_histogram = meter.create_histogram(
        name=Meters.LLM_OPERATION_DURATION,
        unit="s",
        description="GenAI operation duration",
    )

    return token_histogram, duration_histogram
