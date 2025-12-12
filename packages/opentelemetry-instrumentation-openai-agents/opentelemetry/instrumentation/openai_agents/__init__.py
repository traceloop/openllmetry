"""OpenTelemetry OpenAI Agents instrumentation"""

import os
from typing import Collection
from opentelemetry.trace import get_tracer
from opentelemetry.metrics import Meter, get_meter
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.openai_agents.version import __version__
from opentelemetry.semconv_ai import Meters


_instruments = ("openai-agents >= 0.2.0",)


class OpenAIAgentsInstrumentor(BaseInstrumentor):
    """An instrumentor for OpenAI Agents SDK."""

    def __init__(self, *, clear: bool = False) -> None:
        """Initialize the instrumentor.

        Args:
            clear: Optional bool and default to False.
                If True, existing trace processors are dropped
                and this instrumentor's processor is set as the only one.
                This is useful for replacing the default OpenAI instrumentation
                (enabled by default) with this one.
        """
        # base class BaseInstrumentor is an ABC without __init__
        self._clear: bool = clear

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs) -> None:
        """Instruments OpenAI Agents SDK.

        Args:
            tracer_provider: An optional TracerProvider to use
                when creating a Tracer.
            meter_provider: An optional MeterProvider to use
                when creating a Meter.

            Additional kwargs are ignored.
        """
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        meter_provider = kwargs.get("meter_provider")
        meter = get_meter(__name__, __version__, meter_provider)

        if is_metrics_enabled():
            _create_metrics(meter)

        # Use hook-based approach with OpenAI Agents SDK callbacks
        try:
            from agents import add_trace_processor, set_trace_processors
            from ._hooks import OpenTelemetryTracingProcessor

            # Create and add our OpenTelemetry processor
            otel_processor = OpenTelemetryTracingProcessor(tracer)
            if self._clear:
                set_trace_processors([otel_processor])
            else:
                add_trace_processor(otel_processor)

        except Exception:
            # Silently handle import errors - OpenAI Agents SDK may not be available
            pass

    def _uninstrument(self, **kwargs):
        # Hook-based approach: cleanup happens automatically when processors are removed
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
