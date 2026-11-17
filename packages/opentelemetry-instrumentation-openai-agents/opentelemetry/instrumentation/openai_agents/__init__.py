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

    # Class-level: BaseInstrumentor is a singleton whose __init__ re-runs on
    # every instantiation; instance defaults here would be wiped between
    # instrument() and uninstrument().
    _otel_processor = None
    _replaced_processors = None

    def __init__(self, *, replace_existing_processors: bool = False) -> None:
        """Initialize the instrumentor.

        Args:
            replace_existing_processors:
                If enabled, any existing trace processors
                will be cleared before this processor is added as the only one.
                By default, the openai-agents library has
                a built-in tracing processor that is always active,
                and this processor will be added alongside it,
                resulting in traces being sent to multiple backends.
        """
        super().__init__()
        self._replace_existing_processors: bool = replace_existing_processors

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs) -> None:
        """Override the abstract base method and instrument openai-agents.

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
            if self._replace_existing_processors:
                # Save the processors we replace so uninstrument() can restore them
                self._replaced_processors = _current_trace_processors()
                set_trace_processors([otel_processor])
            else:
                add_trace_processor(otel_processor)
            self._otel_processor = otel_processor

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

        # Remove the trace processor registered in _instrument so runs stop
        # being traced and a later instrument() does not double-register it
        otel_processor = self._otel_processor
        if otel_processor is not None:
            try:
                _remove_trace_processor(otel_processor, self._replaced_processors)
            except Exception:
                pass
            finally:
                self._otel_processor = None
                self._replaced_processors = None


def _current_trace_processors():
    """Return the processors registered with the Agents SDK default trace provider.

    The Agents SDK exposes no API to enumerate or remove trace processors, so
    this reads the default provider's internal multi-processor. Returns None
    when a custom provider is in use or the SDK internals differ.
    """
    try:
        from agents.tracing.setup import get_trace_provider
    except ImportError:
        return None
    provider = get_trace_provider()
    multi_processor = getattr(provider, "_multi_processor", None)
    if multi_processor is None:
        return None
    return list(getattr(multi_processor, "_processors", ()))


def _remove_trace_processor(processor, replaced_processors=None):
    """Remove a processor previously registered by _instrument from the Agents SDK.

    Processors the user added after instrument() are kept; processors saved
    when replace_existing_processors=True are restored.
    """
    current = _current_trace_processors()
    if current is None or processor not in current:
        return
    remaining = [p for p in current if p is not processor]
    if replaced_processors:
        remaining.extend(p for p in replaced_processors if p not in remaining)

    from agents import set_trace_processors

    set_trace_processors(remaining)


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
