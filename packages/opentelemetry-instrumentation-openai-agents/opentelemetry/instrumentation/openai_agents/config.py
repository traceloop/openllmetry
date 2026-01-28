from typing import Any, Optional

# Note: opentelemetry._events is the official incubating Events API. The underscore
# prefix indicates the API is experimental/not yet stable, not that it's private.
# This is the recommended way to use OpenTelemetry events until the API stabilizes.
# See: https://opentelemetry.io/docs/specs/otel/logs/event-api/
from opentelemetry._events import EventLogger


class Config:
    """
    Global configuration for OpenAI Agents instrumentation.

    This class uses class-level attributes intentionally to provide a singleton
    configuration pattern. OpenTelemetry instrumentors are designed to be used
    as singletons - only one instrumentor instance should exist per instrumented
    library. The BaseInstrumentor class enforces this by preventing double
    instrumentation.

    Warning:
        Creating multiple OpenAIAgentsInstrumentor instances with different
        configurations is not supported. The last instance's configuration
        will override all previous ones.
    """

    exception_logger: Optional[Any] = None
    use_legacy_attributes: bool = True
    event_logger: Optional[EventLogger] = None
