"""OpenTelemetry OpenAI Agents instrumentation"""

from opentelemetry.instrumentation.openai_agents.version import __version__
from opentelemetry.instrumentation.openai_agents.instrumentation import (
    OpenAIAgentsInstrumentor,
)

__all__ = ["OpenAIAgentsInstrumentor", "__version__"]
