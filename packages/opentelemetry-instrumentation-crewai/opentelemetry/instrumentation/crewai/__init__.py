"""OpenTelemetry CrewAI instrumentation"""
from opentelemetry.instrumentation.crewai.instrumentation import CrewAIInstrumentor
from opentelemetry.instrumentation.crewai.version import __version__

__all__ = ["CrewAIInstrumentor", "__version__"]
