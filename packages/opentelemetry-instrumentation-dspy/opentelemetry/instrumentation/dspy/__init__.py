"""OpenTelemetry DSPy instrumentation"""
from opentelemetry.instrumentation.dspy.version import __version__
from opentelemetry.instrumentation.dspy.instrumentation import DSPyInstrumentor

__all__ = ["DSPyInstrumentor", "__version__"]
