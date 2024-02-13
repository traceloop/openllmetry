"""OpenTelemetry LlamaIndex instrumentation"""
import logging
from importlib.metadata import version as package_version, PackageNotFoundError
from typing import Collection

from opentelemetry.trace import get_tracer

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor

from opentelemetry.instrumentation.llamaindex.retriever_query_engine_instrumentor import (
    RetrieverQueryEngineInstrumentor)
from opentelemetry.instrumentation.llamaindex.base_retriever_instrumentor import BaseRetrieverInstrumentor
from opentelemetry.instrumentation.llamaindex.base_synthesizer_instrumentor import BaseSynthesizerInstrumentor
from opentelemetry.instrumentation.llamaindex.base_embedding_instrumentor import BaseEmbeddingInstrumentor
from opentelemetry.instrumentation.llamaindex.custom_llm_instrumentor import CustomLLMInstrumentor
from opentelemetry.instrumentation.llamaindex.version import __version__

logger = logging.getLogger(__name__)


class LlamaIndexInstrumentor(BaseInstrumentor):
    """An instrumentor for LlamaIndex SDK."""

    def _set_instruments(self) -> Collection[str]:
        try:
            package_version("llama-index-core")
            return ("llama-index-core >= 0.10.0",)
        except PackageNotFoundError:
            try:
                package_version("llama-index-legacy")
                return ("llama-index-legacy >= 0.7.0, <= 0.9.48",)
            except PackageNotFoundError:
                logger.error("Neither llama-index-core nor llama-index-legacy package is found. "
                             "Ensure one is installed.")

    def instrumentation_dependencies(self) -> Collection[str]:
        return self._set_instruments()

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        RetrieverQueryEngineInstrumentor(tracer).instrument()
        BaseRetrieverInstrumentor(tracer).instrument()
        BaseSynthesizerInstrumentor(tracer).instrument()
        BaseEmbeddingInstrumentor(tracer).instrument()
        CustomLLMInstrumentor(tracer).instrument()

    def _uninstrument(self, **kwargs):
        pass
