"""OpenTelemetry LlamaIndex instrumentation"""
import logging
from typing import Collection
from wrapt import wrap_function_wrapper

from opentelemetry.trace import get_tracer

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap

from opentelemetry.instrumentation.llamaindex.task_wrapper import task_wrapper
from opentelemetry.instrumentation.llamaindex.workflow_wrapper import workflow_wrapper
from opentelemetry.instrumentation.llamaindex.version import __version__

logger = logging.getLogger(__name__)

_instruments = ("llama-index >= 0.7.0",)

WRAPPED_METHODS = [
    {
        "package": "llama_index.indices.query.base",
        "object": "BaseQueryEngine",
        "method": "query",
        "wrapper": workflow_wrapper,
    },
    {
        "package": "llama_index.indices.query.base",
        "object": "BaseQueryEngine",
        "method": "aquery",
        "wrapper": workflow_wrapper,
    },
    {
        "package": "llama_index.indices.base_retriever",
        "object": "BaseRetriever",
        "method": "retrieve",
        "span_name": "retrieve",
        "wrapper": task_wrapper
    },
    {
        "package": "llama_index.indices.base_retriever",
        "object": "BaseRetriever",
        "method": "aretrieve",
        "span_name": "retrieve",
        "wrapper": task_wrapper
    },
    {
        "package": "llama_index.embeddings.base",
        "object": "BaseEmbedding",
        "method": "get_query_embedding",
        "span_name": "get_query_embedding",
        "wrapper": task_wrapper
    },
    {
        "package": "llama_index.embeddings.base",
        "object": "BaseEmbedding",
        "method": "aget_query_embedding",
        "span_name": "get_query_embedding",
        "wrapper": task_wrapper
    },
    {
        "package": "llama_index.response_synthesizers",
        "object": "BaseSynthesizer",
        "method": "synthesize",
        "span_name": "synthesize",
        "wrapper": task_wrapper
    },
    {
        "package": "llama_index.response_synthesizers",
        "object": "BaseSynthesizer",
        "method": "asynthesize",
        "span_name": "synthesize",
        "wrapper": task_wrapper
    }
]


class LlamaIndexInstrumentor(BaseInstrumentor):
    """An instrumentor for LlamaIndex SDK."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            wrapper = wrapped_method.get("wrapper")
            wrap_function_wrapper(
                wrap_package,
                f"{wrap_object}.{wrap_method}" if wrap_object else wrap_method,
                wrapper(tracer, wrapped_method),
            )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            unwrap(
                f"{wrap_package}.{wrap_object}" if wrap_object else wrap_package,
                wrap_method,
            )
