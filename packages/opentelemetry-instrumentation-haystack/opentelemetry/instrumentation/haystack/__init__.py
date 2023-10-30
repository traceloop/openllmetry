import logging
from typing import Collection
from wrapt import wrap_function_wrapper

from opentelemetry.trace import get_tracer
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (
    unwrap,
)
from opentelemetry.instrumentation.haystack.wrap_openai import wrap as openai_wrapper
from opentelemetry.instrumentation.haystack.wrap_pipeline import (
    wrap as pipeline_wrapper,
)
from opentelemetry.instrumentation.haystack.wrap_node import (
    wrap as node_wrapper,
)
from opentelemetry.instrumentation.haystack.version import __version__

logger = logging.getLogger(__name__)

_instruments = ("farm-haystack >= 1.20.1",)

WRAPPED_METHODS = [
    {
        "package": "haystack.nodes.prompt.invocation_layer.chatgpt",
        "object": "ChatGPTInvocationLayer",
        "method": "_execute_openai_request",
        "wrapper": openai_wrapper,
    },
    {
        "package": "haystack.nodes.prompt.invocation_layer.open_ai",
        "object": "OpenAIInvocationLayer",
        "method": "_execute_openai_request",
        "wrapper": openai_wrapper,
    },
    {
        "package": "haystack.pipelines.base",
        "object": "Pipeline",
        "method": "run",
        "wrapper": pipeline_wrapper,
    },
    {
        "package": "haystack.nodes.prompt.prompt_node",
        "object": "PromptNode",
        "method": "run",
        "wrapper": node_wrapper,
    },
    {
        "package": "haystack.nodes.retriever.dense",
        "object": "EmbeddingRetriever",
        "method": "retrieve",
        "wrapper": node_wrapper,
    },
]


class HaystackInstrumentor(BaseInstrumentor):
    """An instrumentor for the Haystack framework."""

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
