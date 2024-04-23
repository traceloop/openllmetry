"""OpenTelemetry Langchain instrumentation"""

import logging
from typing import Collection
from opentelemetry.instrumentation.langchain.config import Config
from wrapt import wrap_function_wrapper
from importlib.metadata import version as package_version, PackageNotFoundError

from opentelemetry.trace import get_tracer

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.instrumentation.langchain.utils import _with_tracer_wrapper

from opentelemetry.instrumentation.langchain.task_wrapper import (
    task_wrapper,
    atask_wrapper,
)
from opentelemetry.instrumentation.langchain.workflow_wrapper import (
    workflow_wrapper,
    aworkflow_wrapper,
)
from opentelemetry.instrumentation.langchain.custom_llm_wrapper import (
    llm_wrapper,
    allm_wrapper,
)
from opentelemetry.instrumentation.langchain.custom_chat_wrapper import (
    chat_wrapper,
    achat_wrapper,
)
from opentelemetry.instrumentation.langchain.version import __version__
from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY

from opentelemetry.semconv.ai import TraceloopSpanKindValues

logger = logging.getLogger(__name__)

_instruments = ("langchain >= 0.0.346", "langchain-core > 0.1.0")

TO_INSTRUMENT = [
    {
        "package": "langchain.chains.base",
        "class": "Chain",
    },
    # {
    #     "package": "langchain.chains",
    #     "class": "LLMChain",
    # },
    # {
    #     "package": "langchain.chains",
    #     "class": "TransformChain",
    # },
    # {
    #     "package": "langchain.chains",
    #     "class": "SequentialChain",
    #     "span_name": "langchain.workflow",
    # },
    {
        "package": "langchain.agents",
        "class": "Agent",
    },
    # {
    #     "package": "langchain.agents",
    #     "class": "AgentExecutor",
    # #     "span_name": "langchain.agent",
    # #     "kind": TraceloopSpanKindValues.AGENT.value,
    # # },
    {
        "package": "langchain.tools",
        "class": "Tool",
        "span_name": "langchain.tool",
        "kind": TraceloopSpanKindValues.TOOL.value,
    },
    {
        "package": "langchain.chains",
        "class": "RetrievalQA",
        "span_name": "retrieval_qa.workflow",
    },
    {
        "package": "langchain.chat_models.base",
        "object": "BaseChatModel",
        "method": "generate",
        "wrapper": chat_wrapper,
    },
    {
        "package": "langchain.chat_models.base",
        "object": "BaseChatModel",
        "method": "agenerate",
        "wrapper": achat_wrapper,
    },
    {
        "package": "langchain.schema",
        "object": "BaseOutputParser",
        "method": "invoke",
        "wrapper": task_wrapper,
    },
    {
        "package": "langchain.schema",
        "object": "BaseOutputParser",
        "method": "ainvoke",
        "wrapper": atask_wrapper,
    },
    {
        "package": "langchain.schema.runnable",
        "object": "RunnableSequence",
        "method": "invoke",
        "span_name": "langchain.workflow",
    },
    {
        "package": "langchain_core.language_models.llms",
        "object": "LLM",
        "method": "_generate",
        "span_name": "llm.generate",
        "wrapper": llm_wrapper,
    },
    {
        "package": "langchain_core.language_models.llms",
        "object": "LLM",
        "method": "_agenerate",
        "span_name": "llm.generate",
        "wrapper": allm_wrapper,
    },
]

@_with_tracer_wrapper
def _init_wrapper(tracer, span_name, kind_name, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        print("SUPPRESS_INSTRUMENTATION_KEY", _SUPPRESS_INSTRUMENTATION_KEY)
        return wrapped(*args, **kwargs)
    kwargs["callbacks"] = [SpanCallbackHandler(tracer, span_name, kind_name)]
    return wrapped(*args, **kwargs)

class LangchainInstrumentor(BaseInstrumentor):
    """An instrumentor for Langchain SDK."""

    def __init__(self, exception_logger=None):
        super().__init__()
        Config.exception_logger = exception_logger

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)
        for module in TO_INSTRUMENT:
            try:
                wrap_package = module.get("package")
                wrap_class = module.get("class")
                wrap_span_name = module.get("span_name", None)
                wrap_kind_name = module.get("kind", None)
                wrap_function_wrapper(
                    wrap_package, f"{wrap_class}.__init__", _init_wrapper(tracer, wrap_span_name, wrap_kind_name)
                )
            except PackageNotFoundError:
                pass

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            unwrap(
                f"{wrap_package}.{wrap_object}" if wrap_object else wrap_package,
                wrap_method,
            )
