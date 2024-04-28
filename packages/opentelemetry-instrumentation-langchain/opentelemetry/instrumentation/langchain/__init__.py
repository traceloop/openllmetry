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

from opentelemetry.instrumentation.langchain.version import __version__
from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY

from opentelemetry.semconv.ai import TraceloopSpanKindValues

from opentelemetry.instrumentation.langchain.init_wrapper import init_wrapper

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

logger = logging.getLogger(__name__)

_instruments = ("langchain >= 0.0.346", "langchain-core > 0.1.0")

TO_INSTRUMENT = [
    {
        "package": "langchain.chains.base",
        "class": "Chain",
        "callback_supported": True,
    },
    {
        "package": "langchain.agents",
        "class": "Agent",
        "callback_supported": True,
    },
    {
        "package": "langchain.tools",
        "class": "Tool",
        "span_name": "langchain.tool",
        "kind": TraceloopSpanKindValues.TOOL.value,
        "callback_supported": True,
    },
    {
        "package": "langchain.chains",
        "class": "RetrievalQA",
        "span_name": "retrieval_qa.workflow",
        "callback_supported": True,
    },
    {
        "package": "langchain.prompts.base",
        "object": "BasePromptTemplate",
        "method": "invoke",
        "wrapper": task_wrapper,
        "callback_supported": False,
    },
    {
        "package": "langchain.prompts.base",
        "object": "BasePromptTemplate",
        "method": "ainvoke",
        "wrapper": atask_wrapper,
        "callback_supported": False,
    },
    {
        "package": "langchain.chat_models.base",
        "object": "BaseChatModel",
        "method": "generate",
        "wrapper": chat_wrapper,
        "callback_supported": False,
    },
    {
        "package": "langchain.chat_models.base",
        "object": "BaseChatModel",
        "method": "agenerate",
        "wrapper": achat_wrapper,
        "callback_supported": False,
    },
    {
        "package": "langchain.schema",
        "object": "BaseOutputParser",
        "method": "invoke",
        "wrapper": task_wrapper,
        "callback_supported": False,
    },
    {
        "package": "langchain.schema",
        "object": "BaseOutputParser",
        "method": "ainvoke",
        "wrapper": atask_wrapper,
        "callback_supported": False,
    },
    {
        "package": "langchain.schema.runnable",
        "object": "RunnableSequence",
        "method": "invoke",
        "span_name": "langchain.workflow",
        "wrapper": workflow_wrapper,
        "callback_supported": False,
    },
    {
        "package": "langchain.schema.runnable",
        "object": "RunnableSequence",
        "method": "ainvoke",
        "span_name": "langchain.workflow",
        "wrapper": aworkflow_wrapper,
        "callback_supported": False,
    },
    {
        "package": "langchain_core.language_models.llms",
        "object": "LLM",
        "method": "_generate",
        "span_name": "llm.generate",
        "wrapper": llm_wrapper,
        "callback_supported": False,
    },
    {
        "package": "langchain_core.language_models.llms",
        "object": "LLM",
        "method": "_agenerate",
        "span_name": "llm.generate",
        "wrapper": allm_wrapper,
        "callback_supported": False,
    },
]


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
                if module.get("callback_supported"):
                    wrap_class = module.get("class")
                    wrap_function_wrapper(
                        wrap_package, f"{wrap_class}.__init__", init_wrapper(tracer, module)
                    )
                else:
                    wrap_object = module.get("object")
                    wrap_method = module.get("method")
                    wrapper = module.get("wrapper")
                    wrap_function_wrapper(
                        wrap_package,
                        f"{wrap_object}.{wrap_method}" if wrap_object else wrap_method,
                        wrapper(tracer, module),
                    )

            except PackageNotFoundError:
                pass

    def _uninstrument(self, **kwargs):
        for module in TO_INSTRUMENT:
            if not module.get("callback_supported"):
                wrap_class = module.get("class")
                unwrap(
                    wrap_package, 
                    f"{wrap_class}.__init__"
                )
            else:
                wrap_package = module.get("package")
                wrap_object = module.get("object")
                wrap_method = module.get("method")
                unwrap(
                    f"{wrap_package}.{wrap_object}" if wrap_object else wrap_package,
                    wrap_method,
                )
