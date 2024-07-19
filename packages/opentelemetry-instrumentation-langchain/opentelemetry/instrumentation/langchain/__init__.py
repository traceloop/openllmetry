"""OpenTelemetry Langchain instrumentation"""

import logging
from typing import Collection
from opentelemetry.instrumentation.langchain.config import Config
from wrapt import wrap_function_wrapper

from opentelemetry.trace import get_tracer

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap

from opentelemetry.instrumentation.langchain.version import __version__


from opentelemetry.instrumentation.langchain.callback_wrapper import callback_wrapper

logger = logging.getLogger(__name__)

_instruments = ("langchain >= 0.0.346", "langchain-core > 0.1.0")

ASYNC_CALLBACK_FUNCTIONS = ("ainvoke", "astream", "atransform")
SYNC_CALLBACK_FUNCTIONS = ("invoke", "stream", "transform")
WRAPPED_METHODS = [
    {"package": "langchain.agents", "class": "AgentExecutor"},
    {
        "package": "langchain.chains.base",
        "class": "Chain",
    },
    {
        "package": "langchain_core.runnables.base",
        "class": "RunnableSequence",
    },
    {
        "package": "langchain.prompts.base",
        "class": "BasePromptTemplate",
    },
    {
        "package": "langchain.chat_models.base",
        "class": "BaseChatModel",
    },
    {
        "package": "langchain.schema",
        "class": "BaseOutputParser",
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
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_class = wrapped_method.get("class")
            for func_name in SYNC_CALLBACK_FUNCTIONS + ASYNC_CALLBACK_FUNCTIONS:
                wrap_function_wrapper(
                    wrap_package,
                    f"{wrap_class}.{func_name}",
                    callback_wrapper(tracer, wrapped_method),
                )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_class = wrapped_method.get("class")
            for func_name in SYNC_CALLBACK_FUNCTIONS + ASYNC_CALLBACK_FUNCTIONS:
                unwrap(wrap_package, f"{wrap_class}.{func_name}")
