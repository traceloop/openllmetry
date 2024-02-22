"""OpenTelemetry Langchain instrumentation"""

import logging
from typing import Collection
from wrapt import wrap_function_wrapper

from opentelemetry.trace import get_tracer

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap

from opentelemetry.instrumentation.langchain.task_wrapper import (
    task_wrapper,
    atask_wrapper,
)
from opentelemetry.instrumentation.langchain.workflow_wrapper import (
    workflow_wrapper,
    aworkflow_wrapper,
)
from opentelemetry.instrumentation.langchain.version import __version__

from opentelemetry.semconv.ai import TraceloopSpanKindValues

logger = logging.getLogger(__name__)

_instruments = ("langchain >= 0.0.346",)

WRAPPED_METHODS = [
    {
        "package": "langchain.chains.base",
        "object": "Chain",
        "method": "__call__",
        "wrapper": task_wrapper,
    },
    {
        "package": "langchain.chains.base",
        "object": "Chain",
        "method": "acall",
        "wrapper": atask_wrapper,
    },
    {
        "package": "langchain.chains",
        "object": "SequentialChain",
        "method": "__call__",
        "span_name": "langchain.workflow",
        "wrapper": workflow_wrapper,
    },
    {
        "package": "langchain.chains",
        "object": "SequentialChain",
        "method": "acall",
        "span_name": "langchain.workflow",
        "wrapper": aworkflow_wrapper,
    },
    {
        "package": "langchain.agents",
        "object": "AgentExecutor",
        "method": "_call",
        "span_name": "langchain.agent",
        "kind": TraceloopSpanKindValues.AGENT.value,
        "wrapper": workflow_wrapper,
    },
    {
        "package": "langchain.tools",
        "object": "Tool",
        "method": "_run",
        "span_name": "langchain.tool",
        "kind": TraceloopSpanKindValues.TOOL.value,
        "wrapper": task_wrapper,
    },
    {
        "package": "langchain.chains",
        "object": "RetrievalQA",
        "method": "__call__",
        "span_name": "retrieval_qa.workflow",
        "wrapper": workflow_wrapper,
    },
    {
        "package": "langchain.chains",
        "object": "RetrievalQA",
        "method": "acall",
        "span_name": "retrieval_qa.workflow",
        "wrapper": aworkflow_wrapper,
    },
    {
        "package": "langchain.prompts.base",
        "object": "BasePromptTemplate",
        "method": "invoke",
        "wrapper": task_wrapper,
    },
    {
        "package": "langchain.prompts.base",
        "object": "BasePromptTemplate",
        "method": "ainvoke",
        "wrapper": atask_wrapper,
    },
    {
        "package": "langchain.chat_models.base",
        "object": "BaseChatModel",
        "method": "invoke",
        "wrapper": task_wrapper,
    },
    {
        "package": "langchain.chat_models.base",
        "object": "BaseChatModel",
        "method": "ainvoke",
        "wrapper": atask_wrapper,
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
        "wrapper": workflow_wrapper,
    },
    {
        "package": "langchain.schema.runnable",
        "object": "RunnableSequence",
        "method": "ainvoke",
        "span_name": "langchain.workflow",
        "wrapper": aworkflow_wrapper,
    },
]


class LangchainInstrumentor(BaseInstrumentor):
    """An instrumentor for Langchain SDK."""

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
