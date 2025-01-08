"""Unit tests configuration module."""

import os
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter
from typing import Sequence
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.instrumentation.langchain import LangchainInstrumentor, Config
from opentelemetry import metrics
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.semconv.ai import SpanAttributes # Correct import
from langchain_cohere.chat_models import ChatCohere
from langchain_cohere.llms import Cohere, completion_with_retry
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import HumanMessage
from typing import Any, List, Optional
import functools

pytest_plugins = []

# Custom Span Exporter that stores spans in a list
class ListSpanExporter(SpanExporter):
    def __init__(self):
        self.spans = []

    def export(self, spans: Sequence[ReadableSpan]) -> None:
        self.spans.extend(spans)

    def shutdown(self) -> None:
        pass  # No cleanup needed for in-memory storage

    def clear(self):
        self.spans = []

@pytest.fixture(scope="session")
def exporter():
    exporter = ListSpanExporter()  # Use the custom exporter
    processor = SimpleSpanProcessor(exporter)

    provider = TracerProvider()
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    OpenAIInstrumentor().instrument()
    LangchainInstrumentor().instrument()

    return exporter

@pytest.fixture(autouse=True)
def clear_exporter(exporter):
    exporter.clear()

@pytest.fixture(autouse=True)
def environment():
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "test"  # Placeholder, not used in Cohere tests
    if not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = "test"  # Placeholder, not used
    if not os.environ.get("COHERE_API_KEY"):
        os.environ["COHERE_API_KEY"] = "YOUR_ACTUAL_COHERE_API_KEY"  # Use your real key here
    if not os.environ.get("TAVILY_API_KEY"):
        os.environ["TAVILY_API_KEY"] = "test"  # Placeholder, not used
    if not os.environ.get("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_API_KEY"] = "test"  # Placeholder, not used

@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["authorization", "x-api-key"],
        "filter_body": ["api_key"],
        "ignore_hosts": ["api.hub.langchain.com", "api.smith.langchain.com"],
    }

@pytest.fixture(scope="session")
def metrics_test_context():
    resource = Resource.create()
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader], resource=resource)

    metrics.set_meter_provider(provider)

    LangchainInstrumentor().instrument()

    return provider, reader

@pytest.fixture(scope="session", autouse=True)
def clear_metrics_test_context(metrics_test_context):
    provider, reader = metrics_test_context

    yield

    reader.shutdown()
    provider.shutdown()

@pytest.fixture(params=[True, False])
def langchain_use_legacy_attributes_fixture(request):
    Config.use_legacy_attributes = request.param
    yield request.param

@pytest.fixture(scope="session")
def test_context():
    exporter = ListSpanExporter()  # Use the custom exporter
    span_processor = SimpleSpanProcessor(exporter)
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(span_processor)
    trace.set_tracer_provider(tracer_provider)

    tracer_provider = trace.get_tracer_provider()
    tracer = tracer_provider.get_tracer("langchain.test_tracer")

    OpenAIInstrumentor().instrument()
    LangchainInstrumentor().instrument()

    context = (exporter, tracer_provider, tracer)

    return context

class MyCohere(Cohere):
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # Remove the model parameter from kwargs before calling the parent class's _call method
        kwargs.pop("model", None)
        return super()._call(prompt, stop, run_manager, **kwargs)

@pytest.fixture(scope="module")
def cohere_llm(request, test_context):
    """Fixture for Cohere LLM"""
    # Apply instrumentation only when legacy attributes are to be used
    # This suppresses the OpenAI instrumentation
    request.applymarker(
        pytest.mark.parametrize(
            "langchain_use_legacy_attributes_fixture",
            [True],
            indirect=True,
        )
    )
    Config.suppress_openai_instrumentation = True
    # Create an instance of MyCohere and return it
    llm = MyCohere(temperature=0)  # Remove model="command" here
    yield llm
    # Reset the flag after the test is done
    Config.suppress_openai_instrumentation = False

@pytest.fixture(scope="module")
def cohere_chat_llm(request, test_context):
    """Fixture for Cohere Chat LLM"""
    # Apply instrumentation only when legacy attributes are to be used
    # This suppresses the OpenAI instrumentation
    request.applymarker(
        pytest.mark.parametrize(
            "langchain_use_legacy_attributes_fixture",
            [True],
            indirect=True,
        )
    )
    Config.suppress_openai_instrumentation = True
    # Create an instance of ChatCohere and return it
    chat_llm = ChatCohere(model="command", temperature=0)
    yield chat_llm
    # Reset the flag after the test is done
    Config.suppress_openai_instrumentation = False