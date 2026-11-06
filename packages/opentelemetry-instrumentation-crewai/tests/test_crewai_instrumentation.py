import pytest
from unittest.mock import MagicMock

import crewai.llms.base_llm as crewai_llms_base_llm
from crewai import Agent, Crew, Task
from crewai.llms.base_llm import BaseLLM
from opentelemetry.instrumentation.crewai import CrewAIInstrumentor
from opentelemetry.instrumentation.crewai import instrumentation as crewai_instrumentation
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace.status import StatusCode
from pydantic import BaseModel


class StubLLM(BaseLLM):
    """Minimal LLM for tests — CrewAI 1.x validates `llm` and rejects arbitrary mocks."""

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model: type[BaseModel] | None = None,
    ):
        return "Mocked response"


@pytest.fixture
def mock_instrumentor():
    # BaseInstrumentor is a singleton: the instance-level MagicMock below would
    # leak into later tests that call the real instrument(), so delete the
    # instance attributes on teardown to restore the class methods.
    instrumentor = CrewAIInstrumentor()
    instrumentor.instrument = MagicMock()
    instrumentor.uninstrument = MagicMock()
    yield instrumentor
    del instrumentor.instrument
    del instrumentor.uninstrument


@pytest.fixture
def mock_crew():
    llm = StubLLM(model="stub-model")
    agent = Agent(
        role="Data Collector",
        goal="Collect accurate and up-to-date financial data",
        backstory="You are an expert in gathering financial data from various sources.",
        llm=llm,
    )

    task = Task(
        description="Collect stock data for AAPL for the past month",
        expected_output=(
            "A comprehensive dataset containing daily stock prices, "
            "trading volumes, and any significant news or events "
            "affecting these stocks over the past month."
        ),
        agent=agent,
    )

    return Crew(agents=[agent], tasks=[task], tracing=False)


def test_crewai_instrumentation(mock_crew, mock_instrumentor):
    mock_instrumentor.instrument()
    mock_instrumentor.instrument.assert_called_once()

    assert len(mock_crew.agents) == 1
    assert mock_crew.agents[0].role == "Data Collector"
    assert len(mock_crew.tasks) == 1
    assert (
        mock_crew.tasks[0].description
        == "Collect stock data for AAPL for the past month"
    )


def test_trace_status(mock_crew, mock_instrumentor):
    mock_span = MagicMock()
    mock_span.set_status = MagicMock()

    mock_span.set_status(StatusCode.OK)
    mock_span.set_status.assert_called_with(StatusCode.OK)

    mock_span.set_status(StatusCode.ERROR)
    mock_span.set_status.assert_called_with(StatusCode.ERROR)

    memory_exporter = MagicMock()
    memory_exporter.get_finished_spans.return_value = [
        MagicMock(status=MagicMock(status_code=StatusCode.ERROR))
    ]

    spans = memory_exporter.get_finished_spans()
    assert spans[-1].status.status_code == StatusCode.ERROR

    mock_instrumentor.uninstrument()
    mock_instrumentor.uninstrument.assert_called_once()


# --- Real instrumentation tests for the native-provider LLM call path (#4453) ---
#
# On crewai >= 1.15, `crewai.llm.LLM.__new__` is a factory that returns native
# provider instances (e.g. OpenAICompletion) which are NOT subclasses of `LLM`,
# so the historical `LLM.call` wrap never fires on them. `_instrument` therefore
# also wraps each native provider class's `call`. These tests drive that wrap
# path with a BaseLLM subclass standing in for a provider class, since provider
# SDKs (openai/anthropic/...) are not test dependencies.


class NativeStyleStubLLM(BaseLLM):
    """BaseLLM subclass shaped like a crewai native provider (not an `LLM`)."""

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
    ):
        return "Mocked native response"


@pytest.fixture
def span_env():
    # crewai registers its own global TracerProvider on import, so never call
    # trace.set_tracer_provider here — pass the provider into instrument().
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


@pytest.fixture
def native_style_wrap(monkeypatch):
    """Route the native-provider wrap list at a BaseLLM subclass in this module."""
    monkeypatch.setattr(crewai_llms_base_llm, "NativeStyleStubLLM", NativeStyleStubLLM, raising=False)
    monkeypatch.setattr(
        crewai_instrumentation,
        "CREWAI_NATIVE_LLM_PROVIDERS",
        [("crewai.llms.base_llm", "NativeStyleStubLLM")],
    )


def test_native_style_llm_call_emits_span(native_style_wrap, span_env):
    provider, exporter = span_env
    instrumentor = CrewAIInstrumentor()
    instrumentor.instrument(tracer_provider=provider)
    try:
        llm = NativeStyleStubLLM(model="gpt-4o-mini")
        result = llm.call([{"role": "user", "content": "hello"}])
        assert result == "Mocked native response"

        llm_spans = [s for s in exporter.get_finished_spans() if s.name == "gpt-4o-mini.llm"]
        assert len(llm_spans) == 1, "native-provider-style LLM.call must produce a {model}.llm span"
        attrs = llm_spans[0].attributes
        assert attrs["gen_ai.request.model"] == "gpt-4o-mini"
        assert attrs["gen_ai.provider.name"] == "openai"
        assert llm_spans[0].status.status_code == StatusCode.OK
    finally:
        # BaseInstrumentor is a singleton — a failed assertion must not leave
        # the wraps in place for later tests.
        instrumentor.uninstrument()


def test_uninstrument_restores_native_style_call(native_style_wrap, span_env):
    provider, _ = span_env
    instrumentor = CrewAIInstrumentor()
    instrumentor.instrument(tracer_provider=provider)
    try:
        assert hasattr(NativeStyleStubLLM.call, "__wrapped__")
    finally:
        instrumentor.uninstrument()
    assert not hasattr(NativeStyleStubLLM.call, "__wrapped__")

    from crewai.llm import LLM

    assert not hasattr(LLM.call, "__wrapped__")


def test_real_native_providers_are_wrapped(span_env):
    try:
        from crewai.llms.providers.openai.completion import OpenAICompletion
    except ImportError:
        pytest.skip("crewai without native provider modules (< 1.15)")

    provider, _ = span_env
    instrumentor = CrewAIInstrumentor()
    instrumentor.instrument(tracer_provider=provider)
    try:
        assert hasattr(OpenAICompletion.call, "__wrapped__")
    finally:
        instrumentor.uninstrument()
    assert not hasattr(OpenAICompletion.call, "__wrapped__")
