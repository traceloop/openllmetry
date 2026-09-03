import pytest
from unittest.mock import MagicMock
from pydantic import BaseModel

from crewai import Agent, Crew, Task
from crewai.llms.base_llm import BaseLLM

from opentelemetry.instrumentation.crewai import CrewAIInstrumentor
from opentelemetry.trace.status import StatusCode


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
    instrumentor = CrewAIInstrumentor()
    instrumentor.instrument = MagicMock()
    instrumentor.uninstrument = MagicMock()
    return instrumentor


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




def test_native_provider_call_is_wrapped(instrument):
    """crewai's native provider classes (BaseLLM subclasses, not LLM) must have
    their call wrapped — previously only crewai.llm.LLM.call was patched, which
    the LLM.__new__ factory never returns on the native path."""
    from crewai.llms.providers.openai.completion import OpenAICompletion

    assert getattr(OpenAICompletion.call, "__wrapped__", None) is not None


def test_native_provider_uninstrument_restores_call(instrument):
    from crewai.llms.providers.openai.completion import OpenAICompletion

    instrument.uninstrument()
    try:
        assert getattr(OpenAICompletion.call, "__wrapped__", None) is None
    finally:
        # re-instrument so the autouse-style fixture teardown stays balanced
        instrument.instrument()


def test_litellm_fallback_still_infers_provider_from_model(mock_crew, mock_instrumentor):
    """The LLM.call wrap has no fixed provider and keeps model-name inference
    (used for the LiteLLM fallback path)."""
    from opentelemetry.instrumentation.crewai.instrumentation import _infer_llm_provider_from_model

    assert _infer_llm_provider_from_model("openai/gpt-4") == "openai"
    assert _infer_llm_provider_from_model("claude-3") == "anthropic"
    assert _infer_llm_provider_from_model("totally-unknown-xyz") is None
