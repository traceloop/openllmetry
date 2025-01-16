import pytest
from unittest.mock import MagicMock
from opentelemetry.instrumentation.crewai import CrewAIInstrumentor
from crewai import Agent, Task, Crew
from opentelemetry.trace.status import StatusCode


@pytest.fixture
def mock_instrumentor():
    instrumentor = CrewAIInstrumentor()
    instrumentor.instrument = MagicMock()
    instrumentor.uninstrument = MagicMock()
    return instrumentor


@pytest.fixture
def mock_crew():
    mock_llm = MagicMock()
    mock_llm.predict.return_value = "Mocked response"
    agent = Agent(
        role="Data Collector",
        goal="Collect accurate and up-to-date financial data",
        backstory="You are an expert in gathering financial data from various sources.",
        llm=mock_llm,
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

    crew = Crew(agents=[agent], tasks=[task])

    with pytest.raises(Exception):
        crew.kickoff()
    return crew


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
