import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock
from pydantic import BaseModel

from crewai import Agent, Crew, Task
from crewai.llms.base_llm import BaseLLM
from crewai.utilities.planning_handler import CrewPlanner, PlanPerTask, PlannerTaskPydanticOutput

from opentelemetry.instrumentation.crewai import CrewAIInstrumentor
from opentelemetry.instrumentation.crewai.instrumentation import (
    _crew_planning_span,
    wrap_create_planning_agent,
    wrap_crew_planning,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
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


def _plan_per_task(task, plan):
    values = {"task": task, "plan": plan}
    if "task_number" in getattr(PlanPerTask, "model_fields", {}):
        values["task_number"] = 1
    return PlanPerTask(**values)


@pytest.fixture
def disable_crewai_telemetry(monkeypatch):
    from crewai.telemetry.telemetry import Telemetry

    def telemetry_init(self):
        self.ready = False
        self.trace_set = False

    monkeypatch.setattr(Telemetry, "__init__", telemetry_init)
    monkeypatch.setattr(Telemetry, "set_tracer", lambda self: None)


@pytest.fixture
def mock_instrumentor():
    return MagicMock(spec=CrewAIInstrumentor)


@pytest.fixture
def mock_crew(disable_crewai_telemetry):
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


def test_wrap_crew_planning_creates_plan_span():
    span = MagicMock()
    tracer = MagicMock()
    tracer.start_as_current_span.return_value.__enter__.return_value = span
    planner = MagicMock()
    wrapped = MagicMock(return_value="planning-result")

    result = wrap_crew_planning(tracer, None, None)(wrapped, planner, (), {})

    assert result == "planning-result"
    tracer.start_as_current_span.assert_called_once()
    args, kwargs = tracer.start_as_current_span.call_args
    assert args[0] == "plan"
    assert kwargs["attributes"]["gen_ai.provider.name"] == "crewai"
    assert kwargs["attributes"]["gen_ai.operation.name"] == "plan"
    assert "gen_ai.agent.id" not in kwargs["attributes"]
    assert "gen_ai.agent.name" not in kwargs["attributes"]
    planner._create_planning_agent.assert_not_called()
    span.set_status.assert_called_once()
    assert span.set_status.call_args.args[0].status_code == StatusCode.OK
    assert _crew_planning_span.get() is None


def test_wrap_crew_planning_records_error_status():
    span = MagicMock()
    tracer = MagicMock()
    tracer.start_as_current_span.return_value.__enter__.return_value = span
    planner = MagicMock()
    planner._create_planning_agent.return_value = MagicMock(
        role="Task Execution Planner",
        id="agent-id",
    )
    wrapped = MagicMock(side_effect=RuntimeError("planning failed"))

    with pytest.raises(RuntimeError, match="planning failed"):
        wrap_crew_planning(tracer, None, None)(wrapped, planner, (), {})

    assert span.set_status.call_args.args[0].status_code == StatusCode.ERROR
    assert _crew_planning_span.get() is None


def test_wrap_create_planning_agent_ignores_agent_outside_plan_span(monkeypatch):
    set_span_attribute = MagicMock()
    monkeypatch.setattr(
        "opentelemetry.instrumentation.crewai.instrumentation.set_span_attribute",
        set_span_attribute,
    )
    planner_agent = MagicMock(role="Task Execution Planner", id="agent-id")
    wrapped = MagicMock(return_value=planner_agent)

    result = wrap_create_planning_agent(wrapped, MagicMock(), (), {})

    assert result is planner_agent
    assert _crew_planning_span.get() is None
    set_span_attribute.assert_not_called()


def test_wrap_create_planning_agent_enriches_stored_plan_span():
    plan_span = MagicMock()
    planner_agent = MagicMock(role="Task Execution Planner", id="agent-id")
    wrapped = MagicMock(return_value=planner_agent)
    token = _crew_planning_span.set(plan_span)
    try:
        result = wrap_create_planning_agent(wrapped, MagicMock(), (), {})
    finally:
        _crew_planning_span.reset(token)

    assert result is planner_agent
    plan_span.update_name.assert_called_once_with("plan Task Execution Planner")
    plan_span.set_attribute.assert_any_call("gen_ai.agent.name", "Task Execution Planner")
    plan_span.set_attribute.assert_any_call("gen_ai.agent.id", "agent-id")


def test_crewai_planner_instrumentation_creates_plan_span_without_state_mutation():
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    task = SimpleNamespace(
        description="Collect stock data",
        expected_output="A financial dataset",
        tools=[],
        agent=SimpleNamespace(role="Researcher", goal="Find market data", tools=[]),
    )
    planner = CrewPlanner(tasks=[task], planning_agent_llm=StubLLM(model="stub-planner-model"))
    assert "_create_planning_agent" not in planner.__dict__

    def create_planner_task(planning_agent, tasks_summary):
        fake_task = MagicMock()
        fake_task.execute_sync.return_value = SimpleNamespace(
            pydantic=PlannerTaskPydanticOutput(
                list_of_plans_per_task=[
                    _plan_per_task(task.description, "Use the researcher to collect the data")
                ]
            )
        )
        return fake_task

    planner._create_planner_task = create_planner_task
    instrumentor = CrewAIInstrumentor()
    try:
        instrumentor.instrument(tracer_provider=tracer_provider)
        result = planner._handle_crew_planning()
    finally:
        instrumentor.uninstrument()

    assert isinstance(result, PlannerTaskPydanticOutput)
    assert "_create_planning_agent" not in planner.__dict__
    plan_spans = [
        span for span in exporter.get_finished_spans() if span.attributes.get("gen_ai.operation.name") == "plan"
    ]
    assert len(plan_spans) == 1
    assert plan_spans[0].name == "plan Task Execution Planner"
    assert plan_spans[0].attributes["gen_ai.provider.name"] == "crewai"
    assert plan_spans[0].attributes["gen_ai.agent.name"] == "Task Execution Planner"
    assert "gen_ai.agent.id" in plan_spans[0].attributes
