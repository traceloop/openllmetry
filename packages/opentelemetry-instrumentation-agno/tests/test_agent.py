import pytest
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


@pytest.mark.vcr
def test_agent_run_basic(instrument, span_exporter, reader):
    """Test basic agent.run() instrumentation."""
    agent = Agent(
        name="TestAgent",
        model=OpenAIChat(id="gpt-4o-mini"),
        description="A simple test agent",
    )

    agent.run("What is 2 + 2?")

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1

    agent_span = spans[-1]
    assert agent_span.name == "agno.agent.TestAgent"
    assert agent_span.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "agno"
    assert agent_span.attributes.get(GenAIAttributes.GEN_AI_AGENT_NAME) == "TestAgent"
    assert agent_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL) == "gpt-4o-mini"

    prompt_content = agent_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.0.content")
    assert prompt_content == "What is 2 + 2?"

    completion_content = agent_span.attributes.get(
        f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"
    )
    assert completion_content is not None


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_agent_arun_basic(instrument, span_exporter, reader):
    """Test basic agent.arun() instrumentation."""
    agent = Agent(
        name="AsyncTestAgent",
        model=OpenAIChat(id="gpt-4o-mini"),
        description="A simple async test agent",
    )

    await agent.arun("What is the capital of France?")

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1

    agent_span = spans[-1]
    assert agent_span.name == "agno.agent.AsyncTestAgent"
    assert agent_span.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "agno"
    assert agent_span.attributes.get(GenAIAttributes.GEN_AI_AGENT_NAME) == "AsyncTestAgent"

    prompt_content = agent_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.0.content")
    assert "capital of France" in prompt_content


@pytest.mark.vcr
def test_agent_with_tools(instrument, span_exporter, reader):
    """Test agent with tools instrumentation."""

    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    agent = Agent(
        name="ToolAgent",
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[add_numbers],
    )

    agent.run("Add 5 and 7")

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1

    agent_span = spans[-1]
    assert agent_span.name == "agno.agent.ToolAgent"
    assert agent_span.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "agno"


@pytest.mark.vcr
def test_agent_metrics(instrument, span_exporter, reader):
    """Test that metrics are recorded for agent runs."""
    agent = Agent(
        name="MetricsAgent",
        model=OpenAIChat(id="gpt-4o-mini"),
    )

    agent.run("Tell me a short joke")

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1

    metrics_data = reader.get_metrics_data()
    if metrics_data is not None:
        resource_metrics = metrics_data.resource_metrics

        assert len(resource_metrics) > 0

        for rm in resource_metrics:
            for sm in rm.scope_metrics:
                for metric in sm.metrics:
                    if metric.name == "gen_ai.client.operation.duration":
                        assert len(metric.data.data_points) > 0
                        for dp in metric.data.data_points:
                            assert dp.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "agno"
