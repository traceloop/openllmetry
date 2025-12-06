import pytest
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes


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
    assert agent_span.name == "TestAgent.agent"
    assert agent_span.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "agno"
    assert agent_span.attributes.get(GenAIAttributes.GEN_AI_AGENT_NAME) == "TestAgent"
    assert agent_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL) == "gpt-4o-mini"

    prompt_content = agent_span.attributes.get(SpanAttributes.TRACELOOP_ENTITY_INPUT)
    assert prompt_content == "What is 2 + 2?"

    completion_content = agent_span.attributes.get(SpanAttributes.TRACELOOP_ENTITY_OUTPUT)
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
    assert agent_span.name == "AsyncTestAgent.agent"
    assert agent_span.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "agno"
    assert agent_span.attributes.get(GenAIAttributes.GEN_AI_AGENT_NAME) == "AsyncTestAgent"

    prompt_content = agent_span.attributes.get(SpanAttributes.TRACELOOP_ENTITY_INPUT)
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

    # Check for agent span
    agent_span = spans[-1]
    assert agent_span.name == "ToolAgent.agent"
    assert agent_span.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "agno"

    # Check for tool spans (if tools were executed)
    tool_spans = [s for s in spans if s.name == "add_numbers.tool"]
    # Tool spans may or may not be present depending on whether the agent actually calls the tool
    # so we just check that if they exist, they have the right attributes
    for tool_span in tool_spans:
        assert tool_span.attributes.get(SpanAttributes.TRACELOOP_ENTITY_NAME) == "add_numbers"
        assert tool_span.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "agno"


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


@pytest.mark.vcr
def test_agent_run_streaming(instrument, span_exporter, reader):
    """Test agent.run() with streaming enabled."""
    agent = Agent(
        name="StreamAgent",
        model=OpenAIChat(id="gpt-4o-mini"),
        description="A streaming test agent",
    )

    events = []
    for event in agent.run("What is 10 + 5?", stream=True):
        events.append(event)

    assert len(events) > 0

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1

    agent_span = spans[-1]
    assert agent_span.name == "StreamAgent.agent"
    assert agent_span.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "agno"
    assert agent_span.attributes.get(GenAIAttributes.GEN_AI_AGENT_NAME) == "StreamAgent"
    assert agent_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL) == "gpt-4o-mini"

    prompt_content = agent_span.attributes.get(SpanAttributes.TRACELOOP_ENTITY_INPUT)
    assert "10 + 5" in prompt_content


@pytest.mark.vcr
def test_agent_run_streaming_with_tools(instrument, span_exporter, reader):
    """Test agent.run() with streaming and tool usage."""

    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    agent = Agent(
        name="SyncStreamToolAgent",
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[multiply],
        description="A sync streaming agent with tools",
    )

    events = []
    for event in agent.run("What is 6 times 7?", stream=True):
        events.append(event)

    assert len(events) > 0

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1

    agent_span = [s for s in spans if "agent" in s.name][-1]
    assert agent_span.name == "SyncStreamToolAgent.agent"
    assert agent_span.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "agno"
    assert agent_span.attributes.get(GenAIAttributes.GEN_AI_AGENT_NAME) == "SyncStreamToolAgent"

    tool_spans = [s for s in spans if s.name == "multiply.tool"]
    for tool_span in tool_spans:
        assert tool_span.attributes.get(SpanAttributes.TRACELOOP_ENTITY_NAME) == "multiply"
        assert tool_span.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "agno"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_agent_arun_streaming(instrument, span_exporter, reader):
    """Test agent.arun() with streaming enabled."""
    agent = Agent(
        name="AsyncStreamAgent",
        model=OpenAIChat(id="gpt-4o-mini"),
        description="An async streaming test agent",
    )

    events = []
    async for event in agent.arun("What is 10 + 5?", stream=True):
        events.append(event)

    assert len(events) > 0

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1

    agent_span = spans[-1]
    assert agent_span.name == "AsyncStreamAgent.agent"
    assert agent_span.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "agno"
    assert agent_span.attributes.get(GenAIAttributes.GEN_AI_AGENT_NAME) == "AsyncStreamAgent"
    assert agent_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL) == "gpt-4o-mini"

    prompt_content = agent_span.attributes.get(SpanAttributes.TRACELOOP_ENTITY_INPUT)
    assert "10 + 5" in prompt_content


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_agent_arun_streaming_with_tools(instrument, span_exporter, reader):
    """Test agent.arun() with streaming and tool usage."""

    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    agent = Agent(
        name="StreamToolAgent",
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[multiply],
        description="A streaming agent with tools",
    )

    events = []
    async for event in agent.arun("What is 6 times 7?", stream=True):
        events.append(event)

    assert len(events) > 0

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1

    agent_span = [s for s in spans if "agent" in s.name][-1]
    assert agent_span.name == "StreamToolAgent.agent"
    assert agent_span.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "agno"
    assert agent_span.attributes.get(GenAIAttributes.GEN_AI_AGENT_NAME) == "StreamToolAgent"

    tool_spans = [s for s in spans if s.name == "multiply.tool"]
    for tool_span in tool_spans:
        assert tool_span.attributes.get(SpanAttributes.TRACELOOP_ENTITY_NAME) == "multiply"
        assert tool_span.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "agno"
