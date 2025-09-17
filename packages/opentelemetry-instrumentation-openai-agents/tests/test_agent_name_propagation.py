"""Test agent name propagation to child spans (tools and responses)."""

import pytest
from agents import Agent, Runner
from opentelemetry.semconv_ai import SpanAttributes, TraceloopSpanKindValues
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_COMPLETION,
    GEN_AI_AGENT_NAME,
)




@pytest.mark.vcr
def test_agent_name_propagation_to_response_spans_basic(exporter):
    """Test that agent names are properly propagated to response spans in simple queries."""

    # Create a simple agent for basic testing
    simple_agent = Agent(
        name="Simple Test Agent",
        instructions="You are a helpful assistant.",
        model="gpt-4o"
    )

    query = "Hello, how are you?"
    Runner.run_sync(simple_agent, query)

    spans = exporter.get_finished_spans()

    # Find the agent span and response spans
    agent_spans = [s for s in spans if s.name == "Simple Test Agent.agent"]
    response_spans = [s for s in spans if s.name == "openai.response"]

    assert len(agent_spans) == 1, f"Expected 1 agent span, got {len(agent_spans)}"
    assert len(response_spans) >= 1, f"Expected at least 1 response span, got {len(response_spans)}"

    agent_span = agent_spans[0]

    # Verify agent span has correct agent name
    assert agent_span.attributes[GEN_AI_AGENT_NAME] == "Simple Test Agent"

    # Verify all response spans have the agent name propagated to them
    for response_span in response_spans:
        assert GEN_AI_AGENT_NAME in response_span.attributes, "Response span should have agent name attribute"
        assert response_span.attributes[GEN_AI_AGENT_NAME] == "Simple Test Agent", (
            f"Response span should have agent name 'Simple Test Agent', "
            f"got '{response_span.attributes.get(GEN_AI_AGENT_NAME)}'"
        )

        # Verify response span has correct LLM-specific attributes
        assert response_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] in ["response", "chat"]
        assert response_span.attributes["gen_ai.system"] == "openai"


@pytest.mark.vcr
def test_agent_name_propagation_to_tool_spans(exporter, function_tool_agent):
    """Test that agent names are properly propagated to tool spans."""

    query = "What is the weather in London?"
    Runner.run_sync(function_tool_agent, query)

    spans = exporter.get_finished_spans()

    # Find the agent span and tool span
    agent_spans = [s for s in spans if s.name == "WeatherAgent.agent"]
    tool_spans = [s for s in spans if s.name == "get_weather.tool"]

    assert len(agent_spans) == 1, f"Expected 1 agent span, got {len(agent_spans)}"
    assert len(tool_spans) == 1, f"Expected 1 tool span, got {len(tool_spans)}"

    agent_span = agent_spans[0]
    tool_span = tool_spans[0]

    # Verify agent span has correct agent name
    assert agent_span.attributes[GEN_AI_AGENT_NAME] == "WeatherAgent"

    # Verify tool span has the agent name propagated to it
    assert GEN_AI_AGENT_NAME in tool_span.attributes, "Tool span should have agent name attribute"
    assert tool_span.attributes[GEN_AI_AGENT_NAME] == "WeatherAgent", (
        f"Tool span should have agent name 'WeatherAgent', "
        f"got '{tool_span.attributes.get(GEN_AI_AGENT_NAME)}'"
    )

    # Verify tool span has correct tool-specific attributes
    assert tool_span.attributes[SpanAttributes.TRACELOOP_SPAN_KIND] == TraceloopSpanKindValues.TOOL.value
    assert tool_span.attributes[f"{GEN_AI_COMPLETION}.tool.name"] == "get_weather"


