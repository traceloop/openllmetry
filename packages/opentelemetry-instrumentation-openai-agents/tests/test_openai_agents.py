import pytest
import json
from unittest.mock import MagicMock
from opentelemetry.instrumentation.openai_agents import (
    OpenAIAgentsInstrumentor,
)
from agents import Runner
from opentelemetry.trace import StatusCode
from opentelemetry.semconv_ai import (
    SpanAttributes,
    TraceloopSpanKindValues,
    Meters,
)
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_COMPLETION,
)


@pytest.fixture
def mock_instrumentor():
    instrumentor = OpenAIAgentsInstrumentor()
    instrumentor.instrument = MagicMock()
    instrumentor.uninstrument = MagicMock()
    return instrumentor


@pytest.mark.vcr
def test_agent_spans(exporter, test_agent):
    query = "What is AI?"
    Runner.run_sync(
        test_agent,
        query,
    )
    spans = exporter.get_finished_spans()

    span = spans[0]

    assert span.name == "testAgent.agent"
    assert span.kind == span.kind.CLIENT
    assert span.attributes[SpanAttributes.GEN_AI_SYSTEM] == "openai"
    assert span.attributes[SpanAttributes.GEN_AI_REQUEST_MODEL] == "gpt-4.1"
    assert (
        span.attributes[SpanAttributes.TRACELOOP_SPAN_KIND]
        == TraceloopSpanKindValues.AGENT.value
    )
    assert span.attributes[SpanAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.3
    assert span.attributes[SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 1024
    assert span.attributes[SpanAttributes.GEN_AI_REQUEST_TOP_P] == 0.2
    assert span.attributes["openai.agent.model.frequency_penalty"] == 1.3
    assert span.attributes["gen_ai.agent.name"] == "testAgent"
    assert (
        span.attributes["gen_ai.agent.description"]
        == "You are a helpful assistant that answers all questions"
    )

    assert span.attributes[f"{SpanAttributes.GEN_AI_PROMPT}.0.role"] == "user"
    assert (
        span.attributes[f"{SpanAttributes.GEN_AI_PROMPT}.0.content"]
        == "What is AI?")

    assert span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] is not None
    assert (
        span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS]
        is not None)
    assert span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] is not None

    assert (
        span.attributes[f"{SpanAttributes.GEN_AI_COMPLETION}.contents"]
        is not None
    )
    assert (
        len(span.attributes[f"{SpanAttributes.GEN_AI_COMPLETION}.contents"]) > 0
    )
    assert (
        span.attributes[f"{SpanAttributes.GEN_AI_COMPLETION}.roles"]
        is not None
    )
    assert (
        len(span.attributes[f"{SpanAttributes.GEN_AI_COMPLETION}.roles"]) > 0
    )
    assert (
        span.attributes[f"{SpanAttributes.GEN_AI_COMPLETION}.types"]
        is not None
    )
    assert (
        len(span.attributes[f"{SpanAttributes.GEN_AI_COMPLETION}.types"]) > 0
    )

    assert span.status.status_code == StatusCode.OK


@pytest.mark.vcr
def test_agent_with_function_tool_spans(exporter, function_tool_agent):
    query = "What is the weather in London?"
    Runner.run_sync(
        function_tool_agent,
        query,
    )
    spans = exporter.get_finished_spans()

    assert len(spans) == 4

    agent_span = next(s for s in spans if s.name == "WeatherAgent.agent")
    tool_span = next(s for s in spans if s.name == "get_weather.tool")

    assert (
        agent_span.attributes[SpanAttributes.TRACELOOP_SPAN_KIND]
        == TraceloopSpanKindValues.AGENT.value
    )
    assert (
        tool_span.attributes[SpanAttributes.TRACELOOP_SPAN_KIND]
        == TraceloopSpanKindValues.TOOL.value
    )
    assert tool_span.kind == tool_span.kind.INTERNAL

    assert (
        tool_span.attributes[f"{GEN_AI_COMPLETION}.tool.name"]
        == "get_weather"
    )
    assert (
        tool_span.attributes[f"{GEN_AI_COMPLETION}.tool.type"]
        == "FunctionTool"
    )
    assert (
        tool_span.attributes[f"{GEN_AI_COMPLETION}.tool.description"]
        == "Gets the current weather for a specified city."
    )

    assert (
        tool_span.attributes[f"{GEN_AI_COMPLETION}.tool.strict_json_schema"]
        is True
    )

    assert agent_span.status.status_code == StatusCode.OK
    assert tool_span.status.status_code == StatusCode.OK


@pytest.mark.vcr
def test_agent_with_web_search_tool_spans(exporter, web_search_tool_agent):
    query = "Search for latest news on AI."
    Runner.run_sync(
        web_search_tool_agent,
        query,
    )
    spans = exporter.get_finished_spans()

    assert len(spans) == 2

    agent_span = next(s for s in spans if s.name == "SearchAgent.agent")
    tool_span = next(s for s in spans if s.name == "WebSearchTool.tool")

    assert (
        agent_span.attributes[SpanAttributes.TRACELOOP_SPAN_KIND]
        == TraceloopSpanKindValues.AGENT.value
    )
    assert (
        tool_span.attributes[SpanAttributes.TRACELOOP_SPAN_KIND]
        == TraceloopSpanKindValues.TOOL.value
    )
    assert tool_span.kind == tool_span.kind.INTERNAL

    assert (
        tool_span.attributes[f"{GEN_AI_COMPLETION}.tool.type"]
        == "WebSearchTool"
    )
    assert (
        tool_span.attributes[f"{GEN_AI_COMPLETION}.tool.search_context_size"]
        is not None
    )
    assert (
        f"{GEN_AI_COMPLETION}.tool.user_location"
        not in tool_span.attributes
    )

    assert agent_span.status.status_code == StatusCode.OK
    assert tool_span.status.status_code == StatusCode.OK


@pytest.mark.vcr
def test_agent_with_handoff_spans(exporter, handoff_agent):

    query = "Please handle this task by delegating to another agent."
    Runner.run_sync(
        handoff_agent,
        query,
    )
    spans = exporter.get_finished_spans()

    assert len(spans) >= 1
    triage_agent_span = next(s for s in spans if s.name == "TriageAgent.agent")

    assert triage_agent_span.attributes["openai.agent.handoff0"] is not None
    handoff0_info = json.loads(
        triage_agent_span.attributes["openai.agent.handoff0"]
    )
    assert handoff0_info["name"] == "AgentA"
    assert handoff0_info["instructions"] == "Agent A does something."

    assert triage_agent_span.attributes["openai.agent.handoff1"] is not None
    handoff1_info = json.loads(
        triage_agent_span.attributes["openai.agent.handoff1"]
    )
    assert handoff1_info["name"] == "AgentB"
    assert handoff1_info["instructions"] == "Agent B does something else."

    assert triage_agent_span.status.status_code == StatusCode.OK


@pytest.mark.vcr
def test_generate_metrics(metrics_test_context, test_agent):

    provider, reader = metrics_test_context

    query = "What is AI?"
    Runner.run_sync(
        test_agent,
        query,
    )
    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics

    assert len(resource_metrics) > 0

    found_token_metric = False
    found_duration_metric = False

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:

                if metric.name == Meters.LLM_TOKEN_USAGE:
                    found_token_metric = True
                    for data_point in metric.data.data_points:
                        assert (
                            (
                                data_point.attributes[
                                    SpanAttributes.GEN_AI_TOKEN_TYPE
                                ]
                                in [
                                    "output",
                                    "input",
                                ]
                            )
                        )
                        assert data_point.count > 0
                        assert data_point.sum > 0

                if metric.name == Meters.LLM_OPERATION_DURATION:
                    found_duration_metric = True
                    assert any(
                        data_point.count > 0
                        for data_point in metric.data.data_points
                    )
                    assert any(
                        data_point.sum > 0
                        for data_point in metric.data.data_points
                    )
                assert (
                    metric.data.data_points[0].attributes[
                        SpanAttributes.GEN_AI_SYSTEM
                    ]
                    == "openai"
                )
                assert (
                    metric.data.data_points[0].attributes[
                        SpanAttributes.GEN_AI_RESPONSE_MODEL
                    ]
                    == "gpt-4.1"
                )

        assert found_token_metric is True
        assert found_duration_metric is True
