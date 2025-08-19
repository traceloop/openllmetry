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
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
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
    assert (
        span.attributes[SpanAttributes.TRACELOOP_SPAN_KIND]
        == TraceloopSpanKindValues.AGENT.value
    )
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.3
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 1024
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.2
    assert span.attributes["openai.agent.model.frequency_penalty"] == 1.3
    assert span.attributes["gen_ai.agent.name"] == "testAgent"
    assert (
        span.attributes["gen_ai.agent.description"]
        == "You are a helpful assistant that answers all questions"
    )

    assert span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"] == "user"
    assert span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"] == "What is AI?"

    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] is not None
    assert (
        span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        is not None)
    assert span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] is not None

    assert span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.contents"] is not None
    assert len(span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.contents"]) > 0
    assert span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.roles"] is not None
    assert len(span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.roles"]) > 0
    assert span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.types"] is not None
    assert len(span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.types"]) > 0

    assert span.status.status_code == StatusCode.OK


@pytest.mark.vcr
def test_agent_with_function_tool_spans(exporter, function_tool_agent):
    query = "What is the weather in London?"
    Runner.run_sync(
        function_tool_agent,
        query,
    )
    spans = exporter.get_finished_spans()

    assert len(spans) == 3

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

    assert tool_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.tool.name"] == "get_weather"
    assert tool_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.tool.type"] == "FunctionTool"
    assert (
        tool_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.tool.description"]
        == "Gets the current weather for a specified city."
    )

    assert tool_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.tool.strict_json_schema"] is True

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

    assert tool_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.tool.type"] == "WebSearchTool"
    assert (
        tool_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.tool.search_context_size"]
        is not None
    )
    assert f"{GenAIAttributes.GEN_AI_COMPLETION}.tool.user_location" not in tool_span.attributes

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
    handoff0_info = json.loads(triage_agent_span.attributes["openai.agent.handoff0"])
    assert handoff0_info["name"] == "AgentA"
    assert handoff0_info["instructions"] == "Agent A does something."

    assert triage_agent_span.attributes["openai.agent.handoff1"] is not None
    handoff1_info = json.loads(triage_agent_span.attributes["openai.agent.handoff1"])
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
                        assert data_point.attributes[GenAIAttributes.GEN_AI_TOKEN_TYPE] in [
                            "output",
                            "input",
                        ]
                        assert data_point.count > 0
                        assert data_point.sum > 0

                if metric.name == Meters.LLM_OPERATION_DURATION:
                    found_duration_metric = True
                    assert any(
                        data_point.count > 0 for data_point in metric.data.data_points
                    )
                    assert any(
                        data_point.sum > 0 for data_point in metric.data.data_points
                    )

        assert found_token_metric is True
        assert found_duration_metric is True


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_recipe_workflow_agent_handoffs_with_function_tools(
    exporter, recipe_workflow_agents
):
    """Test agent handoffs with function tools matching the recipe management example."""

    main_chat_agent, recipe_editor_agent = recipe_workflow_agents

    query = "Can you edit the carbonara recipe to be vegetarian?"

    messages = [{"role": "user", "content": query}]
    main_runner = Runner().run_streamed(starting_agent=main_chat_agent, input=messages)

    handoff_info = None
    async for event in main_runner.stream_events():
        if event.type == "run_item_stream_event" and event.name == "handoff_occurred":
            handoff_info = event.item.raw_item

    if handoff_info and "recipe" in str(handoff_info).lower():
        recipe_messages = [{"role": "user", "content": query}]
        recipe_runner = Runner().run_streamed(
            starting_agent=recipe_editor_agent, input=recipe_messages
        )
        async for _ in recipe_runner.stream_events():
            pass

    spans = exporter.get_finished_spans()
    non_rest_spans = [span for span in spans if not span.name.endswith("v1/responses")]
    span_names = [span.name for span in non_rest_spans]

    assert span_names.count("Main Chat Agent.agent") == 1
    assert span_names.count("Recipe Editor Agent.agent") == 3
    assert span_names.count("search_recipes.tool") == 1
    assert span_names.count("plan_and_apply_recipe_modifications.tool") == 1

    assert "Main Chat Agent.agent" in span_names
    assert "Recipe Editor Agent.agent" in span_names

    assert "search_recipes.tool" in span_names
    assert "plan_and_apply_recipe_modifications.tool" in span_names

    main_chat_span = next(
        s for s in non_rest_spans if s.name == "Main Chat Agent.agent"
    )
    recipe_editor_spans = [
        s for s in non_rest_spans if s.name == "Recipe Editor Agent.agent"
    ]
    search_tool_span = next(
        s for s in non_rest_spans if s.name == "search_recipes.tool"
    )
    modify_tool_span = next(
        s
        for s in non_rest_spans
        if s.name == "plan_and_apply_recipe_modifications.tool"
    )

    assert main_chat_span.attributes["gen_ai.agent.name"] == "Main Chat Agent"
    assert (
        main_chat_span.attributes[SpanAttributes.TRACELOOP_SPAN_KIND]
        == TraceloopSpanKindValues.AGENT.value
    )

    assert "traceloop.entity.input" in main_chat_span.attributes
    assert "traceloop.entity.output" in main_chat_span.attributes

    # Validate that input and output are valid JSON
    main_chat_input = json.loads(main_chat_span.attributes["traceloop.entity.input"])
    main_chat_output = json.loads(main_chat_span.attributes["traceloop.entity.output"])
    assert isinstance(main_chat_input, dict)
    assert isinstance(main_chat_output, dict)

    assert "openai.agent.handoff0" in main_chat_span.attributes
    handoff_info = json.loads(main_chat_span.attributes["openai.agent.handoff0"])
    assert handoff_info["name"] == "Recipe Editor Agent"

    recipe_editor_span = recipe_editor_spans[0]
    assert recipe_editor_span.attributes["gen_ai.agent.name"] == "Recipe Editor Agent"
    assert (
        recipe_editor_span.attributes[SpanAttributes.TRACELOOP_SPAN_KIND]
        == TraceloopSpanKindValues.AGENT.value
    )

    assert "traceloop.entity.input" in recipe_editor_span.attributes
    assert "traceloop.entity.output" in recipe_editor_span.attributes

    # Validate that input and output are valid JSON
    recipe_editor_input = json.loads(
        recipe_editor_span.attributes["traceloop.entity.input"]
    )
    recipe_editor_output = json.loads(
        recipe_editor_span.attributes["traceloop.entity.output"]
    )
    assert isinstance(recipe_editor_input, dict)
    assert isinstance(recipe_editor_output, dict)

    assert (
        search_tool_span.attributes[SpanAttributes.TRACELOOP_SPAN_KIND]
        == TraceloopSpanKindValues.TOOL.value
    )
    assert (
        search_tool_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.tool.name"]
        == "search_recipes"
    )
    assert (
        search_tool_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.tool.type"] == "FunctionTool"
    )

    assert "traceloop.entity.input" in search_tool_span.attributes
    assert "traceloop.entity.output" in search_tool_span.attributes

    assert (
        modify_tool_span.attributes[SpanAttributes.TRACELOOP_SPAN_KIND]
        == TraceloopSpanKindValues.TOOL.value
    )
    assert (
        modify_tool_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.tool.name"]
        == "plan_and_apply_recipe_modifications"
    )
    assert (
        modify_tool_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.tool.type"] == "FunctionTool"
    )

    assert "traceloop.entity.input" in modify_tool_span.attributes
    assert "traceloop.entity.output" in modify_tool_span.attributes

    assert main_chat_span.parent is None

    assert search_tool_span.parent is not None
    assert modify_tool_span.parent is not None

    assert main_chat_span.status.status_code == StatusCode.OK
    for span in recipe_editor_spans:
        assert span.status.status_code == StatusCode.OK
    assert search_tool_span.status.status_code == StatusCode.OK
    assert modify_tool_span.status.status_code == StatusCode.OK

    main_trace_id = main_chat_span.get_span_context().trace_id
    all_trace_ids = {main_trace_id}

    for span in recipe_editor_spans:
        span_trace_id = span.get_span_context().trace_id
        assert span_trace_id == main_trace_id
        all_trace_ids.add(span_trace_id)

    assert search_tool_span.get_span_context().trace_id == main_trace_id
    all_trace_ids.add(search_tool_span.get_span_context().trace_id)

    assert modify_tool_span.get_span_context().trace_id == main_trace_id
    all_trace_ids.add(modify_tool_span.get_span_context().trace_id)

    assert len(all_trace_ids) == 1
