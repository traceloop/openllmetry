import pytest
from openai import OpenAI
from typing import TypedDict
from langgraph.graph import StateGraph
from opentelemetry import trace
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import INVALID_SPAN


@pytest.mark.vcr
def test_langgraph_invoke(instrument_legacy, span_exporter):
    client = OpenAI()

    class State(TypedDict):
        request: str
        result: str

    def calculate(state: State):
        request = state["request"]
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a mathematician."},
                {"role": "user", "content": request},
            ],
        )
        return {"result": completion.choices[0].message.content}

    workflow = StateGraph(State)
    workflow.add_node("calculate", calculate)
    workflow.set_entry_point("calculate")

    langgraph = workflow.compile()

    user_request = "What's 5 + 5?"
    response = langgraph.invoke(input={"request": user_request})["result"]
    spans = span_exporter.get_finished_spans()
    assert set(["LangGraph.workflow", "calculate.task", "openai.chat"]) == set(
        [span.name for span in spans]
    )

    openai_span = next(span for span in spans if span.name == "openai.chat")
    calculate_task_span = next(span for span in spans if span.name == "calculate.task")

    assert openai_span.parent.span_id == calculate_task_span.context.span_id
    assert openai_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert openai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-4o"
    assert (
        openai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
    ) == "You are a mathematician."
    assert (openai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"]) == "system"
    assert (
        openai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.content"]
    ) == user_request
    assert (openai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.role"]) == "user"
    assert (
        openai_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"]
        == response
    )
    assert (
        openai_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role"]
    ) == "assistant"

    assert openai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 24
    assert openai_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 11
    assert openai_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 35


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_langgraph_ainvoke(instrument_legacy, span_exporter):
    client = OpenAI()

    class State(TypedDict):
        request: str
        result: str

    def calculate(state: State):
        request = state["request"]
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a mathematician."},
                {"role": "user", "content": request},
            ],
        )
        return {"result": completion.choices[0].message.content}

    workflow = StateGraph(State)
    workflow.add_node("calculate", calculate)
    workflow.set_entry_point("calculate")

    langgraph = workflow.compile()

    user_request = "What's 5 + 5?"
    await langgraph.ainvoke(input={"request": user_request})
    spans = span_exporter.get_finished_spans()
    assert set(["LangGraph.workflow", "calculate.task", "openai.chat"]) == set(
        [span.name for span in spans]
    )
    openai_span = next(span for span in spans if span.name == "openai.chat")
    calculate_task_span = next(span for span in spans if span.name == "calculate.task")
    assert openai_span.parent.span_id == calculate_task_span.context.span_id


@pytest.mark.vcr
def test_langgraph_double_invoke(instrument_legacy, span_exporter):
    class DummyGraphState(TypedDict):
        result: str

    def mynode_func(state: DummyGraphState) -> DummyGraphState:
        return state

    def build_graph():
        workflow = StateGraph(DummyGraphState)
        workflow.add_node("mynode", mynode_func)
        workflow.set_entry_point("mynode")
        langgraph = workflow.compile()
        return langgraph

    graph = build_graph()

    from opentelemetry import trace

    assert trace.get_current_span() == INVALID_SPAN

    graph.invoke({"result": "init"})
    assert trace.get_current_span() == INVALID_SPAN

    spans = span_exporter.get_finished_spans()
    assert [
        "mynode.task",
        "LangGraph.workflow",
    ] == [span.name for span in spans]

    graph.invoke({"result": "init"})
    assert trace.get_current_span() == INVALID_SPAN

    spans = span_exporter.get_finished_spans()
    assert [
        "mynode.task",
        "LangGraph.workflow",
        "mynode.task",
        "LangGraph.workflow",
    ] == [span.name for span in spans]


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_langgraph_double_ainvoke(instrument_legacy, span_exporter):
    class DummyGraphState(TypedDict):
        result: str

    def mynode_func(state: DummyGraphState) -> DummyGraphState:
        return state

    def build_graph():
        workflow = StateGraph(DummyGraphState)
        workflow.add_node("mynode", mynode_func)
        workflow.set_entry_point("mynode")
        langgraph = workflow.compile()
        return langgraph

    graph = build_graph()

    assert trace.get_current_span() == INVALID_SPAN

    await graph.ainvoke({"result": "init"})

    spans = span_exporter.get_finished_spans()
    assert [
        "mynode.task",
        "LangGraph.workflow",
    ] == [span.name for span in spans]

    await graph.ainvoke({"result": "init"})

    spans = span_exporter.get_finished_spans()
    assert [
        "mynode.task",
        "LangGraph.workflow",
        "mynode.task",
        "LangGraph.workflow",
    ] == [span.name for span in spans]


@pytest.mark.vcr
def test_nesting_of_langgraph_spans(instrument_legacy, span_exporter, tracer_provider):
    """Test that exactly reproduces the GitHub issue #3203 with the exact same code structure."""
    from opentelemetry import trace
    import asyncio
    import httpx
    from langgraph.graph import END, START, StateGraph

    trace.set_tracer_provider(tracer_provider)
    tracer = trace.get_tracer(__name__)

    class TestAgentState(TypedDict):
        http_result: str
        span_result: str
        messages: list

    async def http_call_node(state: TestAgentState) -> dict:
        try:
            data = {"a": 10, "b": 25}
            async with httpx.AsyncClient() as _:
                with tracer.start_as_current_span("POST") as span:
                    span.set_attribute("http.method", "POST")
                    span.set_attribute("http.url", "https://httpbin.org/post")
                    sum_result = data.get("a", 0) + data.get("b", 0)
                    http_result = f"HTTP call successful! Sum of {data.get('a')} + {data.get('b')} = {sum_result}"

                    span.set_attribute("http.response.status_code", 200)
                    span.set_attribute("calculation.result", sum_result)

        except Exception as e:
            http_result = f"HTTP call error: {str(e)}"

        return {"http_result": http_result}

    async def opentelemetry_span_node(state: TestAgentState) -> dict:
        with tracer.start_as_current_span("test_agent_span") as span:
            span.set_attribute("node.name", "opentelemetry_span_node")
            span.set_attribute("agent.type", "test_agent")
            span.set_attribute("operation.type", "span_creation")

            span.add_event("Starting span processing")

            await asyncio.sleep(0.01)

            http_result = state.get("http_result", "No HTTP result available")
            span.set_attribute("previous.http_result", http_result)

            span.add_event("Processing HTTP result from previous node")

            span_result = f"OpenTelemetry span created successfully! Span ID: {span.get_span_context().span_id}"

            span.add_event("Span processing completed")
            span.set_attribute("processing.status", "completed")

        return {"span_result": span_result}

    def create_test_agent():
        """Create a simple LangGraph agent with 2 nodes matching the GitHub issue exactly."""
        builder = StateGraph(TestAgentState)

        builder.add_node("http_call", http_call_node)
        builder.add_node("otel_span", opentelemetry_span_node)

        builder.add_edge(START, "http_call")
        builder.add_edge("http_call", "otel_span")
        builder.add_edge("otel_span", END)

        agent = builder.compile()
        return agent

    async def run_test_agent():
        with tracer.start_as_current_span("test_agent_execution_root") as root_span:
            root_span.set_attribute("agent.name", "test_agent")
            root_span.set_attribute("agent.version", "1.0.0")
            root_span.set_attribute("execution.type", "full_agent_run")

            root_span.add_event("Agent execution started")

            try:
                root_span.add_event("Creating agent graph")
                agent = create_test_agent()
                root_span.set_attribute("agent.nodes_count", 2)

                initial_state = {"http_result": "", "span_result": "", "messages": []}
                root_span.add_event("Initial state prepared")

                root_span.add_event("Starting agent invocation")
                final_state = await agent.ainvoke(initial_state)

                root_span.set_attribute("execution.status", "completed")
                return final_state

            except Exception as e:
                root_span.set_attribute("execution.status", "failed")
                root_span.set_attribute("error.type", type(e).__name__)
                root_span.set_attribute("error.message", str(e))
                root_span.add_event("Agent execution failed", {"error": str(e)})
                raise

    final_state = asyncio.run(run_test_agent())

    assert "http_result" in final_state
    assert "span_result" in final_state
    assert "Sum of 10 + 25 = 35" in final_state["http_result"]

    spans = span_exporter.get_finished_spans()
    span_names = [span.name for span in spans]

    print(f"\nCaptured {len(spans)} spans:")
    for span in spans:
        parent_name = "None"
        if span.parent:
            parent_span = next(
                (s for s in spans if s.context.span_id == span.parent.span_id), None
            )
            if parent_span:
                parent_name = parent_span.name
            else:
                parent_name = f"Unknown({span.parent.span_id})"
        print(f"  - {span.name} (parent: {parent_name})")

    assert "test_agent_execution_root" in span_names
    assert "POST" in span_names
    assert "test_agent_span" in span_names
    assert "http_call.task" in span_names
    assert "otel_span.task" in span_names
    assert "LangGraph.workflow" in span_names

    root_span = next(span for span in spans if span.name == "test_agent_execution_root")
    post_span = next(span for span in spans if span.name == "POST")
    test_agent_span = next(span for span in spans if span.name == "test_agent_span")
    http_call_task_span = next(span for span in spans if span.name == "http_call.task")
    otel_span_task_span = next(span for span in spans if span.name == "otel_span.task")
    workflow_span = next(span for span in spans if span.name == "LangGraph.workflow")

    print("\nHierarchy check:")
    print(f"POST parent: {post_span.parent.span_id if post_span.parent else 'None'}")
    print(f"http_call.task ID: {http_call_task_span.context.span_id}")
    print(
        f"test_agent_span parent: {test_agent_span.parent.span_id if test_agent_span.parent else 'None'}"
    )
    print(f"otel_span.task ID: {otel_span_task_span.context.span_id}")

    assert (
        post_span.parent.span_id == http_call_task_span.context.span_id
    ), "POST span should be child of http_call.task span"
    assert (
        test_agent_span.parent.span_id == otel_span_task_span.context.span_id
    ), "test_agent_span should be child of otel_span.task span"

    assert http_call_task_span.parent.span_id == workflow_span.context.span_id
    assert otel_span_task_span.parent.span_id == workflow_span.context.span_id
    assert workflow_span.parent.span_id == root_span.context.span_id


def test_context_detachment_error_handling(
    instrument_legacy, span_exporter, tracer_provider, caplog
):
    """
    Test that context detachment errors are handled properly without logging.

    This test specifically validates the fix for the issue where OpenTelemetry
    context detachment failures in async scenarios would cause error logging:
    'ERROR:opentelemetry.context:Failed to detach context'

    The test creates conditions that trigger context tokens to be created in
    one context and detached in another, which previously caused ValueError
    exceptions to be logged by OpenTelemetry's context_api.detach().
    """
    import asyncio
    import logging
    from opentelemetry import trace
    from langgraph.graph import END, START, StateGraph

    trace.set_tracer_provider(tracer_provider)
    tracer = trace.get_tracer(__name__)

    with caplog.at_level(logging.ERROR):

        class AsyncTestState(TypedDict):
            counter: int
            result: str

        async def concurrent_span_node(state: AsyncTestState) -> dict:
            """Node that creates spans in async context, triggering potential context issues."""
            with tracer.start_as_current_span("concurrent_async_span") as span:
                span.set_attribute("node.type", "concurrent_async")
                span.set_attribute("input.counter", state["counter"])

                await asyncio.sleep(0.001)

                with tracer.start_as_current_span("nested_span") as nested_span:
                    nested_span.set_attribute("nested.work", True)
                    await asyncio.sleep(0.001)

                result = f"processed_{state['counter']}"
                span.set_attribute("output.result", result)

                return {"counter": state["counter"] + 1, "result": result}

        async def parallel_processing_node(state: AsyncTestState) -> dict:
            """Node that processes multiple tasks in parallel, stressing context management."""

            async def parallel_task(task_id: int):
                with tracer.start_as_current_span(f"parallel_task_{task_id}") as span:
                    span.set_attribute("task.id", task_id)
                    await asyncio.sleep(0.001)
                    return f"task_{task_id}_done"

            tasks = [parallel_task(i) for i in range(5)]
            parallel_results = await asyncio.gather(*tasks)
            combined_result = (
                f"{state['result']} + parallel_results: {','.join(parallel_results)}"
            )
            return {"counter": state["counter"], "result": combined_result}

        def build_context_stress_graph():
            """Build a graph designed to stress context management."""
            builder = StateGraph(AsyncTestState)
            builder.add_node("concurrent", concurrent_span_node)
            builder.add_node("parallel", parallel_processing_node)

            builder.add_edge(START, "concurrent")
            builder.add_edge("concurrent", "parallel")
            builder.add_edge("parallel", END)

            return builder.compile()

        async def run_concurrent_executions():
            """Run multiple concurrent graph executions to trigger context issues."""
            graph = build_context_stress_graph()

            tasks = []
            for i in range(10):
                task = graph.ainvoke({"counter": i, "result": ""})
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results

        results = asyncio.run(run_concurrent_executions())

        assert len(results) == 10
        for i, result in enumerate(results):
            assert not isinstance(result, Exception), f"Execution {i} failed: {result}"
            assert result["counter"] == i + 1
            assert f"processed_{i}" in result["result"]

        spans = span_exporter.get_finished_spans()

        assert len(spans) >= 100, f"Expected at least 100 spans, got {len(spans)}"

        workflow_spans = [s for s in spans if s.name == "LangGraph.workflow"]
        concurrent_spans = [s for s in spans if s.name == "concurrent_async_span"]
        nested_spans = [s for s in spans if s.name == "nested_span"]
        parallel_task_spans = [s for s in spans if s.name.startswith("parallel_task_")]

        assert (
            len(workflow_spans) == 10
        ), f"Expected 10 workflow spans, got {len(workflow_spans)}"
        assert (
            len(concurrent_spans) == 10
        ), f"Expected 10 concurrent spans, got {len(concurrent_spans)}"
        assert (
            len(nested_spans) == 10
        ), f"Expected 10 nested spans, got {len(nested_spans)}"
        assert (
            len(parallel_task_spans) == 50
        ), f"Expected 50 parallel task spans, got {len(parallel_task_spans)}"

        error_logs = [
            record.message
            for record in caplog.records
            if record.levelno >= logging.ERROR
        ]
        context_errors = [
            msg for msg in error_logs if "Failed to detach context" in msg
        ]

        assert len(context_errors) == 0, (
            f"Found {len(context_errors)} context detachment errors in logs. "
            f"This indicates the fix is not working properly. Errors: {context_errors}"
        )

        for nested_span in nested_spans:
            assert nested_span.parent is not None, "Nested spans should have parents"
            parent_span = next(
                (s for s in spans if s.context.span_id == nested_span.parent.span_id),
                None,
            )
            assert parent_span is not None, "Parent span should exist"
            assert (
                parent_span.name == "concurrent_async_span"
            ), "Nested span should be child of concurrent_async_span"
