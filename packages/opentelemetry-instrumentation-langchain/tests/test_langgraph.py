import pytest
from openai import OpenAI
from typing import TypedDict
from langgraph.graph import StateGraph
from opentelemetry import trace
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_langgraph_invoke(exporter):
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
                {"role": "user", "content": request}
            ]
        )
        return {"result": completion.choices[0].message.content}
    workflow = StateGraph(State)
    workflow.add_node("calculate", calculate)
    workflow.set_entry_point("calculate")

    langgraph = workflow.compile()

    user_request = "What's 5 + 5?"
    response = langgraph.invoke(input={"request": user_request})["result"]
    spans = exporter.get_finished_spans()
    assert set(
        [
            "LangGraph.workflow",
            "calculate.task",
            "openai.chat"
        ]
    ) == set([span.name for span in spans])

    openai_span = next(span for span in spans if span.name == "openai.chat")
    calculate_task_span = next(span for span in spans if span.name == "calculate.task")

    assert openai_span.parent.span_id == calculate_task_span.context.span_id
    assert openai_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert openai_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-4o"
    assert (
        openai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
    ) == "You are a mathematician."
    assert (openai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"]) == "system"
    assert (
        openai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.content"]
    ) == user_request
    assert (openai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.role"]) == "user"
    assert (
        openai_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
        == response
    )
    assert (
        openai_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"]
    ) == "assistant"

    assert openai_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 24
    assert openai_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 11
    assert openai_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 35
    assert openai_span.attributes[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 0


@pytest.mark.vcr
@pytest.mark.asyncio
@pytest.mark.xfail(reason="Context propagation is not yet supported for async LangChain callbacks", strict=True)
async def test_langgraph_ainvoke(exporter):
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
                {"role": "user", "content": request}
            ]
        )
        return {"result": completion.choices[0].message.content}
    workflow = StateGraph(State)
    workflow.add_node("calculate", calculate)
    workflow.set_entry_point("calculate")

    langgraph = workflow.compile()

    user_request = "What's 5 + 5?"
    await langgraph.ainvoke(input={"request": user_request})
    spans = exporter.get_finished_spans()
    assert set(
        [
            "LangGraph.workflow",
            "calculate.task",
            "openai.chat"
        ]
    ) == set([span.name for span in spans])
    openai_span = next(span for span in spans if span.name == "openai.chat")
    calculate_task_span = next(span for span in spans if span.name == "calculate.task")
    assert openai_span.parent.span_id == calculate_task_span.context.span_id


@pytest.mark.vcr
def test_langgraph_double_invoke(exporter):
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

    assert "test_langgraph_double_invoke" == trace.get_current_span().name

    graph.invoke({"result": "init"})
    assert "test_langgraph_double_invoke" == trace.get_current_span().name

    spans = exporter.get_finished_spans()
    assert [
        "mynode.task",
        "LangGraph.workflow",
    ] == [span.name for span in spans]

    graph.invoke({"result": "init"})
    assert "test_langgraph_double_invoke" == trace.get_current_span().name

    spans = exporter.get_finished_spans()
    assert [
        "mynode.task",
        "LangGraph.workflow",
        "mynode.task",
        "LangGraph.workflow",
    ] == [span.name for span in spans]


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_langgraph_double_ainvoke(exporter):
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

    assert "test_langgraph_double_ainvoke" == trace.get_current_span().name

    await graph.ainvoke({"result": "init"})
    assert "test_langgraph_double_ainvoke" == trace.get_current_span().name

    spans = exporter.get_finished_spans()
    assert [
        "mynode.task",
        "LangGraph.workflow",
    ] == [span.name for span in spans]

    await graph.ainvoke({"result": "init"})
    assert "test_langgraph_double_ainvoke" == trace.get_current_span().name

    spans = exporter.get_finished_spans()
    assert [
        "mynode.task",
        "LangGraph.workflow",
        "mynode.task",
        "LangGraph.workflow",
    ] == [span.name for span in spans]
