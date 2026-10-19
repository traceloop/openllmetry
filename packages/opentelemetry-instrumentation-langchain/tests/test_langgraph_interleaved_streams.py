"""Regression tests for interleaved and abandoned LangGraph stream sessions.

Two sessions whose streams are read interleaved (or whose async stream is
abandoned mid-read) must stay in separate traces; a suspended or abandoned
session must not adopt another session's spans, and no "Failed to detach
context" errors may surface.
"""

import asyncio
import gc
import json

import pytest

pytest.importorskip("langgraph")

import httpx
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from typing_extensions import Annotated, TypedDict


class _State(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def _lookup(q: str) -> str:
    """Look something up."""
    return f"result for {q}"


def _mock_llm(seen):
    def reply(request):
        body = json.loads(request.content)
        session = body["messages"][0]["content"].split()[0]
        n = seen[session] = seen.get(session, 0) + 1
        if n == 1:
            msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": f"call_{session}",
                    "type": "function",
                    "function": {"name": "_lookup", "arguments": json.dumps({"q": session})},
                }],
            }
            finish = "tool_calls"
        else:
            msg = {"role": "assistant", "content": f"{session} answer"}
            finish = "stop"
        if not body.get("stream"):
            return httpx.Response(
                200,
                json={
                    "id": f"c-{session}{n}",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "m",
                    "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                },
            )
        base = {"id": f"c-{session}{n}", "object": "chat.completion.chunk", "created": 0, "model": "m"}
        delta = dict(msg, tool_calls=[dict(msg["tool_calls"][0], index=0)]) if n == 1 else msg
        chunks = [
            dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": None}]),
            dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": finish}]),
        ]
        data = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=data.encode())

    return ChatOpenAI(
        model="m",
        api_key="x",
        base_url="http://mock/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(reply)),
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(reply)),
    )


def _build_graph(llm, session):
    model = llm.bind_tools([_lookup])
    system = SystemMessage(f"{session} you are a helper")

    def agent(state):
        return {"messages": [model.invoke([system] + state["messages"])]}

    async def aagent(state):
        return {"messages": [await model.ainvoke([system] + state["messages"])]}

    from langchain_core.runnables import RunnableLambda

    g = StateGraph(_State)
    g.add_node("agent", RunnableLambda(agent, aagent, name="agent"))
    g.add_node("tools", ToolNode([_lookup]))
    g.add_edge(START, "agent")
    g.add_edge("tools", "agent")
    g.add_conditional_edges(
        "agent", lambda s: "tools" if s["messages"][-1].tool_calls else END, ["tools", END]
    )
    return g.compile(checkpointer=InMemorySaver())


def _sessions_by_trace(spans):
    out = {}
    for span in spans:
        text = json.dumps({k: str(v) for k, v in (span.attributes or {}).items()})
        for session in ("alpha", "bravo"):
            if session in text:
                out.setdefault(span.context.trace_id, set()).add(session)
    return [sorted(v) for v in out.values()]


@pytest.fixture
def span_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    LangchainInstrumentor().instrument(tracer_provider=provider)
    yield exporter
    LangchainInstrumentor().uninstrument()


def test_interleaved_stream_sessions_stay_in_separate_traces(span_exporter):
    seen = {}
    llm = _mock_llm(seen)
    cfg = lambda s: {"configurable": {"thread_id": s}}

    alpha = _build_graph(llm, "alpha").stream(
        {"messages": [("user", "alpha question")]}, cfg("alpha"), stream_mode="updates"
    )
    next(alpha)
    list(
        _build_graph(llm, "bravo").stream(
            {"messages": [("user", "bravo question")]}, cfg("bravo"), stream_mode="updates"
        )
    )
    alpha.close()

    from opentelemetry.trace import get_current_span
    from opentelemetry.trace.span import INVALID_SPAN

    current = get_current_span()
    assert current is INVALID_SPAN, f"context leaked after close: {getattr(current, 'name', current)}"

    sessions = _sessions_by_trace(span_exporter.get_finished_spans())
    assert ["alpha"] in sessions and ["bravo"] in sessions
    assert ["alpha", "bravo"] not in sessions


def test_abandoned_async_stream_does_not_adopt_next_session(span_exporter):
    seen = {}
    llm = _mock_llm(seen)
    cfg = lambda s: {"configurable": {"thread_id": s}}

    async def abandoned_then_new():
        stream = _build_graph(llm, "alpha").astream(
            {"messages": [("user", "alpha question")]}, cfg("alpha2"), stream_mode="messages"
        )
        await stream.__anext__()
        del stream
        gc.collect()
        await _build_graph(llm, "bravo").ainvoke(
            {"messages": [("user", "bravo question")]}, cfg("bravo2")
        )

    asyncio.run(abandoned_then_new())

    sessions = _sessions_by_trace(span_exporter.get_finished_spans())
    assert ["alpha"] in sessions and ["bravo"] in sessions
    assert ["alpha", "bravo"] not in sessions
