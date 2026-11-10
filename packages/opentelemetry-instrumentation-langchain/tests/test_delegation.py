"""Opt-in delegation / authority attributes on LangGraph agent and goto spans."""

from typing import TypedDict

import pytest
from langgraph.graph import END, StateGraph
from opentelemetry import context as context_api
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

# Same key the opentelemetry.semconv_ai.delegation helper writes to. Set directly so
# the test also runs against semconv-ai releases that predate the helper.
DELEGATION_KEY = "traceloop.delegation"
DELEGATION = {
    "traceloop.delegation.root_principal": "user:42",
    "traceloop.delegation.chain_id": "dlg_7f3a",
    "traceloop.delegation.depth": 1,
    "traceloop.delegation.scope_hash": "sha256:" + "ab" * 32,
    "traceloop.delegation.actor": "agent:router",
    "traceloop.delegation.chain_hash": "sha256:" + "cd" * 32,
    "traceloop.delegation.parent_scope_hash": "sha256:" + "ef" * 32,
    "traceloop.delegation.credential_ref": "spiffe://example.org/agent/router",
}


class State(TypedDict):
    """Graph state for the router test."""

    value: str


def _delegation_attrs(span):
    """Return only the delegation attributes recorded on ``span``."""
    return {k: v for k, v in span.attributes.items() if k.startswith("traceloop.delegation.")}


def _build_router_graph():
    """Build a two-node graph whose router node issues a ``Command(goto=...)``."""
    from langgraph.types import Command

    def router_node(state: State):
        """Route to the target node via a Command."""
        token = context_api.attach(context_api.set_value("langgraph_current_node", "router_node"))
        try:
            return Command(goto="target_node", update={"value": "routed"})
        finally:
            context_api.detach(token)

    def target_node(state: State):
        """Finish the run."""
        return {"value": "done"}

    workflow = StateGraph(State)
    workflow.add_node("router_node", router_node)
    workflow.add_node("target_node", target_node)
    workflow.set_entry_point("router_node")
    workflow.add_edge("target_node", END)
    return workflow.compile()


@pytest.fixture
def delegation_context():
    """Put the sample delegation attributes in the context for one test."""
    token = context_api.attach(context_api.set_value(DELEGATION_KEY, dict(DELEGATION)))
    yield
    context_api.detach(token)


def test_agent_and_goto_spans_record_delegation(instrument_legacy, span_exporter, delegation_context):
    """Agent and goto spans carry the delegation attributes; task spans don't."""
    _build_router_graph().invoke({"value": "start"}, config={"run_name": "Router"})

    spans = span_exporter.get_finished_spans()
    agent_span = next(s for s in spans if s.name == "invoke_agent Router")
    assert agent_span.attributes[GenAIAttributes.GEN_AI_AGENT_NAME] == "Router"
    assert _delegation_attrs(agent_span) == DELEGATION

    goto_span = next(s for s in spans if s.name.startswith("goto "))
    assert _delegation_attrs(goto_span) == DELEGATION

    # Plain node (task) spans don't repeat it.
    task_spans = [s for s in spans if s.name.startswith("execute_task ")]
    assert task_spans
    assert all(not _delegation_attrs(s) for s in task_spans)


def test_no_delegation_attributes_without_opt_in(instrument_legacy, span_exporter):
    """Without opt-in no span carries delegation attributes."""
    _build_router_graph().invoke({"value": "start"})

    spans = span_exporter.get_finished_spans()
    assert spans
    assert all(not _delegation_attrs(s) for s in spans)


def test_foreign_or_malformed_values_are_ignored(instrument_legacy, span_exporter):
    """Foreign keys and non str/int values never reach the span."""
    value = {
        "traceloop.delegation.root_principal": "user:1",
        "authorization": "Bearer secret",
        "traceloop.delegation.depth": True,
        "traceloop.delegation.chain_id": {"nested": "x"},
    }
    token = context_api.attach(context_api.set_value(DELEGATION_KEY, value))
    try:
        _build_router_graph().invoke({"value": "start"}, config={"run_name": "Router"})
    finally:
        context_api.detach(token)

    agent_span = next(s for s in span_exporter.get_finished_spans() if s.name == "invoke_agent Router")
    assert _delegation_attrs(agent_span) == {"traceloop.delegation.root_principal": "user:1"}
    assert "authorization" not in agent_span.attributes
