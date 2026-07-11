"""Regression tests for the @agent decorator's effect on workflow_name context.

Before the fix in `_setup_span`, @agent unconditionally wrote
`traceloop.workflow.name = <agent_name>` into the OTel context, clobbering
the name set by an enclosing @workflow. Any child span (LLM or manual)
created inside the agent then inherited the wrong workflow name, breaking
downstream aggregations that group by (agent_name, workflow_name).

These tests pin the fixed behavior:
- @agent nested inside @workflow inherits workflow_name from the workflow.
- The same agent name running under two different workflows stays distinct.
- A bare @agent (no enclosing @workflow) leaves workflow_name unset.
"""

from opentelemetry import trace
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_AGENT_NAME,
)
from opentelemetry.semconv_ai import SpanAttributes

from traceloop.sdk.decorators import agent, task, workflow


def _make_child_span(name: str) -> None:
    """Create and immediately end a manual span, simulating a child LLM call."""
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span(name):
        pass


def test_agent_inside_workflow_inherits_workflow_name(exporter):
    """@agent nested inside @workflow must NOT overwrite workflow_name."""

    @agent(name="planner")
    def planner_agent():
        _make_child_span("child.llm")

    @workflow(name="rag")
    def rag_workflow():
        planner_agent()

    rag_workflow()

    spans = exporter.get_finished_spans()
    by_name = {span.name: span for span in spans}

    workflow_span = by_name["rag.workflow"]
    agent_span = by_name["planner.agent"]
    child_span = by_name["child.llm"]

    assert workflow_span.attributes[SpanAttributes.TRACELOOP_WORKFLOW_NAME] == "rag"

    assert agent_span.attributes[SpanAttributes.TRACELOOP_WORKFLOW_NAME] == "rag"
    assert agent_span.attributes[GEN_AI_AGENT_NAME] == "planner"

    assert child_span.attributes[SpanAttributes.TRACELOOP_WORKFLOW_NAME] == "rag"
    assert child_span.attributes[GEN_AI_AGENT_NAME] == "planner"


def test_same_agent_under_two_workflows_stays_distinct(exporter):
    """Two workflows sharing one @agent name keep their own workflow_name on child spans.

    This is the scenario that motivated the fix: pre-fix, both rag and summarize
    workflows had child spans tagged workflow_name="planner", collapsing the
    aggregator's (agent_name, workflow_name) groups into one.
    """

    @agent(name="planner")
    def planner_for_rag():
        _make_child_span("rag.child")

    @agent(name="planner")
    def planner_for_summarize():
        _make_child_span("summarize.child")

    @workflow(name="rag")
    def rag_workflow():
        planner_for_rag()

    @workflow(name="summarize")
    def summarize_workflow():
        planner_for_summarize()

    @task(name="outer")
    def outer():
        rag_workflow()
        summarize_workflow()

    outer()

    spans = exporter.get_finished_spans()
    by_name = {span.name: span for span in spans}

    rag_child = by_name["rag.child"]
    summarize_child = by_name["summarize.child"]

    assert rag_child.attributes[SpanAttributes.TRACELOOP_WORKFLOW_NAME] == "rag"
    assert rag_child.attributes[GEN_AI_AGENT_NAME] == "planner"

    assert summarize_child.attributes[SpanAttributes.TRACELOOP_WORKFLOW_NAME] == "summarize"
    assert summarize_child.attributes[GEN_AI_AGENT_NAME] == "planner"


def test_bare_agent_does_not_set_workflow_name(exporter):
    """A bare @agent (no enclosing @workflow) must NOT set workflow_name.

    Pins the deliberate Option B semantics: an agent is not a workflow.
    Previously the agent's own name was used as workflow_name, which made
    `(agent_name, workflow_name)` groupings impossible to disambiguate.
    """

    @agent(name="solo")
    def solo_agent():
        _make_child_span("solo.child")

    solo_agent()

    spans = exporter.get_finished_spans()
    by_name = {span.name: span for span in spans}

    agent_span = by_name["solo.agent"]
    child_span = by_name["solo.child"]

    assert SpanAttributes.TRACELOOP_WORKFLOW_NAME not in agent_span.attributes
    assert agent_span.attributes[GEN_AI_AGENT_NAME] == "solo"

    assert SpanAttributes.TRACELOOP_WORKFLOW_NAME not in child_span.attributes
    assert child_span.attributes[GEN_AI_AGENT_NAME] == "solo"
