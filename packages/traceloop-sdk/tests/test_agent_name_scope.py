"""Regression tests for gen_ai.agent.name / entity_path leaking across entities.

The @agent / @workflow / @task decorators wrote the entity name (and TASK/TOOL
entity path) into the OTel context with `attach()` but discarded the returned
token, so the value was never detached when the decorated function returned.
The innermost entity's name then stuck on the context and got stamped onto
every span that started afterwards in the same trace -- including spans
belonging to the *parent* entity.

The tests are grouped as:
  * leak reproducers -- fail before the fix, pass after (nested/sibling/async,
    plus a context-level check and a re-entry check);
  * guardrails -- pass before AND after, to catch over-detaching or blanking a
    name mid-unit.

See packages/sample-app/sample_app/faithfulness_traces/BUG_agent_name_leak.md
(shapes Q and T) for the discovery fixtures.
"""

import pytest
from opentelemetry import context as context_api, trace
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_AGENT_NAME,
)
from opentelemetry.semconv_ai import SpanAttributes

from traceloop.sdk.decorators import agent, task, workflow


def _make_child_span(name: str) -> None:
    """Create and immediately end a manual span, simulating a child LLM call.

    A real instrumented LLM span reads gen_ai.agent.name off the context at the
    moment it starts, which is exactly what this stand-in does.
    """
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span(name):
        pass


def test_nested_agent_name_does_not_leak_to_parent_terminal_span(exporter):
    """shape_Q: an orchestrator's terminal span keeps the orchestrator's name.

    research_lead (agent)
      child.plan           -> research_lead
      fact_checker (agent)
        child.subagent     -> fact_checker
      child.terminal       -> research_lead  (leaks to fact_checker before the fix)
    """

    @agent(name="fact_checker")
    def fact_checker_agent():
        _make_child_span("child.subagent")

    @agent(name="research_lead")
    def research_lead():
        _make_child_span("child.plan")       # own work, before nesting
        fact_checker_agent()
        _make_child_span("child.terminal")   # own work, after the sub-agent returned

    research_lead()

    by_name = {span.name: span for span in exporter.get_finished_spans()}

    assert by_name["child.plan"].attributes[GEN_AI_AGENT_NAME] == "research_lead"
    assert by_name["child.subagent"].attributes[GEN_AI_AGENT_NAME] == "fact_checker"
    # The crux: the parent's terminal span must NOT inherit the sub-agent's name.
    assert by_name["child.terminal"].attributes[GEN_AI_AGENT_NAME] == "research_lead"


# --- Broader leak reproducers (also failed before the fix) --------------------


def test_sibling_agents_names_do_not_leak(exporter):
    """shape_T: after a worker agent returns, the orchestrator's own span is clean.

    trip_orchestrator (agent)
      weather_agent (agent)
        weather.child   -> weather_agent
      synthesis         -> trip_orchestrator  (leaked to weather_agent before the fix)
    """

    @agent(name="weather_agent")
    def weather_agent():
        _make_child_span("weather.child")

    @agent(name="trip_orchestrator")
    def trip_orchestrator():
        weather_agent()
        _make_child_span("synthesis")  # orchestrator's own work, after worker returned

    trip_orchestrator()

    by_name = {span.name: span for span in exporter.get_finished_spans()}

    assert by_name["weather.child"].attributes[GEN_AI_AGENT_NAME] == "weather_agent"
    assert by_name["synthesis"].attributes[GEN_AI_AGENT_NAME] == "trip_orchestrator"


@pytest.mark.asyncio
async def test_async_nested_agent_name_does_not_leak(exporter):
    """The async @agent path must scope names the same way the sync path does."""

    @agent(name="inner")
    async def inner_agent():
        _make_child_span("async.inner.child")

    @agent(name="outer")
    async def outer_agent():
        await inner_agent()
        _make_child_span("async.outer.terminal")

    await outer_agent()

    by_name = {span.name: span for span in exporter.get_finished_spans()}

    assert by_name["async.inner.child"].attributes[GEN_AI_AGENT_NAME] == "inner"
    assert by_name["async.outer.terminal"].attributes[GEN_AI_AGENT_NAME] == "outer"


def test_agent_name_cleared_after_bare_agent_returns(exporter):
    """Root cause at the context level: the name is gone once the agent returns."""

    @agent(name="solo")
    def solo_agent():
        _make_child_span("solo.child")

    solo_agent()

    # Back at the top level nothing should still be attached.
    assert context_api.get_value("agent_name") is None

    by_name = {span.name: span for span in exporter.get_finished_spans()}
    assert by_name["solo.child"].attributes[GEN_AI_AGENT_NAME] == "solo"


def test_reentering_same_parent_restores_its_name_between_children(exporter):
    """Deep re-entry: the parent's name is restored between two nested calls.

    outer (agent)
      inner_a (agent)      inner_a.child   -> inner_a
      between              -> outer   (must be restored, not stuck on inner_a)
      inner_b (agent)      inner_b.child   -> inner_b
      after                -> outer   (must be restored, not stuck on inner_b)
    """

    @agent(name="inner_a")
    def inner_a():
        _make_child_span("inner_a.child")

    @agent(name="inner_b")
    def inner_b():
        _make_child_span("inner_b.child")

    @agent(name="outer")
    def outer():
        inner_a()
        _make_child_span("between")
        inner_b()
        _make_child_span("after")

    outer()

    by_name = {span.name: span for span in exporter.get_finished_spans()}

    assert by_name["inner_a.child"].attributes[GEN_AI_AGENT_NAME] == "inner_a"
    assert by_name["between"].attributes[GEN_AI_AGENT_NAME] == "outer"
    assert by_name["inner_b.child"].attributes[GEN_AI_AGENT_NAME] == "inner_b"
    assert by_name["after"].attributes[GEN_AI_AGENT_NAME] == "outer"


# --- Guardrails: passed before the fix and must keep passing ------------------


def test_flat_single_agent_labels_child(exporter):
    """Simplest happy path: a lone agent's child span is labeled correctly."""

    @agent(name="assistant")
    def assistant():
        _make_child_span("llm.call")

    assistant()

    by_name = {span.name: span for span in exporter.get_finished_spans()}
    assert by_name["llm.call"].attributes[GEN_AI_AGENT_NAME] == "assistant"


def test_agent_inside_workflow_keeps_both_names(exporter):
    """The token-detach fix must not disturb workflow_name propagation.

    A child span inside @agent-inside-@workflow must carry BOTH the enclosing
    workflow name and the agent name.
    """

    @agent(name="planner")
    def planner_agent():
        _make_child_span("child.llm")

    @workflow(name="rag")
    def rag_workflow():
        planner_agent()

    rag_workflow()

    by_name = {span.name: span for span in exporter.get_finished_spans()}
    child = by_name["child.llm"]

    assert child.attributes[SpanAttributes.TRACELOOP_WORKFLOW_NAME] == "rag"
    assert child.attributes[GEN_AI_AGENT_NAME] == "planner"


def test_sibling_task_entity_path_does_not_leak(exporter):
    """entity_path (the other affected setter) is scoped per task subtree."""

    @task(name="child_a")
    def child_a():
        _make_child_span("a.child")

    @task(name="child_b")
    def child_b():
        _make_child_span("b.child")

    @workflow(name="parent")
    def parent():
        child_a()
        child_b()

    parent()

    by_name = {span.name: span for span in exporter.get_finished_spans()}

    # A task's child span sees the task's own chained path. (The task's own span
    # does not, because its entity_path is attached after the span starts, so we
    # assert on the child spans, which is where a leak would show.)
    assert by_name["a.child"].attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] == "child_a"
    # The crux: child_a's path must not leak onto child_b's subtree.
    assert by_name["b.child"].attributes[SpanAttributes.TRACELOOP_ENTITY_PATH] == "child_b"


# --- Generator cleanup paths (the most reworked part of the fix) --------------


def test_sync_generator_agent_name_does_not_leak_after_exhaustion(exporter):
    """A @agent generator's name is detached in _handle_generator's finally.

    We probe two ways, because a following @agent would mask a leak by attaching its
    own name on top:
      1. a plain child span created AFTER exhaustion must carry no agent name;
      2. the context value itself must be cleared.
    """

    @agent(name="streamer")
    def streaming_agent():
        _make_child_span("stream.child")
        yield 1
        yield 2

    # Fully consume the generator so its finally-block cleanup runs.
    assert list(streaming_agent()) == [1, 2]

    # Context value must be gone once the generator is exhausted.
    assert context_api.get_value("agent_name") is None

    # A plain span created afterwards (no @agent to mask a leak) must be unnamed.
    _make_child_span("after.plain")

    by_name = {span.name: span for span in exporter.get_finished_spans()}
    assert by_name["stream.child"].attributes[GEN_AI_AGENT_NAME] == "streamer"
    assert GEN_AI_AGENT_NAME not in by_name["after.plain"].attributes


@pytest.mark.asyncio
async def test_async_generator_agent_name_does_not_leak_after_exhaustion(exporter):
    """Same scoping guarantee for the async-generator path (_ahandle_generator)."""

    @agent(name="astreamer")
    async def streaming_agent():
        _make_child_span("astream.child")
        yield 1
        yield 2

    collected = [item async for item in streaming_agent()]
    assert collected == [1, 2]

    assert context_api.get_value("agent_name") is None

    _make_child_span("aafter.plain")

    by_name = {span.name: span for span in exporter.get_finished_spans()}
    assert by_name["astream.child"].attributes[GEN_AI_AGENT_NAME] == "astreamer"
    assert GEN_AI_AGENT_NAME not in by_name["aafter.plain"].attributes


def test_generator_span_not_current_after_exhaustion(exporter):
    """_handle_generator re-attaches the span as current (SPAN_KEY) at entry and
    detaches it on cleanup, so the generator's span is not left current afterwards.

    NOTE: this is a smoke check, not a strict regression guard — with the redundant
    re-attach left undetached, the surrounding token detach still nets out to the
    right current span in this harness, so the assertions below pass either way. It
    documents the intended end state and catches a gross imbalance, but the real
    protection for the unbalanced attach is the capture+detach in _handle_generator.
    """

    @task(name="gen_task")
    def gen_task():
        yield 1
        yield 2

    gen = gen_task()
    gen_obj_span_id = None
    for item in gen:
        # While iterating, the generator's span is the current one.
        gen_obj_span_id = trace.get_current_span().get_span_context().span_id
    # Generator exhausted -> its span-context re-attach must have been detached, so
    # it is no longer the current span.
    current_after = trace.get_current_span().get_span_context().span_id
    assert current_after != gen_obj_span_id, (
        "generator span is still current after exhaustion — the re-attached "
        "span context was never detached"
    )

    # And a plain span created afterwards must NOT be parented to the generator.
    _make_child_span("after.plain")

    by_name = {span.name: span for span in exporter.get_finished_spans()}
    gen_span_id = by_name["gen_task.task"].get_span_context().span_id
    after = by_name["after.plain"]
    after_parent_id = after.parent.span_id if after.parent else None
    assert after_parent_id != gen_span_id, (
        "plain span after generator exhaustion is mis-parented to the generator span"
    )


@pytest.mark.asyncio
async def test_async_generator_early_exit_is_clean_under_aclosing(exporter):
    """Abandoning an async-gen @agent is only deterministic under `aclosing()`.

    `async_gen_wrap` closes the inner generator in a finally, so an explicit close
    ends the span and detaches the tokens on the caller's context. A bare `break`
    cannot get this guarantee: Python defers finalizing the wrapper to the event
    loop, so the detach runs against the finalizer's own contextvars copy and the
    caller's context keeps the name. `aclosing()` is the documented workaround.
    """
    import contextlib

    @agent(name="early_exit")
    async def streaming_agent():
        yield 1
        yield 2

    async with contextlib.aclosing(streaming_agent()) as gen:
        async for _ in gen:
            break

    assert context_api.get_value("agent_name") is None

    _make_child_span("after.aclosing")
    by_name = {span.name: span for span in exporter.get_finished_spans()}
    assert GEN_AI_AGENT_NAME not in by_name["after.aclosing"].attributes
