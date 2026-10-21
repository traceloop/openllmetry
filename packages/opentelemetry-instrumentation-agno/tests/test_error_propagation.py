"""The wrappers must not swallow failures of the call they trace.

Each wrapper used to be decorated with ``@dont_throw``, which caught every
``Exception`` raised by the traced call and returned ``None`` instead, so a
failing tool or agent run looked like an empty result. These tests fail a fake
wrapped callable and require both the exception and the error span.
"""

from types import SimpleNamespace

import pytest
from opentelemetry.instrumentation.agno import _AgentRunWrapper, _TeamARunWrapper
from opentelemetry.instrumentation.agno._tool_wrappers import (
    _FunctionCallAExecuteWrapper,
    _FunctionCallExecuteWrapper,
)
from opentelemetry.trace import StatusCode


class _Histogram:
    """Minimal histogram stand-in; these tests only assert on spans."""

    def __init__(self):
        self.records = []

    def record(self, value, attributes=None):
        self.records.append((value, attributes))


def _explode(*args, **kwargs):
    raise RuntimeError("tool exploded")


async def _aexplode(*args, **kwargs):
    raise RuntimeError("tool exploded")


def _tool_instance():
    return SimpleNamespace(
        function=SimpleNamespace(name="explode", description="Explodes"),
        arguments={},
    )


def _assert_error_span(span, name):
    assert span.name == name
    assert span.status.status_code is StatusCode.ERROR
    assert "tool exploded" in span.status.description
    exception_events = [event for event in span.events if event.name == "exception"]
    assert exception_events, "the failure must be recorded on the span"
    assert exception_events[0].attributes["exception.type"] == "RuntimeError"


def test_tool_execute_propagates_failure(span_exporter, tracer_provider):
    histogram = _Histogram()
    wrapper = _FunctionCallExecuteWrapper(
        tracer_provider.get_tracer(__name__), histogram, histogram
    )

    with pytest.raises(RuntimeError, match="tool exploded"):
        wrapper(_explode, _tool_instance(), (), {})

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    _assert_error_span(spans[0], "explode.tool")


@pytest.mark.asyncio
async def test_tool_aexecute_propagates_failure(span_exporter, tracer_provider):
    histogram = _Histogram()
    wrapper = _FunctionCallAExecuteWrapper(
        tracer_provider.get_tracer(__name__), histogram, histogram
    )

    with pytest.raises(RuntimeError, match="tool exploded"):
        await wrapper(_aexplode, _tool_instance(), (), {})

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    _assert_error_span(spans[0], "explode.tool")


def test_agent_run_propagates_failure(span_exporter, tracer_provider):
    histogram = _Histogram()
    wrapper = _AgentRunWrapper(
        tracer_provider.get_tracer(__name__), histogram, histogram
    )
    agent = SimpleNamespace(name="FailingAgent", model=SimpleNamespace(id="gpt-4o-mini"))

    with pytest.raises(RuntimeError, match="tool exploded"):
        wrapper(_explode, agent, ("hi",), {})

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    _assert_error_span(spans[0], "FailingAgent.agent")


@pytest.mark.asyncio
async def test_team_arun_propagates_failure(span_exporter, tracer_provider):
    histogram = _Histogram()
    wrapper = _TeamARunWrapper(
        tracer_provider.get_tracer(__name__), histogram, histogram
    )
    team = SimpleNamespace(name="FailingTeam")

    with pytest.raises(RuntimeError, match="tool exploded"):
        await wrapper(_aexplode, team, ("hi",), {})

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    _assert_error_span(spans[0], "FailingTeam.team")


def test_tool_execute_runs_when_tracer_fails():
    """Instrumentation failures stay in the instrumentation: the tool runs."""

    class _BrokenTracer:
        def start_as_current_span(self, *args, **kwargs):
            raise RuntimeError("tracer exploded")

    histogram = _Histogram()
    wrapper = _FunctionCallExecuteWrapper(_BrokenTracer(), histogram, histogram)

    assert wrapper(lambda *args, **kwargs: "result", _tool_instance(), (), {}) == "result"
