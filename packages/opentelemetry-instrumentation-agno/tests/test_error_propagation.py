"""The wrappers must not swallow failures of the call they trace.

Each wrapper used to be decorated with ``@dont_throw``, which caught every
``Exception`` raised by the traced call and returned ``None`` instead, so a
failing tool or agent run looked like an empty result. These tests fail a fake
wrapped callable and require both the exception and the error span.
"""

from types import SimpleNamespace

import pytest
from opentelemetry.instrumentation.agno import (
    _AgentRunWrapper, _AgentARunWrapper, _TeamRunWrapper, _TeamARunWrapper,
)
from opentelemetry.instrumentation.agno._tool_wrappers import (
    _FunctionCallAExecuteWrapper,
    _FunctionCallExecuteWrapper,
)
from opentelemetry import trace
from opentelemetry.trace import StatusCode
from opentelemetry.sdk.trace import TracerProvider, SpanProcessor
from opentelemetry.instrumentation.agno.config import Config


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


class _FailingProcessor(SpanProcessor):
    def __init__(self, phase):
        self.phase = phase

    def on_start(self, span, parent_context=None):
        if self.phase == "start":
            raise RuntimeError("span start failed")

    def on_end(self, span):
        if self.phase == "end":
            raise RuntimeError("span end failed")


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapper_type", [
    _FunctionCallExecuteWrapper, _FunctionCallAExecuteWrapper,
    _AgentRunWrapper, _AgentARunWrapper, _TeamRunWrapper, _TeamARunWrapper,
])
@pytest.mark.parametrize("phase", ["start", "end", "detach"])
@pytest.mark.parametrize("tool_fails", [False, True])
async def test_sdk_processor_failure_never_changes_the_wrapped_call(wrapper_type, phase, tool_fails, monkeypatch):
    if phase == "detach":
        detach = trace.context_api.detach

        def broken_detach(token):
            detach(token)
            raise RuntimeError("span detach failed")

        monkeypatch.setattr(trace.context_api, "detach", broken_detach)
    provider = TracerProvider()
    provider.add_span_processor(_FailingProcessor(phase))
    histogram = _Histogram()
    wrapper = wrapper_type(provider.get_tracer(__name__), histogram, histogram)
    instance = _tool_instance() if "FunctionCall" in wrapper_type.__name__ else SimpleNamespace(name="test")
    is_async = wrapper_type in (_FunctionCallAExecuteWrapper, _AgentARunWrapper, _TeamARunWrapper)
    calls = []
    failure = ValueError("application failed")
    result = object()

    def call():
        calls.append(1)
        if tool_fails:
            raise failure
        return result

    async def async_call():
        return call()

    if tool_fails:
        with pytest.raises(ValueError) as raised:
            value = wrapper(async_call if is_async else call, instance, (), {})
            if is_async:
                await value
        assert raised.value is failure
    else:
        value = wrapper(async_call if is_async else call, instance, (), {})
        if is_async:
            value = await value
        assert value is result
    assert calls == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_exception_logger_failure_does_not_prevent_untraced_call(monkeypatch, is_async):
    provider = TracerProvider()
    provider.add_span_processor(_FailingProcessor("start"))

    def logger(_error):
        raise RuntimeError("callback failed")

    monkeypatch.setattr(Config, "exception_logger", logger)
    histogram = _Histogram()
    wrapper_type = _FunctionCallAExecuteWrapper if is_async else _FunctionCallExecuteWrapper
    wrapper = wrapper_type(provider.get_tracer(__name__), histogram, histogram)
    calls = []

    def call():
        calls.append(1)
        return "result"

    async def async_call():
        return call()

    value = wrapper(async_call if is_async else call, _tool_instance(), (), {})
    if is_async:
        value = await value
    assert value == "result"
    assert calls == [1]
