"""The mcp.client.session span must end however Client.__aenter__/__aexit__ finish.

Driven directly: Client.__aenter__ and __aexit__ are already wrapped, so
patching them would replace the wrappers under test.
"""

import asyncio

import pytest
from opentelemetry import trace
from opentelemetry.instrumentation.mcp import McpInstrumentor
from opentelemetry.instrumentation.mcp.utils import Config
from opentelemetry.trace import StatusCode

# Cancellation is not an error (open-telemetry/opentelemetry-python#4484), but
# the span must still end rather than never being exported.
CASES = [
    (None, StatusCode.UNSET),
    (RuntimeError("session failed"), StatusCode.ERROR),
    (asyncio.CancelledError(), StatusCode.UNSET),
]


class _Client:
    """Stands in for a FastMCP client carrying the wrappers' own state."""


async def _ok(*args, **kwargs):
    """Enter or exit successfully, the way a healthy client would."""
    return None


def _failing_with(failure):
    """A wrapped __aenter__/__aexit__ that raises `failure`."""

    async def _fail(*args, **kwargs):
        raise failure

    return _fail


async def _call(wrapper, wrapped, client, failure):
    """Run a wrapper; the traced call's own failure reaches the caller.

    Instrumentation used to swallow Exception, because @dont_throw decorated the
    whole wrapper and only cancellation got past it. That turned a dead
    transport into `async with` running with `c = None` (issue #4518).
    """
    if failure is None:
        await wrapper(wrapped, client, (), {})
    else:
        with pytest.raises(type(failure)):
            await wrapper(wrapped, client, (), {})


def _session_spans(span_exporter):
    return [
        s for s in span_exporter.get_finished_spans() if s.name == "mcp.client.session"
    ]


@pytest.mark.parametrize(("failure", "expected_status"), CASES)
async def test_session_span_ends_on_exit(
    span_exporter, tracer_provider, failure, expected_status
) -> None:
    """However __aexit__ finishes, the span is ended exactly once."""
    instrumentor = McpInstrumentor()
    tracer = tracer_provider.get_tracer(__name__)
    client = _Client()

    await instrumentor._fastmcp_client_enter_wrapper(tracer)(_ok, client, (), {})
    await _call(
        instrumentor._fastmcp_client_exit_wrapper(tracer),
        _failing_with(failure) if failure else _ok,
        client,
        failure,
    )

    spans = _session_spans(span_exporter)
    assert len(spans) == 1, "the session span must be ended exactly once"
    assert spans[0].status.status_code is expected_status
    assert not trace.get_current_span().is_recording(), "span left current"


@pytest.mark.parametrize(("failure", "expected_status"), CASES[1:])
async def test_session_span_ends_when_enter_fails(
    span_exporter, tracer_provider, failure, expected_status
) -> None:
    """A failed __aenter__ gets no __aexit__ from `async with`.

    So the enter wrapper has to end the span itself, and detach it: otherwise
    later spans in the task are parented to a span that is never exported.
    """
    instrumentor = McpInstrumentor()
    tracer = tracer_provider.get_tracer(__name__)

    await _call(
        instrumentor._fastmcp_client_enter_wrapper(tracer),
        _failing_with(failure),
        _Client(),
        failure,
    )

    spans = _session_spans(span_exporter)
    assert len(spans) == 1, "the session span must be ended exactly once"
    assert spans[0].status.status_code is expected_status
    assert not trace.get_current_span().is_recording(), "span left current"


async def test_instrumentation_failure_never_reaches_the_caller(broken_tracer) -> None:
    """Spans are the instrumentation's own code: their failures stay logged.

    Only the span work is guarded, not the wrapper as a whole, so a tracer
    outage neither blocks entering the client nor masks a failure of the traced
    call itself -- and the operator's exception logger still hears about it.
    """
    reported = []
    instrumentor = McpInstrumentor(exception_logger=reported.append)
    try:
        client = _Client()

        async def _returns(*args, **kwargs):
            return "entered"

        result = await instrumentor._fastmcp_client_enter_wrapper(broken_tracer)(
            _returns, client, (), {}
        )
        assert result == "entered", "a broken tracer must not block the traced call"
        assert reported, "the instrumentation failure must reach Config.exception_logger"

        with pytest.raises(RuntimeError, match="call failed"):
            await instrumentor._fastmcp_client_enter_wrapper(broken_tracer)(
                _failing_with(RuntimeError("call failed")), client, (), {}
            )
    finally:
        # McpInstrumentor.__init__ writes Config.exception_logger globally, and
        # conftest's session-scoped instrumentor set it to None.
        Config.exception_logger = None
