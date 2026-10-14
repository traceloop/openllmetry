"""The mcp.client.session span must end however Client.__aenter__/__aexit__ finish.

Driven directly: Client.__aenter__ and __aexit__ are already wrapped, so
patching them would replace the wrappers under test.
"""

import asyncio
import contextlib
import logging

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


async def test_session_enter_failure_survives_span_end_failure(end_raises) -> None:
    """The traced call's failure is the one raised, not the span end's."""
    instrumentor = McpInstrumentor()
    with pytest.raises(ValueError, match="enter failed"):
        await instrumentor._fastmcp_client_enter_wrapper(end_raises)(
            _failing_with(ValueError("enter failed")), _Client(), (), {}
        )


async def test_session_enter_failure_survives_record_error_failure(
    tracer_provider, record_raises
) -> None:
    """The traced call's failure is the one raised, not record_error's."""
    instrumentor = McpInstrumentor()
    tracer = tracer_provider.get_tracer(__name__)
    with pytest.raises(ValueError, match="enter failed"):
        await instrumentor._fastmcp_client_enter_wrapper(tracer)(
            _failing_with(ValueError("enter failed")), _Client(), (), {}
        )


async def test_session_exit_failure_survives_span_end_failure(end_raises) -> None:
    """The teardown failure is the one raised, not the span end's."""
    instrumentor = McpInstrumentor()
    client = _Client()
    await instrumentor._fastmcp_client_enter_wrapper(end_raises)(_ok, client, (), {})
    with pytest.raises(ValueError, match="exit failed"):
        await instrumentor._fastmcp_client_exit_wrapper(end_raises)(
            _failing_with(ValueError("exit failed")), client, (), {}
        )


async def test_session_exit_failure_survives_record_error_failure(
    tracer_provider, record_raises
) -> None:
    """The teardown failure is the one raised, not record_error's."""
    instrumentor = McpInstrumentor()
    tracer = tracer_provider.get_tracer(__name__)
    client = _Client()
    await instrumentor._fastmcp_client_enter_wrapper(tracer)(_ok, client, (), {})
    with pytest.raises(ValueError, match="exit failed"):
        await instrumentor._fastmcp_client_exit_wrapper(tracer)(
            _failing_with(ValueError("exit failed")), client, (), {}
        )


async def test_client_raises_when_server_fails_to_start() -> None:
    """Through the real Client: the body must not run with the client as None.

    Issue #4518: the enter wrapper used to swallow the connection failure, so
    `async with` ran its body with `c = None`.
    """
    from fastmcp import Client
    from fastmcp.client.transports import ClientTransport

    async def _fail_to_connect():
        raise ConnectionError("server failed to start")

    class _FailingTransport(ClientTransport):
        @contextlib.asynccontextmanager
        async def connect_session(self, **session_kwargs):
            await _fail_to_connect()
            yield

    body_ran = False
    with pytest.raises(RuntimeError, match="server failed to start"):
        async with Client(_FailingTransport()):
            body_ran = True

    assert not body_ran, "the body must not run against a client that never connected"


async def _enter(instrumentor, tracer, client, wrapped=_ok):
    await instrumentor._fastmcp_client_enter_wrapper(tracer)(wrapped, client, (), {})


async def _exit(instrumentor, tracer, client, wrapped=_ok):
    await instrumentor._fastmcp_client_exit_wrapper(tracer)(wrapped, client, (), {})


async def test_nested_entries_share_one_session_span(span_exporter, tracer_provider) -> None:
    """Nested enter/exit on one client: one session span, ended by the outermost exit."""
    instrumentor = McpInstrumentor()
    tracer = tracer_provider.get_tracer(__name__)
    client = _Client()

    await _enter(instrumentor, tracer, client)
    outer = trace.get_current_span()
    await _enter(instrumentor, tracer, client)
    assert trace.get_current_span() is outer, "a nested enter must reuse the session span"
    with tracer.start_as_current_span("inside"):
        pass

    await _exit(instrumentor, tracer, client)
    assert trace.get_current_span() is outer, "the inner exit must not end the span"
    assert not _session_spans(span_exporter)

    await _exit(instrumentor, tracer, client)
    assert not trace.get_current_span().is_recording(), "session span leaked as current"
    with tracer.start_as_current_span("after"):
        pass

    sessions = _session_spans(span_exporter)
    assert len(sessions) == 1
    by_name = {s.name: s for s in span_exporter.get_finished_spans()}
    assert by_name["inside"].parent.span_id == sessions[0].context.span_id
    assert by_name["after"].parent is None


async def test_failed_nested_enter_keeps_depth_balanced(span_exporter, tracer_provider) -> None:
    """A failed nested __aenter__ gets no __aexit__, so it must not count as an entry."""
    instrumentor = McpInstrumentor()
    tracer = tracer_provider.get_tracer(__name__)
    client = _Client()

    await _enter(instrumentor, tracer, client)
    with pytest.raises(RuntimeError, match="nested connect failed"):
        await _enter(instrumentor, tracer, client, _failing_with(RuntimeError("nested connect failed")))
    assert trace.get_current_span().is_recording(), "the outer session must still be open"

    await _exit(instrumentor, tracer, client)

    assert len(_session_spans(span_exporter)) == 1
    assert not trace.get_current_span().is_recording(), "session span leaked as current"


async def test_failed_first_enter_allows_retry(span_exporter, tracer_provider) -> None:
    """A failed first enter leaves no state behind: the next enter starts a fresh span."""
    instrumentor = McpInstrumentor()
    tracer = tracer_provider.get_tracer(__name__)
    client = _Client()

    with pytest.raises(RuntimeError, match="connect failed"):
        await _enter(instrumentor, tracer, client, _failing_with(RuntimeError("connect failed")))
    await _enter(instrumentor, tracer, client)
    await _exit(instrumentor, tracer, client)

    sessions = _session_spans(span_exporter)
    assert [s.status.status_code for s in sessions] == [StatusCode.ERROR, StatusCode.UNSET]
    assert not trace.get_current_span().is_recording()


async def test_client_reuse_starts_a_new_span(span_exporter, tracer_provider) -> None:
    """After the outermost exit the client is idle again; re-entering is a new session."""
    instrumentor = McpInstrumentor()
    tracer = tracer_provider.get_tracer(__name__)
    client = _Client()

    for _ in range(2):
        await _enter(instrumentor, tracer, client)
        await _exit(instrumentor, tracer, client)

    assert len(_session_spans(span_exporter)) == 2
    assert not trace.get_current_span().is_recording()


async def test_span_end_failure_still_clears_the_entry(span_exporter, end_raises) -> None:
    """A span end that raises (logged by guard_trace) must not leave the entry behind.

    Otherwise the next `async with` would look nested and open no span at all.
    """
    instrumentor = McpInstrumentor()
    client = _Client()

    for _ in range(2):
        await _enter(instrumentor, end_raises, client)
        await _exit(instrumentor, end_raises, client)

    assert len(_session_spans(span_exporter)) == 2


async def test_concurrent_tasks_get_independent_spans(span_exporter, tracer_provider, caplog) -> None:
    """Two tasks entering one client overlap: each ends its own span in its own context."""
    instrumentor = McpInstrumentor()
    tracer = tracer_provider.get_tracer(__name__)
    client = _Client()
    leaked = []

    async def worker(mine, other):
        await _enter(instrumentor, tracer, client)
        mine.set()
        await other.wait()
        await _exit(instrumentor, tracer, client)
        leaked.append(trace.get_current_span().is_recording())

    # OTel logs, rather than raises, when a token is detached in the wrong
    # context, so the span count alone can't catch a cross-task detach.
    first, second = asyncio.Event(), asyncio.Event()
    with caplog.at_level(logging.ERROR, logger="opentelemetry.context"):
        await asyncio.gather(worker(first, second), worker(second, first))

    assert "Failed to detach context" not in caplog.text, "a task detached another task's span"
    assert len(_session_spans(span_exporter)) == 2
    assert leaked == [False, False]
    assert not trace.get_current_span().is_recording()


async def test_real_fastmcp_nested_client_does_not_leak_session_span(span_exporter) -> None:
    """The issue's repro, against the real, instrumented FastMCP client."""
    from fastmcp import Client, FastMCP

    server = FastMCP("nested-server")

    @server.tool()
    async def fast() -> str:
        return "ok"

    client = Client(server)
    async with client:
        async with client:
            await client.call_tool("fast", {})
        assert trace.get_current_span().is_recording(), "outer session must still be open"

    assert not trace.get_current_span().is_recording(), "session span leaked as current"
    assert len(_session_spans(span_exporter)) == 1


async def test_exit_from_another_task_still_ends_the_span(span_exporter, tracer_provider) -> None:
    """Enter and exit in different tasks, as pytest-asyncio runs a fixture's
    setup and teardown: the span is still ended, and nothing is left behind."""
    instrumentor = McpInstrumentor()
    tracer = tracer_provider.get_tracer(__name__)
    client = _Client()

    await asyncio.create_task(_enter(instrumentor, tracer, client))
    await asyncio.create_task(_exit(instrumentor, tracer, client))

    assert len(_session_spans(span_exporter)) == 1
    await _enter(instrumentor, tracer, client)
    assert trace.get_current_span().is_recording(), "a stale entry made the next enter look nested"
    await _exit(instrumentor, tracer, client)
    assert len(_session_spans(span_exporter)) == 2


async def test_real_fastmcp_new_client_gets_its_own_session_span(span_exporter) -> None:
    """Client.new() is a shallow copy: it must not share the original's span state."""
    from fastmcp import Client, FastMCP

    server = FastMCP("new-client-server")

    @server.tool()
    async def fast() -> str:
        return "ok"

    client = Client(server)
    async with client:
        outer = trace.get_current_span()
        fresh = client.new()
        async with fresh:
            assert trace.get_current_span() is not outer, "the new client must open its own span"
            await fresh.call_tool("fast", {})

    assert len(_session_spans(span_exporter)) == 2
    assert not trace.get_current_span().is_recording()


async def test_real_fastmcp_body_failure_in_nested_block_ends_the_span(span_exporter, caplog) -> None:
    """User code raising inside the nested block unwinds both exits cleanly."""
    from fastmcp import Client, FastMCP

    server = FastMCP("body-failure-server")

    @server.tool()
    async def fast() -> str:
        return "ok"

    client = Client(server)
    with caplog.at_level(logging.ERROR, logger="opentelemetry.context"):
        with pytest.raises(ValueError, match="user code failed"):
            async with client:
                async with client:
                    await client.call_tool("fast", {})
                    raise ValueError("user code failed")

    assert "Failed to detach context" not in caplog.text
    assert not trace.get_current_span().is_recording(), "session span leaked as current"
    assert len(_session_spans(span_exporter)) == 1


async def test_failed_inner_exit_keeps_the_session_open(span_exporter, tracer_provider) -> None:
    """A nested __aexit__ that raises is reported on the session span, but only
    the outermost exit ends it."""
    instrumentor = McpInstrumentor()
    tracer = tracer_provider.get_tracer(__name__)
    client = _Client()

    await _enter(instrumentor, tracer, client)
    await _enter(instrumentor, tracer, client)
    with pytest.raises(RuntimeError, match="inner teardown failed"):
        await _exit(instrumentor, tracer, client, _failing_with(RuntimeError("inner teardown failed")))
    assert trace.get_current_span().is_recording(), "the outer session must still be open"

    await _exit(instrumentor, tracer, client)

    sessions = _session_spans(span_exporter)
    assert len(sessions) == 1
    assert sessions[0].status.status_code is StatusCode.ERROR
    assert not trace.get_current_span().is_recording(), "session span leaked as current"


async def test_cancelled_inner_exit_keeps_the_session_open(span_exporter, tracer_provider) -> None:
    """A cancelled nested __aexit__ propagates, and the outermost exit still
    ends the span, UNSET: cancellation is not an error."""
    instrumentor = McpInstrumentor()
    tracer = tracer_provider.get_tracer(__name__)
    client = _Client()

    await _enter(instrumentor, tracer, client)
    await _enter(instrumentor, tracer, client)
    with pytest.raises(asyncio.CancelledError):
        await _exit(instrumentor, tracer, client, _failing_with(asyncio.CancelledError()))
    assert trace.get_current_span().is_recording(), "the outer session must still be open"

    await _exit(instrumentor, tracer, client)

    sessions = _session_spans(span_exporter)
    assert len(sessions) == 1
    assert sessions[0].status.status_code is StatusCode.UNSET
    assert not trace.get_current_span().is_recording(), "session span leaked as current"


async def test_broken_tracer_with_nesting_leaves_the_client_untraced(span_exporter, broken_tracer) -> None:
    """With no span opened, nested enters and exits still reach the client."""
    instrumentor = McpInstrumentor()
    client = _Client()
    calls = []

    async def _records(*args, **kwargs):
        calls.append("called")
        return "entered"

    enter = instrumentor._fastmcp_client_enter_wrapper(broken_tracer)
    exit_ = instrumentor._fastmcp_client_exit_wrapper(broken_tracer)
    assert await enter(_records, client, (), {}) == "entered"
    assert await enter(_records, client, (), {}) == "entered"
    await exit_(_records, client, (), {})
    await exit_(_records, client, (), {})

    assert len(calls) == 4, "every enter and exit must reach the client"
    assert not _session_spans(span_exporter)
    assert not trace.get_current_span().is_recording()


async def test_real_fastmcp_concurrent_tasks_share_one_client(span_exporter, caplog) -> None:
    """Two tasks overlapping on one real client: one balanced span each."""
    from fastmcp import Client, FastMCP

    server = FastMCP("concurrent-server")

    @server.tool()
    async def fast() -> str:
        return "ok"

    client = Client(server)
    open_inside = []
    leaked = []

    async def worker(mine, other):
        async with client:
            mine.set()
            await other.wait()
            # The other task's enter has run by now: it must not have touched
            # this task's session span.
            open_inside.append(trace.get_current_span().is_recording())
            await client.call_tool("fast", {})
        leaked.append(trace.get_current_span().is_recording())

    first, second = asyncio.Event(), asyncio.Event()
    with caplog.at_level(logging.ERROR, logger="opentelemetry.context"):
        await asyncio.gather(worker(first, second), worker(second, first))

    assert "Failed to detach context" not in caplog.text, "a task detached another task's span"
    assert open_inside == [True, True], "each task's own session span must stay open in its block"
    assert len(_session_spans(span_exporter)) == 2
    assert leaked == [False, False]
