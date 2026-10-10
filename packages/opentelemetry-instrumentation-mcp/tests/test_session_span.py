"""The mcp.client.session span must end however Client.__aenter__/__aexit__ finish.

Driven directly: Client.__aenter__ and __aexit__ are already wrapped, so
patching them would replace the wrappers under test.
"""

import asyncio

import pytest
from opentelemetry import trace
from opentelemetry.instrumentation.mcp import McpInstrumentor
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
    """Run a wrapper and verify lifecycle failures remain visible to FastMCP."""
    if failure is not None:
        with pytest.raises(type(failure)):
            await wrapper(wrapped, client, (), {})
    else:
        await wrapper(wrapped, client, (), {})


def _session_spans(span_exporter):
    return [
        s for s in span_exporter.get_finished_spans() if s.name == "mcp.client.session"
    ]


async def test_reentrant_fastmcp_client_uses_one_session_span(span_exporter) -> None:
    """The issue's real FastMCP nesting pattern must not leak context."""
    from fastmcp import Client, FastMCP

    client = Client(FastMCP("reentrant-test-server"))
    async with client:
        session_span = trace.get_current_span()
        async with client:
            assert trace.get_current_span() is session_span
        assert trace.get_current_span() is session_span

    assert len(_session_spans(span_exporter)) == 1
    assert not trace.get_current_span().is_recording()


async def test_reentrant_client_uses_one_session_span(
    span_exporter, tracer_provider
) -> None:
    """Nested entries share one span and restore the caller's context."""
    instrumentor = McpInstrumentor()
    tracer = tracer_provider.get_tracer(__name__)
    client = _Client()
    enter = instrumentor._fastmcp_client_enter_wrapper(tracer)
    exit_ = instrumentor._fastmcp_client_exit_wrapper(tracer)

    with tracer.start_as_current_span("parent") as parent:
        await enter(_ok, client, (), {})
        session_span = trace.get_current_span()
        assert session_span is not parent

        await enter(_ok, client, (), {})
        assert trace.get_current_span() is session_span

        await exit_(_ok, client, (), {})
        assert trace.get_current_span() is session_span

        await exit_(_ok, client, (), {})
        assert trace.get_current_span() is parent

    spans = _session_spans(span_exporter)
    assert len(spans) == 1
    assert spans[0].parent.span_id == parent.get_span_context().span_id


async def test_failed_nested_enter_preserves_outer_session(
    span_exporter, tracer_provider
) -> None:
    """A failed nested async-with enter never consumes the outer entry."""
    instrumentor = McpInstrumentor()
    tracer = tracer_provider.get_tracer(__name__)
    client = _Client()
    enter = instrumentor._fastmcp_client_enter_wrapper(tracer)
    exit_ = instrumentor._fastmcp_client_exit_wrapper(tracer)
    nested_exit_called = False

    class _NestedContext:
        async def __aenter__(self):
            return await enter(
                _failing_with(RuntimeError("nested enter failed")),
                client,
                (),
                {},
            )

        async def __aexit__(self, *args):
            nonlocal nested_exit_called
            nested_exit_called = True
            return await exit_(_ok, client, args, {})

    await enter(_ok, client, (), {})
    session_span = trace.get_current_span()
    with pytest.raises(RuntimeError, match="nested enter failed"):
        async with _NestedContext():
            pytest.fail("a failed __aenter__ must not run the context body")

    assert not nested_exit_called
    assert trace.get_current_span() is session_span
    assert not _session_spans(span_exporter)

    await exit_(_ok, client, (), {})
    spans = _session_spans(span_exporter)
    assert len(spans) == 1
    assert spans[0].status.status_code is StatusCode.ERROR
    assert not trace.get_current_span().is_recording()


async def test_failed_nested_exit_preserves_outer_session(
    span_exporter, tracer_provider
) -> None:
    """A failed nested exit releases only its own context entry."""
    instrumentor = McpInstrumentor()
    tracer = tracer_provider.get_tracer(__name__)
    client = _Client()
    enter = instrumentor._fastmcp_client_enter_wrapper(tracer)
    exit_ = instrumentor._fastmcp_client_exit_wrapper(tracer)

    await enter(_ok, client, (), {})
    session_span = trace.get_current_span()
    await enter(_ok, client, (), {})

    with pytest.raises(RuntimeError, match="nested exit failed"):
        await exit_(
            _failing_with(RuntimeError("nested exit failed")), client, (), {}
        )

    assert trace.get_current_span() is session_span
    assert not _session_spans(span_exporter)

    await exit_(_ok, client, (), {})
    spans = _session_spans(span_exporter)
    assert len(spans) == 1
    assert spans[0].status.status_code is StatusCode.ERROR
    assert not trace.get_current_span().is_recording()


async def test_concurrent_entries_share_session_span(
    span_exporter, tracer_provider
) -> None:
    """Concurrent users get task-local context for one shared session."""
    instrumentor = McpInstrumentor()
    tracer = tracer_provider.get_tracer(__name__)
    client = _Client()
    enter = instrumentor._fastmcp_client_enter_wrapper(tracer)
    exit_ = instrumentor._fastmcp_client_exit_wrapper(tracer)
    both_entered = asyncio.Event()
    first_entered = asyncio.Event()
    observed_spans = []

    async def use_client(first):
        await enter(_ok, client, (), {})
        observed_spans.append(trace.get_current_span())
        if first:
            first_entered.set()
            await both_entered.wait()
        else:
            await first_entered.wait()
            both_entered.set()
        await exit_(_ok, client, (), {})
        assert not trace.get_current_span().is_recording()

    await asyncio.gather(use_client(True), use_client(False))

    assert observed_spans[0] is observed_spans[1]
    assert len(_session_spans(span_exporter)) == 1


async def test_failed_concurrent_enter_does_not_end_live_session(
    span_exporter, tracer_provider
) -> None:
    """One failed concurrent entry cannot end another task's live session."""
    instrumentor = McpInstrumentor()
    tracer = tracer_provider.get_tracer(__name__)
    client = _Client()
    enter = instrumentor._fastmcp_client_enter_wrapper(tracer)
    exit_ = instrumentor._fastmcp_client_exit_wrapper(tracer)
    first_entered = asyncio.Event()
    failed_entry_finished = asyncio.Event()

    async def use_client():
        await enter(_ok, client, (), {})
        session_span = trace.get_current_span()
        first_entered.set()
        await failed_entry_finished.wait()
        assert trace.get_current_span() is session_span
        assert not _session_spans(span_exporter)
        await exit_(_ok, client, (), {})

    async def fail_to_enter():
        await first_entered.wait()
        with pytest.raises(RuntimeError, match="concurrent enter failed"):
            await enter(
                _failing_with(RuntimeError("concurrent enter failed")),
                client,
                (),
                {},
            )
        failed_entry_finished.set()

    await asyncio.gather(use_client(), fail_to_enter())

    spans = _session_spans(span_exporter)
    assert len(spans) == 1
    assert spans[0].status.status_code is StatusCode.ERROR


async def test_instrumentation_setup_failure_does_not_break_client() -> None:
    """A partial tracing setup is rolled back without blocking FastMCP."""

    class _FailingSpan:
        def __init__(self):
            self.ended = False

        def set_attribute(self, *args):
            raise RuntimeError("attribute setup failed")

        def end(self):
            self.ended = True

    class _FailingTracer:
        def __init__(self):
            self.span = _FailingSpan()

        def start_span(self, *args, **kwargs):
            return self.span

    tracer = _FailingTracer()
    client = _Client()

    result = await McpInstrumentor()._fastmcp_client_enter_wrapper(tracer)(
        _ok, client, (), {}
    )

    assert result is None
    assert tracer.span.ended
    assert not hasattr(client, "_tracing_session")


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
