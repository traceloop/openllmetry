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
    """Run a wrapper; @dont_throw swallows Exception, cancellation propagates."""
    if isinstance(failure, asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError):
            await wrapper(wrapped, client, (), {})
    else:
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


async def test_nested_session_reuses_one_span_and_restores_parent(
    span_exporter, tracer_provider
) -> None:
    """A reentrant FastMCP client shares its session until the outer exit."""
    instrumentor = McpInstrumentor()
    tracer = tracer_provider.get_tracer(__name__)
    client = _Client()
    enter = instrumentor._fastmcp_client_enter_wrapper(tracer)
    exit_ = instrumentor._fastmcp_client_exit_wrapper(tracer)

    with tracer.start_as_current_span("parent") as parent:
        await enter(_ok, client, (), {})
        outer = trace.get_current_span()
        await enter(_ok, client, (), {})
        assert trace.get_current_span() is outer
        await exit_(_ok, client, (), {})
        assert trace.get_current_span() is outer
        assert _session_spans(span_exporter) == []
        await exit_(_ok, client, (), {})
        assert trace.get_current_span() is parent
        spans = _session_spans(span_exporter)
        assert len(spans) == 1
        assert spans[0].parent.span_id == parent.get_span_context().span_id
    assert not trace.get_current_span().is_recording()


@pytest.mark.parametrize("failure", [RuntimeError("nested enter failed"), asyncio.CancelledError()])
async def test_failed_nested_enter_leaves_outer_session_active(
    span_exporter, tracer_provider, failure
) -> None:
    instrumentor = McpInstrumentor()
    tracer = tracer_provider.get_tracer(__name__)
    client = _Client()
    enter = instrumentor._fastmcp_client_enter_wrapper(tracer)
    exit_ = instrumentor._fastmcp_client_exit_wrapper(tracer)
    await enter(_ok, client, (), {})
    outer = trace.get_current_span()
    await _call(enter, _failing_with(failure), client, failure)
    assert trace.get_current_span() is outer
    await exit_(_ok, client, (), {})
    assert len(_session_spans(span_exporter)) == 1
    assert not trace.get_current_span().is_recording()


@pytest.mark.parametrize("failure", [RuntimeError("nested exit failed"), asyncio.CancelledError()])
async def test_failed_nested_exit_does_not_end_outer_session(
    span_exporter, tracer_provider, failure
) -> None:
    instrumentor = McpInstrumentor()
    tracer = tracer_provider.get_tracer(__name__)
    client = _Client()
    enter = instrumentor._fastmcp_client_enter_wrapper(tracer)
    exit_ = instrumentor._fastmcp_client_exit_wrapper(tracer)
    await enter(_ok, client, (), {})
    await enter(_ok, client, (), {})
    outer = trace.get_current_span()
    await _call(exit_, _failing_with(failure), client, failure)
    assert trace.get_current_span() is outer
    assert _session_spans(span_exporter) == []
    await exit_(_ok, client, (), {})
    assert len(_session_spans(span_exporter)) == 1
    assert not trace.get_current_span().is_recording()


async def test_overlapping_entries_share_session_without_leaking_context(
    span_exporter, tracer_provider
) -> None:
    """Two tasks entering one FastMCP client concurrently share one session."""
    instrumentor = McpInstrumentor()
    tracer = tracer_provider.get_tracer(__name__)
    client = _Client()
    enter = instrumentor._fastmcp_client_enter_wrapper(tracer)
    exit_ = instrumentor._fastmcp_client_exit_wrapper(tracer)
    first_entered = asyncio.Event()
    second_entered = asyncio.Event()
    allow_first = asyncio.Event()
    allow_second = asyncio.Event()
    observed = []

    async def first():
        async def wait_first():
            first_entered.set()
            await allow_first.wait()
        await enter(wait_first, client, (), {})
        observed.append(("first", trace.get_current_span().name))
        await second_entered.wait()
        await exit_(_ok, client, (), {})
        observed.append(("first after", trace.get_current_span().is_recording()))

    async def second():
        async def wait_second():
            second_entered.set()
            await allow_second.wait()
        await enter(wait_second, client, (), {})
        observed.append(("second", trace.get_current_span().name))
        await exit_(_ok, client, (), {})
        observed.append(("second after", trace.get_current_span().is_recording()))

    one = asyncio.create_task(first())
    await first_entered.wait()
    two = asyncio.create_task(second())
    await second_entered.wait()
    allow_second.set()
    allow_first.set()
    await asyncio.gather(one, two)
    assert len([item for item in observed if item[1] == "mcp.client.session"]) == 2
    assert all(not active for label, active in observed if label.endswith("after"))
    # Each concurrent task owns its own OTel context token; neither may detach
    # the other's span, even though FastMCP shares one underlying connection.
    assert len(_session_spans(span_exporter)) == 2
    assert not trace.get_current_span().is_recording()
