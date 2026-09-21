"""Regression tests for #4463.

BaseSession.send_request returns the RPC result the MCP SDK dereferences, so the
tracing wrapper must never turn a tracing failure into a None return (which the
SDK then dereferences as ``result.isError`` -> AttributeError) and must not
raise on valid non-text MCP content blocks.
"""
import types

import pytest

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.instrumentation.mcp.instrumentation import McpInstrumentor


def _make_traced_method():
    tracer = TracerProvider().get_tracer(__name__)
    return McpInstrumentor().patch_mcp_client(tracer)


class _Root:
    def __init__(self, method, params):
        self.method = method
        self.params = params


class _Req:
    def __init__(self, method, params):
        self.root = _Root(method, params)


class _CallToolResult:
    def __init__(self, is_error=False, content=None):
        self.isError = is_error
        self.content = content if content is not None else []


class _TextBlock:
    def __init__(self, text):
        self.text = text


class _ImageBlock:
    # A valid non-text MCP content block: no `.text` attribute.
    def __init__(self, data):
        self.data = data


@pytest.mark.asyncio
async def test_result_passthrough_on_success():
    traced = _make_traced_method()
    sentinel = _CallToolResult(is_error=False, content=[_TextBlock("hi")])

    async def wrapped(*args, **kwargs):
        return sentinel

    # params without a `.meta` attribute still works
    req = _Req("tools/call", types.SimpleNamespace(name="my_tool"))
    result = await traced(wrapped, None, (req,), {})
    assert result is sentinel  # not None


@pytest.mark.asyncio
async def test_tracing_prep_failure_does_not_break_the_call():
    # args[0].root has no usable structure -> the trace-context prep block would
    # raise, but it must be swallowed WITHOUT replacing the real result.
    traced = _make_traced_method()
    sentinel = _CallToolResult(is_error=False)

    async def wrapped(*args, **kwargs):
        return sentinel

    broken = types.SimpleNamespace()  # no `.root`
    result = await traced(wrapped, None, (broken,), {})
    assert result is sentinel  # tracing failure must not yield None


@pytest.mark.asyncio
async def test_real_exception_propagates_not_swallowed():
    traced = _make_traced_method()

    async def wrapped(*args, **kwargs):
        raise ValueError("boom from the real RPC")

    req = _Req("tools/call", types.SimpleNamespace(name="my_tool"))
    with pytest.raises(ValueError, match="boom from the real RPC"):
        await traced(wrapped, None, (req,), {})


@pytest.mark.asyncio
async def test_iserror_with_non_text_content_does_not_raise():
    # An error result whose first content block is an image must not raise
    # AttributeError on `.text` while decorating the span.
    traced = _make_traced_method()
    result_obj = _CallToolResult(is_error=True, content=[_ImageBlock(b"...")])

    async def wrapped(*args, **kwargs):
        return result_obj

    req = _Req("tools/call", types.SimpleNamespace(name="my_tool"))
    result = await traced(wrapped, None, (req,), {})
    assert result is result_obj


@pytest.mark.asyncio
async def test_post_call_span_decoration_failure_does_not_break_result(monkeypatch):
    # A tracing failure AFTER the RPC completes (e.g. span.set_attribute raising)
    # must not replace the valid result. Force _decorate_result_span's span ops
    # to raise and assert the real result still comes back.
    sentinel = _CallToolResult(is_error=False, content=[_TextBlock("ok")])

    async def wrapped(*args, **kwargs):
        return sentinel


    class _ExplodingSpan:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def set_attribute(self, *a, **k):
            raise RuntimeError("span backend down")

        def set_status(self, *a, **k):
            raise RuntimeError("span backend down")

        def record_exception(self, *a, **k):
            raise RuntimeError("span backend down")

    class _Tracer:
        def start_as_current_span(self, *a, **k):
            return _ExplodingSpan()

    # Patch the tracer this traced_method closed over is hard; instead patch the
    # decoration helpers' target span via a fresh traced_method using a tracer
    # that yields an exploding span.
    tm = McpInstrumentor().patch_mcp_client(_Tracer())
    req = _Req("tools/call", types.SimpleNamespace(name="my_tool"))
    result = await tm(wrapped, None, (req,), {})
    assert result is sentinel  # decoration failure must not touch the result
