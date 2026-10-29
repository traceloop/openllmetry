"""Ordinary tool calls must carry the caller's trace context to the server."""

from mcp.types import CallToolRequest, CallToolRequestParams, ClientRequest
from opentelemetry import propagate, trace
from opentelemetry.instrumentation.mcp import McpInstrumentor


def _wire_meta(request):
    """The JSON-RPC `_meta` field as it is serialized on the wire."""
    return request.model_dump(by_alias=True, exclude_none=True)["params"]["_meta"]


async def test_call_tool_without_meta_gets_traceparent(tracer_provider) -> None:
    """ClientSession.call_tool() leaves params.meta None; inject anyway (#4374).

    #3179 removed the fallback that created an empty Meta along with the
    overwrite it was meant to prevent. The default case then sent no `_meta` at
    all, so the MCP server started a new trace instead of continuing the
    caller's.
    """
    instrumentor = McpInstrumentor()
    wrapper = instrumentor.patch_mcp_client(tracer_provider.get_tracer(__name__))

    sent = []

    async def send(request):
        sent.append(request)

    request = ClientRequest(
        CallToolRequest(params=CallToolRequestParams(name="echo", arguments={}))
    )
    assert request.root.params.meta is None

    tracer = tracer_provider.get_tracer(__name__)
    with tracer.start_as_current_span("caller") as caller_span:
        await wrapper(send, object(), (request,), {})

    assert len(sent) == 1, "send_request must still be called"
    meta = _wire_meta(sent[0])
    traceparent = meta["traceparent"]
    caller_trace_id = format(caller_span.get_span_context().trace_id, "032x")
    assert traceparent.startswith(f"00-{caller_trace_id}-")

    # The server extracts the same trace from the `_meta` it received.
    parent = trace.get_current_span(propagate.extract(meta))
    assert (
        parent.get_span_context().trace_id
        == caller_span.get_span_context().trace_id
    )
