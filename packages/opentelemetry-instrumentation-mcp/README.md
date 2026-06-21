OpenTelemetry MCP Instrumentation

<a href="https://pypi.org/project/opentelemetry-instrumentation-mcp/">
    <img src="https://badge.fury.io/py/opentelemetry-instrumentation-mcp.svg">
</a>

This library allows tracing of agentic workflows implemented with MCP framework [mcp python sdk](https://github.com/modelcontextprotocol/python-sdk).

## Installation

```bash
pip install opentelemetry-instrumentation-mcp
```

## Example usage

```python
from opentelemetry.instrumentation.mcp import McpInstrumentor

McpInstrumentor().instrument()
```

## Privacy

**By default, this instrumentation logs prompts, completions, and embeddings to span attributes**. This gives you a clear visibility into how your LLM application tool usage is working, and can make it easy to debug and evaluate the tool usage.

However, you may want to disable this logging for privacy reasons, as they may contain highly sensitive data from your users. You may also simply want to reduce the size of your traces.

To disable logging, set the `TRACELOOP_TRACE_CONTENT` environment variable to `false`.

```bash
TRACELOOP_TRACE_CONTENT=false
```

### What is and isn't gated

The `TRACELOOP_TRACE_CONTENT` flag only gates **content**. Operational **metadata** is
always traced so that spans remain useful for monitoring even when content capture is off.

| Always traced (metadata) | Gated behind `TRACELOOP_TRACE_CONTENT=true` (content) |
| --- | --- |
| Method / tool name (span name and `traceloop.entity.name`) | Tool arguments / prompt (`traceloop.entity.input`) |
| Span kind (`traceloop.span.kind`) | Tool / method result (`traceloop.entity.output`) |
| Request id (`mcp.request.id`) | Serialized response payload (`mcp.response.value`) |
| Duration (span start/end) | Error message text (span status description, recorded exception) |
| Status code (`OK` / `ERROR`) | |
| Error class (`error.type`) | |

When content is disabled, a failed call still produces an `ERROR` span carrying its
`error.type`, but the status description is replaced with a generic, content-free message so
that the tool's error text and arguments cannot leak through it.
