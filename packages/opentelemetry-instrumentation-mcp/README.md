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

To stop tracing, finish the active client sessions and call
`McpInstrumentor().uninstrument()`. Calling `instrument()` again restores tracing
without adding duplicate spans. Re-enabling can also accept a new tracer provider.
Switching instrumentation while a client session is still open is not supported.

## Privacy

**By default, this instrumentation logs prompts, completions, and embeddings to span attributes**. This gives you a clear visibility into how your LLM application tool usage is working, and can make it easy to debug and evaluate the tool usage.

However, you may want to disable this logging for privacy reasons, as they may contain highly sensitive data from your users. You may also simply want to reduce the size of your traces.

To disable logging, set the `TRACELOOP_TRACE_CONTENT` environment variable to `false`.

```bash
TRACELOOP_TRACE_CONTENT=false
```

## Diagnostics

A failure inside this instrumentation never reaches your application: the traced call still goes out, and its result -- or its own exception -- is what the caller sees. Such a failure is logged at `WARNING` on the `opentelemetry.instrumentation.mcp` logger, with the stack at `DEBUG`.

Some of these failures sit on a per-message path (the stream writer), so a persistent one logs a line per MCP message. To quiet it without touching the rest of your logging:

```python
import logging

logging.getLogger("opentelemetry.instrumentation.mcp").setLevel(logging.ERROR)
```
