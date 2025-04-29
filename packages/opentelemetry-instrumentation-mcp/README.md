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
