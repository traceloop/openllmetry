# OpenTelemetry Langchain Instrumentation

<a href="https://pypi.org/project/opentelemetry-instrumentation-langchain/">
    <img src="https://badge.fury.io/py/opentelemetry-instrumentation-langchain.svg">
</a>

This library allows tracing complete LLM applications built with [Langchain](https://github.com/langchain-ai/langchain).

## Installation

```bash
pip install opentelemetry-instrumentation-langchain traceloop-sdk
```

## Example usage

```python
from opentelemetry.instrumentation.langchain import LangchainInstrumentor

LangchainInstrumentor().instrument()
```

## Trace Continuity in Agent Loops

When running a bare LangChain agent loop (e.g., executing `model.invoke` and tools in a loop without an active parent span), each invocation starts a new root span with its own trace ID.

To maintain trace continuity across all steps in an agent loop, wrap the execution in a parent span or use the `@workflow` decorator:

```python
from traceloop.sdk.decorators import workflow

@workflow(name="langchain_agent_loop")
def run_agent_loop(query: str):
    # Your LangChain model invocation & tool execution loop

    ...
```
## Privacy

**By default, this instrumentation logs prompts, completions, and embeddings to span attributes**. This gives you a clear visibility into how your LLM application is working, and can make it easy to debug and evaluate the quality of the outputs.

However, you may want to disable this logging for privacy reasons, as they may contain highly sensitive data from your users. You may also simply want to reduce the size of your traces.

To disable logging, set the `TRACELOOP_TRACE_CONTENT` environment variable to `false`.

```bash
TRACELOOP_TRACE_CONTENT=false
```
