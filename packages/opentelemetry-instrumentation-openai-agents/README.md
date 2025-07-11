# OpenTelemetry OpenAI Agents Instrumentation

<a href="https://pypi.org/project/opentelemetry-instrumentation-openai-agents/">
    <img src="https://badge.fury.io/py/opentelemetry-instrumentation-openai-agents.svg">
</a>

This library enables tracing of agentic workflows implemented using the [OpenAI Agents framework](https://github.com/openai/openai-agents-python), allowing visibility into agent reasoning, tool usage, and decision-making steps.

## Installation

```bash
pip install opentelemetry-instrumentation-openai-agents
```

## Example usage

```python
from opentelemetry.instrumentation.openai_agents import OpenAIAgentsInstrumentor

OpenAIAgentsInstrumentor().instrument()
```

## Privacy

**By default, this instrumentation logs prompts, completions, and embeddings to span attributes**. This gives you a clear visibility into how your LLM application is working, and can make it easy to debug and evaluate the quality of the outputs.

However, you may want to disable this logging for privacy reasons, as they may contain highly sensitive data from your users. You may also simply want to reduce the size of your traces.

To disable logging, set the `TRACELOOP_TRACE_CONTENT` environment variable to `false`.

```bash
TRACELOOP_TRACE_CONTENT=false
```
