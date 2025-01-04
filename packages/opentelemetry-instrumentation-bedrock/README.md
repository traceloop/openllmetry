# OpenTelemetry Bedrock Instrumentation

<a href="https://pypi.org/project/opentelemetry-instrumentation-bedrock/">
    <img src="https://badge.fury.io/py/opentelemetry-instrumentation-bedrock.svg">
</a>

This library allows tracing any of AWS Bedrock's models prompts and completions sent with [Boto3](https://github.com/boto/boto3) to Bedrock.

## Features

- Traces all calls to Bedrock's model endpoints
- Supports both legacy attribute-based and new event-based semantic conventions
- Captures prompts, completions, and token usage metrics
- Supports streaming responses with chunk-by-chunk instrumentation
- Handles multiple model providers (Anthropic, Cohere, AI21, Meta, Amazon)

## Installation

```bash
pip install opentelemetry-instrumentation-bedrock
```

## Example usage

```python
from opentelemetry.instrumentation.bedrock import BedrockInstrumentor

# Use legacy attribute-based semantic conventions (default)
BedrockInstrumentor().instrument()

# Or use new event-based semantic conventions
from opentelemetry.instrumentation.bedrock.config import Config
Config.use_legacy_attributes = False
BedrockInstrumentor().instrument()
```

## Configuration

The instrumentation can be configured using the following options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `use_legacy_attributes` | bool | `True` | Controls whether to use legacy attribute-based semantic conventions or new event-based semantic conventions. When `True`, prompts and completions are stored as span attributes. When `False`, they are stored as span events following the new OpenTelemetry semantic conventions. |
| `enrich_token_usage` | bool | `False` | When `True`, calculates token usage even when not provided by the API response. |
| `exception_logger` | Callable | `None` | Optional callback for logging exceptions that occur during instrumentation. |

## Privacy

**By default, this instrumentation logs prompts, completions, and embeddings**. This gives you clear visibility into how your LLM application is working, and can make it easy to debug and evaluate the quality of the outputs.

The data can be stored either as span attributes (legacy mode) or span events (new mode), controlled by the `use_legacy_attributes` configuration option.

However, you may want to disable this logging for privacy reasons, as they may contain highly sensitive data from your users. You may also simply want to reduce the size of your traces.

To disable logging, set the `TRACELOOP_TRACE_CONTENT` environment variable to `false`.

```bash
TRACELOOP_TRACE_CONTENT=false
```

## Semantic Conventions

This instrumentation supports two modes of operation:

### Legacy Attribute-based Mode (Default)

In this mode, prompts and completions are stored as span attributes following the pattern:
- `gen_ai.prompt.{index}.content` - The prompt text
- `gen_ai.prompt.{index}.role` - The role (e.g., "user", "system")
- `gen_ai.completion.{index}.content` - The completion text
- `gen_ai.completion.{index}.role` - The role (e.g., "assistant")

### Event-based Mode

In this mode, prompts and completions are stored as span events following the new OpenTelemetry semantic conventions:
- Prompt events with attributes:
  - `llm.prompt.index` - The index of the prompt
  - `llm.prompt.type` - The type of prompt (e.g., "chat", "completion")
  - `llm.prompt.content` - The prompt text
  - `llm.prompt.role` - The role (e.g., "user", "system")
- Completion events with attributes:
  - `llm.completion.index` - The index of the completion
  - `llm.completion.content` - The completion text
  - `llm.completion.role` - The role (e.g., "assistant")
  - `llm.completion.stop_reason` - The reason the completion stopped

For streaming responses in event-based mode, each chunk generates a `llm.content.completion.chunk` event with the chunk's content.

Token usage metrics and other metadata are recorded in both modes using standard attributes.
