# opentelemetry-instrumentation-dspy

OpenTelemetry instrumentation for [DSPy](https://github.com/stanfordnlp/dspy).

## Installation

```bash
pip install opentelemetry-instrumentation-dspy
```

## Usage

```python
from opentelemetry.instrumentation.dspy import DSPyInstrumentor

DSPyInstrumentor().instrument()
```

## Spans

| Span | Kind | Description |
|------|------|-------------|
| `chat {model}` | CLIENT | LM call via `LM.forward` / `LM.aforward` |
| `{signature}.predict` | INTERNAL | `Predict.forward` / `Predict.aforward` |

## Attributes

- `gen_ai.operation.name`
- `gen_ai.request.model`
- `gen_ai.response.model`
- `gen_ai.provider.name`
- `gen_ai.usage.input_tokens`
- `gen_ai.usage.output_tokens`
- `gen_ai.input.messages`
- `gen_ai.output.messages`
- `dspy.signature`
- `dspy.cache_hit`

## Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| `gen_ai.client.token.usage` | `token` | Input and output tokens per LM call (recorded with `gen_ai.token.type` = `input` / `output`) |
| `gen_ai.client.operation.duration` | `s` | Duration of each LM call |

Metrics are emitted unless `TRACELOOP_METRICS_ENABLED=false`.
