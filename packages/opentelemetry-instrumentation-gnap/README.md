# OpenTelemetry GNAP instrumentation

This package traces GNAP task coordination operations as OpenTelemetry spans.
Install the optional GNAP dependency and instrument it with:

```python
from opentelemetry.instrumentation.gnap import GNAPInstrumentor

GNAPInstrumentor().instrument()
```

The instrumentor observes `create_task`, `claim_task`, and `complete_task`.
It records task and agent identifiers, result size where available, and the
operation outcome without copying prompts or task payloads into spans.
