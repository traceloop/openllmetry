# OpenTelemetry IBM Watsonx Instrumentation

This library allows tracing IBM Watsonx prompts and completions sent with the official [IBM Watson Machine Learning library](https://ibm.github.io/watson-machine-learning-sdk/) and [IBM watsonx.ai library](https://ibm.github.io/watsonx-ai-python-sdk/).

## Installation

```bash
pip install opentelemetry-instrumentation-watsonx
```

## Example usage

```python
from opentelemetry.instrumentation.watsonx import WatsonxInstrumentor

WatsonxInstrumentor().instrument()
```

## Privacy

**By default, this instrumentation logs prompts, completions, and embeddings to span attributes**. This gives you a clear visibility into how your LLM application is working, and can make it easy to debug and evaluate the quality of the outputs.

However, you may want to disable this logging for privacy reasons, as they may contain highly sensitive data from your users. You may also simply want to reduce the size of your traces.

To disable logging, set the `TRACELOOP_TRACE_CONTENT` environment variable to `false`.

```bash
TRACELOOP_TRACE_CONTENT=false
```

## SSL Issue

In case of SSL handshake issues (or similar ones) as follows:

```
E0423 17:04:25.197068000 6150713344 ssl_transport_security.cc:1420]    Handshake failed with fatal error SSL_ERROR_SSL: error:100000f7:SSL routines:OPENSSL_internal:WRONG_VERSION_NUMBER.
```

You can instruct the exporter with an environment variable to ignore SSL errors:

```bash
OTEL_EXPORTER_OTLP_INSECURE=true
```
