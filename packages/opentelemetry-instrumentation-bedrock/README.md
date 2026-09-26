# OpenTelemetry Bedrock Instrumentation

<a href="https://pypi.org/project/opentelemetry-instrumentation-bedrock/">
    <img src="https://badge.fury.io/py/opentelemetry-instrumentation-bedrock.svg">
</a>

This library allows tracing any of AWS Bedrock's models prompts and completions sent with [Boto3](https://github.com/boto/boto3) to Bedrock.

## Installation

```bash
pip install opentelemetry-instrumentation-bedrock
```

## Example usage

```python
from opentelemetry.instrumentation.bedrock import BedrockInstrumentor

BedrockInstrumentor().instrument()
```

## Request and response hooks

Pass optional hooks to `instrument()` to add application attributes to a
recording Bedrock span. The signatures follow OpenTelemetry's Botocore hooks:
`request_hook(span, service_name, operation_name, request_params)` and
`response_hook(span, service_name, operation_name, response)`.

```python
from opentelemetry.instrumentation.bedrock import BedrockInstrumentor

def request_hook(span, service_name, operation_name, request_params):
    span.set_attribute("app.model", request_params["modelId"])

def response_hook(span, service_name, operation_name, response):
    request_id = response.get("ResponseMetadata", {}).get("RequestId")
    if request_id:
        span.set_attribute("app.bedrock_request_id", request_id)

BedrockInstrumentor().instrument(
    request_hook=request_hook,
    response_hook=response_hook,
)
```

`service_name` is `bedrock-runtime`. `operation_name` is `InvokeModel`,
`InvokeModelWithResponseStream`, `Converse`, or `ConverseStream`. Hooks are
regular synchronous functions, including when using an aioboto3 client. The
Bedrock span is current during each hook, and callback return values are ignored.
Hook exceptions are logged and do not interrupt the application operation.

The request hook runs before the call. For non-streaming calls, the response
hook runs after response processing, before the span ends. For streams, it runs
once at completion: after InvokeModel stream consumption or the final Converse
metadata event. It also runs when an actively consumed iterator is explicitly
closed or fails, with the response available so far. An unconsumed stream does
not trigger a response hook; merely breaking a loop does not guarantee immediate
iterator cleanup.

Hooks receive the original parameter values and the response dictionary returned
to the caller. Treat these as read-only: do not mutate them or read/iterate the
response body in a hook. Streaming bodies have already been consumed at hook
time; use response metadata and the span's recorded attributes for enrichment.
These hooks are separate from the SDK's `span_postprocess_callback`.

## Privacy

**By default, this instrumentation logs prompts, completions, and embeddings to span attributes**. This gives you a clear visibility into how your LLM application is working, and can make it easy to debug and evaluate the quality of the outputs.

However, you may want to disable this logging for privacy reasons, as they may contain highly sensitive data from your users. You may also simply want to reduce the size of your traces.

To disable logging, set the `TRACELOOP_TRACE_CONTENT` environment variable to `false`.

```bash
TRACELOOP_TRACE_CONTENT=false
```
