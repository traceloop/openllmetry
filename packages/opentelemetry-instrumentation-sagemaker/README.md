# OpenTelemetry SageMaker Instrumentation

<a href="https://pypi.org/project/opentelemetry-instrumentation-sagemaker/">
    <img src="https://badge.fury.io/py/opentelemetry-instrumentation-sagemaker.svg">
</a>

This library allows tracing of any models deployed on Amazon SageMaker and invoked with [Boto3](https://github.com/boto/boto3) to SageMaker.

## Installation

```bash
pip install opentelemetry-instrumentation-sagemaker
```

## Example usage

```python
from opentelemetry.instrumentation.sagemaker import SageMakerInstrumentor

SageMakerInstrumentor().instrument()
```

## Privacy

**By default, this instrumentation logs SageMaker endpoint request bodies and responses to span attributes**. This gives you a clear visibility into how your LLM application is working, and can make it easy to debug and evaluate the quality of the outputs.

However, you may want to disable this logging for privacy reasons, as they may contain highly sensitive data from your users. You may also simply want to reduce the size of your traces.

To disable logging, set the `TRACELOOP_TRACE_CONTENT` environment variable to `false`.

```bash
TRACELOOP_TRACE_CONTENT=false
```
