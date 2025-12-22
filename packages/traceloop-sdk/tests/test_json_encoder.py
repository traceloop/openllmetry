from pathlib import Path
import pytest
from pydantic import BaseModel

from traceloop.sdk.decorators import task
from opentelemetry.semconv_ai import SpanAttributes


def test_json_encoder_task(exporter, recwarn):

    class TestValue(BaseModel):
        value: int

    @task(name="test_task")
    def test_method(a: TestValue, b: TestValue):
        return TestValue(value=a.value + b.value)

    result = test_method(TestValue(value=2), TestValue(value=3))

    assert result.value == 5

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT] == r'{"args": ["{\"value\":2}", "{\"value\":3}"], "kwargs": {}}'
    assert span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] == r'"{\"value\":5}"'

    for warning in recwarn:
        file = Path(warning.filename)
        if file.name == "json_encoder.py" and "`json` method is deprecated" in str(warning.message):
            pytest.fail(f"Deprecation warning found: {warning.message}")
