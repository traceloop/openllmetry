import asyncio
from unittest.mock import MagicMock
import pytest
from opentelemetry.trace import StatusCode
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.instrumentation.anthropic import _awrap, _wrap


@pytest.fixture
def isolated_tracer():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test_anthropic_cancellation")
    return tracer, exporter


@pytest.mark.asyncio
async def test_awrap_cancelled_ends_span(isolated_tracer):
    tracer, exporter = isolated_tracer

    async def failing_call(*args, **kwargs):
        raise asyncio.CancelledError()

    to_wrap = {"span_name": "anthropic.chat"}
    wrapped = _awrap(tracer, None, None, None, None, None, to_wrap)
    mock_instance = MagicMock()

    with pytest.raises(asyncio.CancelledError):
        await wrapped(failing_call, mock_instance, (), {})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.UNSET


def test_wrap_keyboard_interrupt_ends_span(isolated_tracer):
    tracer, exporter = isolated_tracer

    def failing_call(*args, **kwargs):
        raise KeyboardInterrupt()

    to_wrap = {"span_name": "anthropic.chat"}
    wrapped = _wrap(tracer, None, None, None, None, None, to_wrap)
    mock_instance = MagicMock()

    with pytest.raises(KeyboardInterrupt):
        wrapped(failing_call, mock_instance, (), {})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.UNSET


def test_wrap_exception_records_error_status(isolated_tracer):
    tracer, exporter = isolated_tracer

    def failing_call(*args, **kwargs):
        raise ValueError("test error")

    to_wrap = {"span_name": "anthropic.chat"}
    wrapped = _wrap(tracer, None, None, None, None, None, to_wrap)
    mock_instance = MagicMock()

    with pytest.raises(ValueError):
        wrapped(failing_call, mock_instance, (), {})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR
