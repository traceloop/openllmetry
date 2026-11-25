import asyncio
from unittest.mock import MagicMock
import pytest
from opentelemetry.trace import StatusCode
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.instrumentation.openai.shared.chat_wrappers import (
    chat_wrapper,
    achat_wrapper,
)
from opentelemetry.instrumentation.openai.shared.completion_wrappers import (
    completion_wrapper,
    acompletion_wrapper,
)


@pytest.fixture
def isolated_tracer():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test_openai_cancellation")
    return tracer, exporter


@pytest.mark.asyncio
async def test_achat_wrapper_cancelled_ends_span(isolated_tracer):
    tracer, exporter = isolated_tracer

    async def failing_call(*args, **kwargs):
        raise asyncio.CancelledError()

    wrapped = achat_wrapper(tracer, None, None, None, None, None, None)
    mock_instance = MagicMock()

    with pytest.raises(asyncio.CancelledError):
        await wrapped(failing_call, mock_instance, (), {"model": "gpt-4", "messages": []})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.UNSET


def test_chat_wrapper_keyboard_interrupt_ends_span(isolated_tracer):
    tracer, exporter = isolated_tracer

    def failing_call(*args, **kwargs):
        raise KeyboardInterrupt()

    wrapped = chat_wrapper(tracer, None, None, None, None, None, None)
    mock_instance = MagicMock()

    with pytest.raises(KeyboardInterrupt):
        wrapped(failing_call, mock_instance, (), {"model": "gpt-4", "messages": []})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.UNSET


def test_chat_wrapper_exception_records_error(isolated_tracer):
    tracer, exporter = isolated_tracer

    def failing_call(*args, **kwargs):
        raise ValueError("test error")

    wrapped = chat_wrapper(tracer, None, None, None, None, None, None)
    mock_instance = MagicMock()

    with pytest.raises(ValueError):
        wrapped(failing_call, mock_instance, (), {"model": "gpt-4", "messages": []})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR


@pytest.mark.asyncio
async def test_acompletion_wrapper_cancelled_ends_span(isolated_tracer):
    tracer, exporter = isolated_tracer

    async def failing_call(*args, **kwargs):
        raise asyncio.CancelledError()

    wrapped = acompletion_wrapper(tracer)
    mock_instance = MagicMock()

    with pytest.raises(asyncio.CancelledError):
        await wrapped(failing_call, mock_instance, (), {"model": "gpt-3.5-turbo-instruct", "prompt": "hi"})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.UNSET


def test_completion_wrapper_keyboard_interrupt_ends_span(isolated_tracer):
    tracer, exporter = isolated_tracer

    def failing_call(*args, **kwargs):
        raise KeyboardInterrupt()

    wrapped = completion_wrapper(tracer)
    mock_instance = MagicMock()

    with pytest.raises(KeyboardInterrupt):
        wrapped(failing_call, mock_instance, (), {"model": "gpt-3.5-turbo-instruct", "prompt": "hi"})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.UNSET
