from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
from openai import AssistantEventHandler
from opentelemetry.instrumentation.openai.v1.event_handler_wrapper import EventHandleWrapper
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAIAttributes
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.trace import StatusCode


def _finish_handler(handler, through_stream):
    if through_stream:
        stream = MagicMock()
        stream.__iter__.return_value = iter(())
        handler._init(stream)
        handler.until_done()
    else:
        handler.on_end()


@pytest.mark.parametrize("through_stream", [False, True])
def test_end_callback_failure_ends_span(tracer_provider, span_exporter, through_stream):
    error = RuntimeError("end callback failed")
    original = Mock(spec=AssistantEventHandler)
    original.on_end.side_effect = error
    span = tracer_provider.get_tracer(__name__).start_span("openai.assistant.run_stream")
    handler = EventHandleWrapper(original, span)

    with pytest.raises(RuntimeError) as raised:
        _finish_handler(handler, through_stream)

    assert raised.value is error
    original.on_end.assert_called_once_with()
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR
    assert spans[0].status.description == str(error)
    assert spans[0].attributes[ERROR_TYPE] == "RuntimeError"
    assert len(spans[0].events) == 1
    assert spans[0].events[0].name == "exception"
    assert spans[0].events[0].attributes["exception.message"] == str(error)


@pytest.mark.parametrize("through_stream", [False, True])
def test_end_callback_success_ends_span(tracer_provider, span_exporter, through_stream):
    original = Mock(spec=AssistantEventHandler)
    span = tracer_provider.get_tracer(__name__).start_span("openai.assistant.run_stream")
    handler = EventHandleWrapper(original, span)

    handler.on_run_step_done(SimpleNamespace(usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3)))

    _finish_handler(handler, through_stream)

    original.on_end.assert_called_once_with()
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.UNSET
    assert not spans[0].events
    assert spans[0].attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 7
    assert spans[0].attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 3
    assert ERROR_TYPE not in spans[0].attributes


def test_end_callback_interrupt_ends_span(tracer_provider, span_exporter):
    error = KeyboardInterrupt()
    original = Mock(spec=AssistantEventHandler)
    original.on_end.side_effect = error
    span = tracer_provider.get_tracer(__name__).start_span("openai.assistant.run_stream")
    handler = EventHandleWrapper(original, span)

    with pytest.raises(KeyboardInterrupt) as raised:
        handler.on_end()

    assert raised.value is error
    original.on_end.assert_called_once_with()
    assert len(span_exporter.get_finished_spans()) == 1
