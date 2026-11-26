import pytest
from opentelemetry._logs import get_logger
from opentelemetry.instrumentation.watsonx import (
    _emit_response_events,
    _handle_stream_response,
)
from opentelemetry.instrumentation.watsonx.config import Config
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogExporter,
    SimpleLogRecordProcessor,
)


class RecordingSpan:
    def __init__(self):
        self.attributes = {}

    def is_recording(self):
        return True

    def set_attribute(self, name, value):
        self.attributes[name] = value


@pytest.fixture
def event_logger():
    previous = Config.use_legacy_attributes
    Config.use_legacy_attributes = False

    exporter = InMemoryLogExporter()
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))

    yield get_logger(__name__, logger_provider=provider), exporter

    Config.use_legacy_attributes = previous


def test_emit_response_events_writes_choice_log(event_logger):
    logger, exporter = event_logger

    _emit_response_events(
        {"results": [{"generated_text": "Two", "stop_reason": "eos_token"}]},
        logger,
    )

    logs = exporter.get_finished_logs()
    assert len(logs) == 1
    assert logs[0].log_record.event_name == "gen_ai.choice"
    assert logs[0].log_record.body == {
        "index": 0,
        "message": {"content": "Two"},
        "finish_reason": "eos_token",
    }


def test_emit_response_events_writes_choice_log_for_list_response(event_logger):
    logger, exporter = event_logger

    _emit_response_events(
        [
            {"results": [{"generated_text": "One", "stop_reason": "eos_token"}]},
            {"results": [{"generated_text": "Two", "stop_reason": "length"}]},
        ],
        logger,
    )

    logs = exporter.get_finished_logs()
    assert len(logs) == 2
    assert logs[0].log_record.event_name == "gen_ai.choice"
    assert logs[0].log_record.body == {
        "index": 0,
        "message": {"content": "One"},
        "finish_reason": "eos_token",
    }
    assert logs[1].log_record.event_name == "gen_ai.choice"
    assert logs[1].log_record.body == {
        "index": 1,
        "message": {"content": "Two"},
        "finish_reason": "length",
    }


def test_emit_response_events_skips_invalid_batched_items(event_logger):
    logger, exporter = event_logger

    _emit_response_events(
        [
            "not a response",
            {
                "results": [
                    {"generated_text": "One", "stop_reason": "eos_token"},
                    None,
                    {"generated_text": "Two", "stop_reason": "length"},
                ]
            },
            {"generated_text": "Three", "stop_reason": "cancelled"},
            {"results": ["not a message"]},
        ],
        logger,
    )

    logs = exporter.get_finished_logs()
    assert len(logs) == 3
    assert [log.log_record.event_name for log in logs] == [
        "gen_ai.choice",
        "gen_ai.choice",
        "gen_ai.choice",
    ]
    assert [log.log_record.body for log in logs] == [
        {
            "index": 0,
            "message": {"content": "One"},
            "finish_reason": "eos_token",
        },
        {
            "index": 2,
            "message": {"content": "Two"},
            "finish_reason": "length",
        },
        {
            "index": 3,
            "message": {"content": "Three"},
            "finish_reason": "cancelled",
        },
    ]


def test_handle_stream_response_writes_choice_log(event_logger):
    logger, exporter = event_logger
    span = RecordingSpan()
    stream_response = {
        "model_id": "ibm/granite",
        "generated_text": "streamed response",
        "generated_token_count": 3,
        "input_token_count": 2,
    }

    _handle_stream_response(span, logger, stream_response, "streamed response", "stop_sequence")

    logs = exporter.get_finished_logs()
    assert len(logs) == 1
    assert logs[0].log_record.event_name == "gen_ai.choice"
    assert logs[0].log_record.body == {
        "index": 0,
        "message": {"content": "streamed response"},
        "finish_reason": "stop_sequence",
    }
