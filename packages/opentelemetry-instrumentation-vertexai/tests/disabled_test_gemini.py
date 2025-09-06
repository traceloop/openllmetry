import vertexai
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    event_attributes as EventAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes
from vertexai.preview.generative_models import GenerativeModel, Part

vertexai.init()


def test_vertexai_generate_content(instrument_legacy, span_exporter, log_exporter):
    multimodal_model = GenerativeModel("gemini-2.0-flash-lite")
    response = multimodal_model.generate_content(
        [
            Part.from_uri(
                "gs://generativeai-downloads/images/scones.jpg",
                mime_type="image/jpeg",
            ),
            "what is shown in this image?",
        ]
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "vertexai.generate_content",
    ]

    vertexai_span = spans[0]
    assert (
        "what is shown in this image?"
        in vertexai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.user"]
    )
    assert (
        vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "gemini-2.0-flash-lite"
    )
    assert (
        vertexai_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
        == response._raw_response.usage_metadata.total_token_count
    )
    assert (
        vertexai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == response._raw_response.usage_metadata.prompt_token_count
    )
    assert (
        vertexai_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        == response._raw_response.usage_metadata.candidates_token_count
    )
    assert (
        vertexai_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"]
        == response.text
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


# @pytest.mark.vcr
def test_vertexai_generate_content_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter
):
    multimodal_model = GenerativeModel("gemini-2.0-flash-lite")
    response = multimodal_model.generate_content(
        [
            Part.from_uri(
                "gs://generativeai-downloads/images/scones.jpg",
                mime_type="image/jpeg",
            ),
            "what is shown in this image?",
        ]
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "vertexai.generate_content",
    ]

    vertexai_span = spans[0]

    assert (
        vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "gemini-2.0-flash-lite"
    )
    assert (
        vertexai_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
        == response._raw_response.usage_metadata.total_token_count
    )
    assert (
        vertexai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == response._raw_response.usage_metadata.prompt_token_count
    )
    assert (
        vertexai_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        == response._raw_response.usage_metadata.candidates_token_count
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(
        logs[0],
        "gen_ai.user.message",
        {
            "content": 'file_data {\n  mime_type: "image/jpeg"\n  file_uri: '
            '"gs://generativeai-downloads/images/scones.jpg"\n}\n\nwhat is shown in this image?\n'
        },
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"content": response.text},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


def test_vertexai_generate_content_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter
):
    multimodal_model = GenerativeModel("gemini-2.0-flash-lite")
    response = multimodal_model.generate_content(
        [
            Part.from_uri(
                "gs://generativeai-downloads/images/scones.jpg",
                mime_type="image/jpeg",
            ),
            "what is shown in this image?",
        ]
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "vertexai.generate_content",
    ]

    vertexai_span = spans[0]

    assert (
        vertexai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "gemini-2.0-flash-lite"
    )
    assert (
        vertexai_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
        == response._raw_response.usage_metadata.total_token_count
    )
    assert (
        vertexai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == response._raw_response.usage_metadata.prompt_token_count
    )
    assert (
        vertexai_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        == response._raw_response.usage_metadata.candidates_token_count
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {"index": 0, "finish_reason": "stop", "message": {}}
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.attributes.get(EventAttributes.EVENT_NAME) == event_name
    assert (
        log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM)
        == GenAIAttributes.GenAiSystemValues.VERTEX_AI.value
    )

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
