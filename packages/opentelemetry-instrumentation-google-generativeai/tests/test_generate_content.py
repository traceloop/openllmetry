from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes  # noqa: F401


def test_gemini_generate_content_legacy(
    instrument_legacy, span_exporter, log_exporter, genai_client
):
    # This test is working, but since Gemini uses gRPC,
    # vcr does not record it, therefore we cannot test this without
    # setting the API key in a shared secret store like GitHub secrets
    pass

    # genai_client.generate_content(
    #     "The opposite of hot is",
    # )
    # spans = span_exporter.get_finished_spans()
    # assert all(span.name == "gemini.generate_content" for span in spans)

    # gemini_span = spans[0]
    # assert (
    #     gemini_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
    #     == "The opposite of hot is\n"
    # )
    # assert gemini_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"] == "user"
    # assert (
    #     gemini_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
    #     == "cold\n"
    # )
    # assert (
    #     gemini_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role")
    #     == "assistant"
    # )

    # assert gemini_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 5
    # assert (
    #     gemini_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
    #     + gemini_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
    #     == gemini_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    # )

    # assert (
    #     gemini_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
    #     == "models/gemini-1.5-flash"
    # )
    # assert (
    #     gemini_span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL]
    #     == "models/gemini-1.5-flash"
    # )

    # logs = log_exporter.get_finished_logs()
    # assert (
    #     len(logs) == 0
    # ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


def test_gemini_generate_content_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, genai_client
):
    # This test is working, but since Gemini uses gRPC,
    # vcr does not record it, therefore we cannot test this without
    # setting the API key in a shared secret store like GitHub secrets
    pass

    # genai_client.generate_content(
    #     "The opposite of hot is",
    # )
    # spans = span_exporter.get_finished_spans()
    # assert all(span.name == "gemini.generate_content" for span in spans)

    # gemini_span = spans[0]

    # assert gemini_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 5
    # assert (
    #     gemini_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
    #     + gemini_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
    #     == gemini_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    # )

    # assert (
    #     gemini_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
    #     == "models/gemini-1.5-flash"
    # )
    # assert (
    #     gemini_span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL]
    #     == "models/gemini-1.5-flash"
    # )

    # logs = log_exporter.get_finished_logs()
    # assert len(logs) == 2

    # # Validate user message Event
    # user_message = {"content": "The opposite of hot is"}
    # assert_message_in_logs(logs[0], "gen_ai.user.message", user_message)

    # # Validate the AI response
    # ai_response = {
    #     "index": 0,
    #     "finish_reason": "STOP",
    #     "message": {"content": [{"text": "cold\n"}], "role": "model"},
    # }
    # assert_message_in_logs(logs[1], "gen_ai.choice", ai_response)


def test_gemini_generate_content_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, genai_client
):
    # This test is working, but since Gemini uses gRPC,
    # vcr does not record it, therefore we cannot test this without
    # setting the API key in a shared secret store like GitHub secrets
    pass

    # genai_client.generate_content(
    #     "The opposite of hot is",
    # )
    # spans = span_exporter.get_finished_spans()
    # assert all(span.name == "gemini.generate_content" for span in spans)

    # gemini_span = spans[0]

    # assert gemini_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 5
    # assert (
    #     gemini_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
    #     + gemini_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
    #     == gemini_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    # )

    # assert (
    #     gemini_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
    #     == "models/gemini-1.5-flash"
    # )
    # assert (
    #     gemini_span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL]
    #     == "models/gemini-1.5-flash"
    # )

    # logs = log_exporter.get_finished_logs()
    # assert len(logs) == 2

    # # Validate user message Event
    # assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    # # Validate the AI response
    # ai_response = {
    #     "index": 0,
    #     "finish_reason": "STOP",
    #     "message": {},
    # }
    # assert_message_in_logs(logs[1], "gen_ai.choice", ai_response)


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.event_name == event_name
    assert log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "gemini"

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
