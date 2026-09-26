import json

import pytest
from oci.generative_ai_inference import models
from opentelemetry.instrumentation.oci_genai.span_utils import (
    GEN_AI_OCI_API_FORMAT,
    GEN_AI_OCI_SERVING_MODE,
    OCI_GENAI_PROVIDER,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes

from tests import assert_message_in_logs, assert_valid_output_message, assert_valid_parts

LLAMA_MODEL = "meta.llama-3.3-70b-instruct"
COHERE_MODEL = "cohere.command-a-03-2025"
OPENAI_MODEL = "openai.gpt-5.4"
PROMPT = "Tell me a joke about opentelemetry"
SERVER_ADDRESS = "inference.generativeai.us-chicago-1.oci.oraclecloud.com"


def _generic_request(text, **kwargs):
    return models.GenericChatRequest(
        api_format=models.BaseChatRequest.API_FORMAT_GENERIC,
        messages=[models.UserMessage(content=[models.TextContent(text=text)])],
        **kwargs,
    )


def _chat(client, compartment_id, model, chat_request):
    return client.chat(
        models.ChatDetails(
            compartment_id=compartment_id,
            serving_mode=models.OnDemandServingMode(model_id=model),
            chat_request=chat_request,
        )
    )


def _assert_common_attributes(span, model, api_format):
    assert span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == OCI_GENAI_PROVIDER
    assert span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "chat"
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == model
    assert span.attributes[GEN_AI_OCI_SERVING_MODE] == "ON_DEMAND"
    assert span.attributes[GEN_AI_OCI_API_FORMAT] == api_format
    assert span.attributes["server.address"] == SERVER_ADDRESS


def _assert_usage(span):
    input_tokens = span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
    output_tokens = span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
    assert input_tokens > 0
    assert output_tokens > 0
    assert span.attributes[SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS] == input_tokens + output_tokens


# ---------------------------------------------------------------------------
# GENERIC api format (Meta Llama)
# ---------------------------------------------------------------------------


@pytest.mark.vcr
def test_generic_chat_legacy(instrument_legacy, oci_client, compartment_id, span_exporter, log_exporter):
    response = _chat(
        oci_client,
        compartment_id,
        LLAMA_MODEL,
        _generic_request(PROMPT, max_tokens=100, temperature=0.2, top_p=0.9, top_k=40, frequency_penalty=0.1),
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [f"chat {LLAMA_MODEL}"]
    span = spans[0]

    _assert_common_attributes(span, LLAMA_MODEL, "GENERIC")
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == LLAMA_MODEL
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 100
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.2
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.9
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_K] == 40
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY] == 0.1
    assert span.attributes[SpanAttributes.GEN_AI_IS_STREAMING] is False

    input_messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    assert input_messages == [{"role": "user", "parts": [{"type": "text", "content": PROMPT}]}]

    generated_text = response.data.chat_response.choices[0].message.content[0].text
    output_messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    assert len(output_messages) == 1
    assert_valid_output_message(output_messages[0])
    assert output_messages[0]["role"] == "assistant"
    assert output_messages[0]["parts"] == [{"type": "text", "content": generated_text}]
    assert output_messages[0]["finish_reason"] == "stop"
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)

    _assert_usage(span)
    usage = response.data.chat_response.usage
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == usage.prompt_tokens
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == usage.completion_tokens

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0, "Assert that it doesn't emit logs when use_attributes is True"


@pytest.mark.vcr
def test_generic_chat_streaming(instrument_legacy, oci_client, compartment_id, span_exporter, log_exporter):
    response = _chat(
        oci_client,
        compartment_id,
        LLAMA_MODEL,
        _generic_request(
            PROMPT,
            max_tokens=100,
            temperature=0.2,
            is_stream=True,
            stream_options=models.StreamOptions(is_include_usage=True),
        ),
    )

    # The span is only completed once the SSE stream has been consumed
    assert span_exporter.get_finished_spans() == ()

    chunks = []
    for event in response.data.events():
        if event.data == "[DONE]":
            continue
        chunks.append(json.loads(event.data))

    streamed_text = "".join(
        part["text"] for chunk in chunks for part in (chunk.get("message") or {}).get("content", [])
    )
    assert streamed_text

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [f"chat {LLAMA_MODEL}"]
    span = spans[0]

    _assert_common_attributes(span, LLAMA_MODEL, "GENERIC")
    assert span.attributes[SpanAttributes.GEN_AI_IS_STREAMING] is True

    input_messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    assert input_messages[0]["parts"][0]["content"] == PROMPT

    output_messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    assert_valid_output_message(output_messages[0])
    assert output_messages[0]["parts"] == [{"type": "text", "content": streamed_text}]
    assert output_messages[0]["finish_reason"] == "stop"
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)

    usage_chunk = next(chunk["usage"] for chunk in chunks if "usage" in chunk)
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == usage_chunk["promptTokens"]
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == usage_chunk["completionTokens"]
    assert span.attributes[SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS] == usage_chunk["totalTokens"]

    assert len(log_exporter.get_finished_logs()) == 0


@pytest.mark.vcr
def test_generic_chat_with_events_with_content(
    instrument_with_content, oci_client, compartment_id, span_exporter, log_exporter
):
    response = _chat(oci_client, compartment_id, LLAMA_MODEL, _generic_request(PROMPT, max_tokens=100))

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [f"chat {LLAMA_MODEL}"]
    span = spans[0]
    _assert_common_attributes(span, LLAMA_MODEL, "GENERIC")
    _assert_usage(span)
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    assert_message_in_logs(logs[0], "gen_ai.user.message", {"content": PROMPT})

    generated_text = response.data.chat_response.choices[0].message.content[0].text
    assert_message_in_logs(
        logs[1],
        "gen_ai.choice",
        {"index": 0, "finish_reason": "stop", "message": {"content": generated_text}},
    )


@pytest.mark.vcr
def test_generic_chat_with_events_with_no_content(
    instrument_with_no_content, oci_client, compartment_id, span_exporter, log_exporter
):
    _chat(oci_client, compartment_id, LLAMA_MODEL, _generic_request(PROMPT, max_tokens=100))

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [f"chat {LLAMA_MODEL}"]
    span = spans[0]
    _assert_common_attributes(span, LLAMA_MODEL, "GENERIC")
    _assert_usage(span)
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    assert_message_in_logs(logs[0], "gen_ai.user.message", {})
    assert_message_in_logs(logs[1], "gen_ai.choice", {"index": 0, "finish_reason": "stop", "message": {}})


@pytest.mark.vcr
def test_generic_chat_streaming_with_events_with_content(
    instrument_with_content, oci_client, compartment_id, span_exporter, log_exporter
):
    response = _chat(
        oci_client,
        compartment_id,
        LLAMA_MODEL,
        _generic_request(PROMPT, max_tokens=100, is_stream=True),
    )
    streamed_text = "".join(
        part["text"]
        for event in response.data.events()
        if event.data != "[DONE]"
        for part in (json.loads(event.data).get("message") or {}).get("content", [])
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [f"chat {LLAMA_MODEL}"]
    assert spans[0].attributes[SpanAttributes.GEN_AI_IS_STREAMING] is True
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in spans[0].attributes

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2
    assert_message_in_logs(logs[0], "gen_ai.user.message", {"content": PROMPT})
    assert_message_in_logs(
        logs[1],
        "gen_ai.choice",
        {"index": 0, "finish_reason": "stop", "message": {"content": streamed_text}},
    )


# ---------------------------------------------------------------------------
# GENERIC api format (OpenAI)
# ---------------------------------------------------------------------------


@pytest.mark.vcr
def test_openai_generic_chat_legacy(instrument_legacy, oci_client, compartment_id, span_exporter, log_exporter):
    chat_request = models.GenericChatRequest(
        api_format=models.BaseChatRequest.API_FORMAT_GENERIC,
        messages=[
            models.SystemMessage(content=[models.TextContent(text="You are a terse assistant.")]),
            models.UserMessage(content=[models.TextContent(text=PROMPT)]),
        ],
        max_completion_tokens=120,
    )
    response = _chat(oci_client, compartment_id, OPENAI_MODEL, chat_request)

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [f"chat {OPENAI_MODEL}"]
    span = spans[0]

    _assert_common_attributes(span, OPENAI_MODEL, "GENERIC")
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == OPENAI_MODEL
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 120

    input_messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    assert [message["role"] for message in input_messages] == ["system", "user"]
    assert input_messages[0]["parts"] == [{"type": "text", "content": "You are a terse assistant."}]
    assert input_messages[1]["parts"] == [{"type": "text", "content": PROMPT}]

    generated_text = response.data.chat_response.choices[0].message.content[0].text
    output_messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    assert_valid_output_message(output_messages[0])
    assert output_messages[0]["parts"] == [{"type": "text", "content": generated_text}]
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)

    _assert_usage(span)
    usage = response.data.chat_response.usage
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == (
        usage.prompt_tokens_details.cached_tokens
    )
    assert span.attributes[SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS] == (
        usage.completion_tokens_details.reasoning_tokens
    )

    assert len(log_exporter.get_finished_logs()) == 0


# ---------------------------------------------------------------------------
# COHERE api format
# ---------------------------------------------------------------------------


def _cohere_request(**kwargs):
    return models.CohereChatRequest(
        api_format=models.BaseChatRequest.API_FORMAT_COHERE,
        preamble_override="You are a terse assistant.",
        chat_history=[
            models.CohereUserMessage(message="Hello"),
            models.CohereChatBotMessage(message="Hi! How can I help?"),
        ],
        message=PROMPT,
        **kwargs,
    )


@pytest.mark.vcr
def test_cohere_chat_legacy(instrument_legacy, oci_client, compartment_id, span_exporter, log_exporter):
    response = _chat(
        oci_client,
        compartment_id,
        COHERE_MODEL,
        _cohere_request(max_tokens=100, temperature=0.3, top_p=0.8, seed=42, stop_sequences=["END"]),
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [f"chat {COHERE_MODEL}"]
    span = spans[0]

    _assert_common_attributes(span, COHERE_MODEL, "COHERE")
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == COHERE_MODEL
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 100
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.3
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.8
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_SEED] == 42
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES] == ("END",)
    assert span.attributes[SpanAttributes.GEN_AI_IS_STREAMING] is False

    system_instructions = json.loads(span.attributes[GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS])
    assert system_instructions == [{"type": "text", "content": "You are a terse assistant."}]

    input_messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    assert [message["role"] for message in input_messages] == ["user", "assistant", "user"]
    for message in input_messages:
        assert_valid_parts(message["parts"])
    assert input_messages[0]["parts"] == [{"type": "text", "content": "Hello"}]
    assert input_messages[1]["parts"] == [{"type": "text", "content": "Hi! How can I help?"}]
    assert input_messages[2]["parts"] == [{"type": "text", "content": PROMPT}]

    output_messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    assert len(output_messages) == 1
    assert_valid_output_message(output_messages[0])
    assert output_messages[0]["parts"] == [{"type": "text", "content": response.data.chat_response.text}]
    # The service answered this request (which sets stop_sequences) with ``finishReason: null``; the
    # OutputMessage schema still requires the key, so it is emitted as an empty string and the
    # span-level finish reasons attribute is omitted.
    assert response.data.chat_response.finish_reason is None
    assert output_messages[0]["finish_reason"] == ""
    assert GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS not in span.attributes

    _assert_usage(span)
    assert len(log_exporter.get_finished_logs()) == 0


@pytest.mark.vcr
def test_cohere_chat_streaming(instrument_legacy, oci_client, compartment_id, span_exporter, log_exporter):
    response = _chat(
        oci_client,
        compartment_id,
        COHERE_MODEL,
        _cohere_request(
            max_tokens=100, temperature=0.3, is_stream=True, stream_options=models.StreamOptions(is_include_usage=True)
        ),
    )

    assert span_exporter.get_finished_spans() == ()

    chunks = [json.loads(event.data) for event in response.data.events() if event.data != "[DONE]"]
    final_chunk = next(chunk for chunk in chunks if "finishReason" in chunk)

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [f"chat {COHERE_MODEL}"]
    span = spans[0]

    _assert_common_attributes(span, COHERE_MODEL, "COHERE")
    assert span.attributes[SpanAttributes.GEN_AI_IS_STREAMING] is True

    output_messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    assert_valid_output_message(output_messages[0])
    assert output_messages[0]["parts"] == [{"type": "text", "content": final_chunk["text"]}]
    assert output_messages[0]["finish_reason"] == "stop"
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)

    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == final_chunk["usage"]["promptTokens"]
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == final_chunk["usage"]["completionTokens"]
    assert span.attributes[SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS] == final_chunk["usage"]["totalTokens"]

    assert len(log_exporter.get_finished_logs()) == 0


@pytest.mark.vcr
def test_cohere_chat_with_events_with_content(
    instrument_with_content, oci_client, compartment_id, span_exporter, log_exporter
):
    response = _chat(oci_client, compartment_id, COHERE_MODEL, _cohere_request(max_tokens=100))

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [f"chat {COHERE_MODEL}"]
    _assert_common_attributes(spans[0], COHERE_MODEL, "COHERE")
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in spans[0].attributes
    assert GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS not in spans[0].attributes

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 5
    assert_message_in_logs(logs[0], "gen_ai.system.message", {"content": "You are a terse assistant."})
    assert_message_in_logs(logs[1], "gen_ai.user.message", {"content": "Hello"})
    assert_message_in_logs(logs[2], "gen_ai.assistant.message", {"content": "Hi! How can I help?"})
    assert_message_in_logs(logs[3], "gen_ai.user.message", {"content": PROMPT})
    assert_message_in_logs(
        logs[4],
        "gen_ai.choice",
        {"index": 0, "finish_reason": "stop", "message": {"content": response.data.chat_response.text}},
    )
