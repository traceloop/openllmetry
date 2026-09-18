"""Unit tests for span_utils / streaming helpers (no network, no cassettes)."""

import json
from unittest.mock import MagicMock

import pytest
from oci.generative_ai_inference import models
from opentelemetry.instrumentation.oci_genai.span_utils import (
    CHAT,
    EMBEDDINGS,
    GEN_AI_OCI_ENDPOINT_ID,
    GEN_AI_OCI_SERVING_MODE,
    RERANK,
    chat_request_to_messages,
    chat_response_to_messages,
    get_request_model,
    map_finish_reason,
    set_input_attributes,
    set_request_attributes,
    set_response_attributes,
    stream_choices_to_messages,
    usage_to_dict,
)
from opentelemetry.instrumentation.oci_genai.streaming import (
    OCIGenAIStreamWrapper,
    StreamAccumulator,
)
from opentelemetry.instrumentation.oci_genai.utils import get_server_address
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from tests import assert_valid_output_message, assert_valid_parts


def _span(recording=True):
    span = MagicMock()
    span.is_recording.return_value = recording
    attributes = {}
    span.set_attribute = lambda name, value: attributes.__setitem__(name, value)
    span._attrs = attributes
    return span


# ---------------------------------------------------------------------------
# finish reasons / usage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("stop", "stop"),
        ("length", "length"),
        ("tool_calls", "tool_call"),
        ("content_filter", "content_filter"),
        ("COMPLETE", "stop"),
        ("STOP_SEQUENCE", "stop"),
        ("MAX_TOKENS", "length"),
        ("TOOL_CALL", "tool_call"),
        ("ERROR_TOXIC", "content_filter"),
        ("ERROR", "error"),
        ("USER_CANCEL", "user_cancel"),
        (None, ""),
    ],
)
def test_map_finish_reason(raw, expected):
    assert map_finish_reason(raw) == expected


def test_usage_to_dict_from_model_and_stream_payload():
    usage = models.Usage(
        prompt_tokens=19,
        completion_tokens=11,
        total_tokens=30,
        prompt_tokens_details=models.PromptTokensDetails(cached_tokens=5),
        completion_tokens_details=models.CompletionTokensDetails(reasoning_tokens=3),
    )
    assert usage_to_dict(usage) == {
        "input_tokens": 19,
        "output_tokens": 11,
        "total_tokens": 30,
        "cached_tokens": 5,
        "reasoning_tokens": 3,
    }

    stream_usage = usage_to_dict({"promptTokens": 41, "completionTokens": 8})
    assert stream_usage["input_tokens"] == 41
    assert stream_usage["output_tokens"] == 8
    assert stream_usage["total_tokens"] == 49

    embeddings_usage = usage_to_dict(models.Usage(prompt_tokens=4, completion_tokens=0, total_tokens=4), False)
    assert embeddings_usage["output_tokens"] is None
    assert embeddings_usage["total_tokens"] == 4


# ---------------------------------------------------------------------------
# request -> input messages
# ---------------------------------------------------------------------------


def test_generic_request_with_tools_and_images():
    request = models.GenericChatRequest(
        api_format="GENERIC",
        messages=[
            models.SystemMessage(content=[models.TextContent(text="Be helpful")]),
            models.UserMessage(
                content=[
                    models.TextContent(text="What is in this picture?"),
                    models.ImageContent(image_url=models.ImageUrl(url="https://example.com/cat.png")),
                    models.ImageContent(image_url=models.ImageUrl(url="data:image/png;base64,AAAA")),
                ]
            ),
            models.AssistantMessage(
                content=[models.TextContent(text="Let me check")],
                tool_calls=[
                    models.FunctionCall(id="call_1", name="lookup", arguments='{"animal": "cat"}'),
                ],
            ),
            models.ToolMessage(tool_call_id="call_1", content=[models.TextContent(text="a tabby cat")]),
        ],
    )

    messages, system_parts = chat_request_to_messages(request)
    assert system_parts == []
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "tool"]
    for message in messages:
        assert_valid_parts(message["parts"])

    assert messages[1]["parts"] == [
        {"type": "text", "content": "What is in this picture?"},
        {"type": "uri", "modality": "image", "uri": "https://example.com/cat.png"},
        {"type": "blob", "modality": "image", "mime_type": "image/png", "content": "AAAA"},
    ]
    assert messages[2]["parts"] == [
        {"type": "text", "content": "Let me check"},
        {"type": "tool_call", "name": "lookup", "arguments": {"animal": "cat"}, "id": "call_1"},
    ]
    assert messages[3]["parts"] == [{"type": "tool_call_response", "id": "call_1", "response": "a tabby cat"}]


def test_cohere_request_with_history_and_tool_results():
    request = models.CohereChatRequest(
        api_format="COHERE",
        preamble_override="You are terse.",
        message="Book it",
        chat_history=[
            models.CohereUserMessage(message="Find flights"),
            models.CohereChatBotMessage(
                message="Searching",
                tool_calls=[models.CohereToolCall(name="search_flights", parameters={"to": "CDG"})],
            ),
            models.CohereToolMessage(
                tool_results=[
                    models.CohereToolResult(
                        call=models.CohereToolCall(name="search_flights", parameters={"to": "CDG"}),
                        outputs=[{"flight": "AF123"}],
                    )
                ]
            ),
        ],
    )

    messages, system_parts = chat_request_to_messages(request)
    assert system_parts == [{"type": "text", "content": "You are terse."}]
    assert [message["role"] for message in messages] == ["user", "assistant", "tool", "user"]
    for message in messages:
        assert_valid_parts(message["parts"])
    assert messages[1]["parts"][1] == {"type": "tool_call", "name": "search_flights", "arguments": {"to": "CDG"}}
    assert messages[2]["parts"] == [
        {"type": "tool_call_response", "id": "search_flights", "response": [{"flight": "AF123"}]}
    ]
    assert messages[3]["parts"] == [{"type": "text", "content": "Book it"}]


def test_cohere_v2_request_and_response():
    request = models.CohereChatRequestV2(
        api_format="COHEREV2",
        messages=[
            models.CohereSystemMessageV2(content=[models.CohereTextContentV2(text="Be terse")]),
            models.CohereUserMessageV2(content=[models.CohereTextContentV2(text="Hi")]),
        ],
    )
    messages, system_parts = chat_request_to_messages(request)
    assert system_parts == []
    assert messages == [
        {"role": "system", "parts": [{"type": "text", "content": "Be terse"}]},
        {"role": "user", "parts": [{"type": "text", "content": "Hi"}]},
    ]

    response = models.CohereChatResponseV2(
        api_format="COHEREV2",
        id="resp-1",
        finish_reason="TOOL_CALL",
        message=models.CohereAssistantMessageV2(
            content=[
                models.CohereTextContentV2(text="Calling a tool"),
                models.CohereThinkingContentV2(thinking="need weather"),
            ],
            tool_calls=[
                # ``CohereToolCallV2.function`` is an untyped object (plain dict) in the SDK
                models.CohereToolCallV2(id="tc-1", function={"name": "weather", "arguments": '{"city": "Paris"}'})
            ],
        ),
    )
    output = chat_response_to_messages(response)
    assert len(output) == 1
    assert_valid_output_message(output[0])
    assert output[0]["finish_reason"] == "tool_call"
    assert output[0]["parts"] == [
        {"type": "text", "content": "Calling a tool"},
        {"type": "reasoning", "content": "need weather"},
        {"type": "tool_call", "name": "weather", "arguments": {"city": "Paris"}, "id": "tc-1"},
    ]


def test_generic_response_with_tool_calls_and_multiple_choices():
    response = models.GenericChatResponse(
        api_format="GENERIC",
        choices=[
            models.ChatChoice(
                index=0,
                finish_reason="tool_calls",
                message=models.AssistantMessage(
                    content=[],
                    tool_calls=[models.FunctionCall(id="c1", name="get_time", arguments="{}")],
                ),
            ),
            models.ChatChoice(
                index=1,
                finish_reason="length",
                message=models.AssistantMessage(content=[models.TextContent(text="Partial")]),
            ),
        ],
    )
    output = chat_response_to_messages(response)
    assert [message["finish_reason"] for message in output] == ["tool_call", "length"]
    assert output[0]["parts"] == [{"type": "tool_call", "name": "get_time", "arguments": {}, "id": "c1"}]
    assert output[1]["parts"] == [{"type": "text", "content": "Partial"}]


# ---------------------------------------------------------------------------
# dedicated endpoints / attribute setters
# ---------------------------------------------------------------------------


def test_dedicated_serving_mode_attributes():
    endpoint_id = "ocid1.generativeaiendpoint.oc1.us-chicago-1.example"
    details = models.ChatDetails(
        compartment_id="ocid1.compartment.oc1..example",
        serving_mode=models.DedicatedServingMode(endpoint_id=endpoint_id),
        chat_request=models.GenericChatRequest(api_format="GENERIC", messages=[], is_stream=True),
    )
    assert get_request_model(details) == endpoint_id

    span = _span()
    set_request_attributes(span, CHAT, details, instance=None)
    assert span._attrs[GEN_AI_OCI_SERVING_MODE] == "DEDICATED"
    assert span._attrs[GEN_AI_OCI_ENDPOINT_ID] == endpoint_id
    assert span._attrs["gen_ai.is_streaming"] is True
    assert "server.address" not in span._attrs


def test_get_server_address_strips_endpoint_templates():
    instance = MagicMock()
    instance.base_client.endpoint = "https://inference.generativeai.us-chicago-1.{dualStack?ds.:}oci.oraclecloud.com/20231130"
    assert get_server_address(instance) == "inference.generativeai.us-chicago-1.oci.oraclecloud.com"
    assert get_server_address(object()) is None


def test_input_attributes_respect_content_tracing(monkeypatch):
    details = models.ChatDetails(
        serving_mode=models.OnDemandServingMode(model_id="m"),
        chat_request=models.GenericChatRequest(
            api_format="GENERIC", messages=[models.UserMessage(content=[models.TextContent(text="secret")])]
        ),
    )
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    span = _span()
    set_input_attributes(span, CHAT, details)
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span._attrs

    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
    set_input_attributes(span, CHAT, details)
    assert json.loads(span._attrs[GenAIAttributes.GEN_AI_INPUT_MESSAGES])[0]["parts"][0]["content"] == "secret"


def test_embeddings_and_rerank_request_attributes():
    span = _span()
    embed_details = models.EmbedTextDetails(
        serving_mode=models.OnDemandServingMode(model_id="cohere.embed-v4.0"),
        inputs=["a", "b", "c"],
        embedding_types=["float"],
        output_dimensions=256,
    )
    set_request_attributes(span, EMBEDDINGS, embed_details, instance=None)
    assert span._attrs["gen_ai.oci.embeddings.input_count"] == 3
    assert span._attrs[GenAIAttributes.GEN_AI_REQUEST_ENCODING_FORMATS] == ["float"]
    assert span._attrs[GenAIAttributes.GEN_AI_EMBEDDINGS_DIMENSION_COUNT] == 256

    result = models.EmbedTextResult(id="req", model_id="cohere.embed-v4.0", embeddings=[[0.1] * 256] * 3)
    response = MagicMock()
    response.data = result
    assert set_response_attributes(span, EMBEDDINGS, response) is None
    assert span._attrs[GenAIAttributes.GEN_AI_EMBEDDINGS_DIMENSION_COUNT] == 256
    assert span._attrs[GenAIAttributes.GEN_AI_RESPONSE_ID] == "req"

    span = _span()
    rerank_details = models.RerankTextDetails(
        serving_mode=models.OnDemandServingMode(model_id="cohere.rerank-v4.0-fast"),
        input="q",
        documents=["d1", "d2"],
        top_n=1,
    )
    set_request_attributes(span, RERANK, rerank_details, instance=None)
    assert span._attrs["gen_ai.oci.rerank.top_n"] == 1
    assert span._attrs["gen_ai.oci.rerank.document_count"] == 2


# ---------------------------------------------------------------------------
# streaming accumulator
# ---------------------------------------------------------------------------

GENERIC_EVENTS = [
    '{"index":0,"message":{"role":"ASSISTANT","content":[{"type":"TEXT","text":""}]},"pad":"aaa"}',
    '{"index":0,"message":{"role":"ASSISTANT","content":[{"type":"TEXT","text":"Hello"}]},"pad":"a"}',
    '{"index":0,"message":{"role":"ASSISTANT","content":[{"type":"TEXT","text":", world"}]},"pad":"aa"}',
    '{"message":{"role":"ASSISTANT","content":[{"type":"TEXT","text":""}]},"finishReason":"stop","pad":"aaaa"}',
    '{"usage":{"completionTokens":8,"promptTokens":41,"totalTokens":49},"pad":"aaaa"}',
    "[DONE]",
]

COHERE_EVENTS = [
    '{"apiFormat":"COHERE","text":"Hi","pad":"aaaa"}',
    '{"apiFormat":"COHERE","text":" there","pad":"aa"}',
    '{"apiFormat":"COHERE","text":"Hi there","chatHistory":[{"role":"USER","message":"Say hi."},'
    '{"role":"CHATBOT","message":"Hi there"}],"finishReason":"COMPLETE","pad":"a",'
    '"usage":{"completionTokens":2,"promptTokens":6,"totalTokens":8}}',
]


def test_stream_accumulator_generic():
    accumulator = StreamAccumulator()
    for event in GENERIC_EVENTS:
        accumulator.process(event)
    choices = accumulator.choices_list()
    assert len(choices) == 1
    assert choices[0]["text"] == "Hello, world"
    assert choices[0]["finish_reason"] == "stop"
    assert accumulator.usage == {"completionTokens": 8, "promptTokens": 41, "totalTokens": 49}

    messages = stream_choices_to_messages(choices)
    assert_valid_output_message(messages[0])
    assert messages[0] == {
        "role": "assistant",
        "parts": [{"type": "text", "content": "Hello, world"}],
        "finish_reason": "stop",
    }


def test_stream_accumulator_cohere_terminal_event_carries_full_text():
    accumulator = StreamAccumulator()
    for event in COHERE_EVENTS:
        accumulator.process(event)
    choices = accumulator.choices_list()
    assert accumulator.api_format == "COHERE"
    assert choices[0]["text"] == "Hi there"
    assert choices[0]["finish_reason"] == "COMPLETE"
    assert usage_to_dict(accumulator.usage)["total_tokens"] == 8
    assert stream_choices_to_messages(choices)[0]["finish_reason"] == "stop"


def test_stream_accumulator_tool_call_deltas_and_garbage():
    accumulator = StreamAccumulator()
    accumulator.process("not json")
    accumulator.process('{"message":{"role":"ASSISTANT","toolCalls":[{"id":"c1","name":"get_weather","arguments":"{\\"ci"}]}}')
    accumulator.process('{"message":{"role":"ASSISTANT","toolCalls":[{"id":"c1","arguments":"ty\\": \\"Paris\\"}"}]}}')
    accumulator.process('{"message":{"role":"ASSISTANT","content":[]},"finishReason":"tool_calls"}')
    messages = stream_choices_to_messages(accumulator.choices_list())
    assert messages[0]["finish_reason"] == "tool_call"
    assert messages[0]["parts"] == [
        {"type": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}, "id": "c1"}
    ]


def test_stream_wrapper_invokes_callback_once_and_on_error():
    class FakeEvent:
        def __init__(self, data):
            self.data = data

    class FakeSSEClient:
        def __init__(self, events, fail=False):
            self._events = events
            self._fail = fail

        def events(self):
            for event in self._events:
                yield FakeEvent(event)
            if self._fail:
                raise RuntimeError("boom")

    calls = []
    wrapper = OCIGenAIStreamWrapper(
        FakeSSEClient(GENERIC_EVENTS), StreamAccumulator(), lambda acc, err: calls.append((acc, err))
    )
    events = list(wrapper.events())
    assert len(events) == len(GENERIC_EVENTS)
    assert len(calls) == 1
    assert calls[0][1] is None
    assert calls[0][0].choices_list()[0]["text"] == "Hello, world"

    calls.clear()
    wrapper = OCIGenAIStreamWrapper(
        FakeSSEClient(GENERIC_EVENTS[:2], fail=True), StreamAccumulator(), lambda acc, err: calls.append((acc, err))
    )
    with pytest.raises(RuntimeError, match="boom"):
        list(wrapper.events())
    assert len(calls) == 1
    assert isinstance(calls[0][1], RuntimeError)
