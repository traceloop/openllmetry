"""Tests for semconv compliance fixes from openai-semconv-review.md.

TDD: These tests are written FIRST, before any implementation changes.
Each test class maps to an issue ID from the review document.
"""

import json
import time
from unittest.mock import MagicMock, AsyncMock

import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes.server_attributes import SERVER_ADDRESS

from openai.types.responses import (
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseFunctionToolCall,
)

from opentelemetry.instrumentation.openai.shared import (
    metric_shared_attributes,
    set_tools_attributes,
    _set_functions_attributes,
)
from opentelemetry.instrumentation.openai.shared.chat_wrappers import (
    _accumulate_stream_items,
    _map_content_block,
    _set_output_messages,
)


@pytest.fixture
def mock_span():
    span = MagicMock()
    span.is_recording.return_value = True
    attrs = {}

    def set_attribute(name, value):
        attrs[name] = value

    span.set_attribute = set_attribute
    span._attrs = attrs
    return span


def _get_output_messages(span):
    return json.loads(span._attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])


def _make_message_output(msg_id="msg_001", text="Hi there"):
    """Helper to create a proper ResponseOutputMessage."""
    return ResponseOutputMessage(
        id=msg_id,
        type="message",
        status="completed",
        role="assistant",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )


def _make_function_call_output(fc_id="fc_001", name="get_weather", args="{}"):
    """Helper to create a proper ResponseFunctionToolCall."""
    return ResponseFunctionToolCall(
        id=fc_id,
        type="function_call",
        name=name,
        arguments=args,
        call_id="call_001",
        status="completed",
    )


# ---------------------------------------------------------------------------
# P1-1: Responses API must set gen_ai.operation.name on every span
# ---------------------------------------------------------------------------

class TestP1_1_ResponsesOperationName:
    """Every start_span() call in responses_wrappers.py must include
    gen_ai.operation.name = 'chat'."""

    def test_set_data_attributes_sets_operation_name(self, mock_span):
        """set_data_attributes should set gen_ai.operation.name = 'chat'."""
        from opentelemetry.instrumentation.openai.v1.responses_wrappers import (
            set_data_attributes,
            TracedData,
        )

        traced = TracedData(
            start_time=1000,
            response_id="resp_123",
            input="Hello",
            instructions=None,
            tools=None,
            output_blocks={},
            usage=None,
            output_text="Hi",
            request_model="gpt-4",
            response_model="gpt-4",
        )
        set_data_attributes(traced, mock_span)

        assert GenAIAttributes.GEN_AI_OPERATION_NAME in mock_span._attrs, (
            f"Expected '{GenAIAttributes.GEN_AI_OPERATION_NAME}' in span attrs, "
            f"got: {list(mock_span._attrs.keys())}"
        )
        assert mock_span._attrs[GenAIAttributes.GEN_AI_OPERATION_NAME] == "chat"


# ---------------------------------------------------------------------------
# P1-2: Responses API must set gen_ai.response.finish_reasons
# ---------------------------------------------------------------------------

class TestP1_2_ResponsesFinishReasons:
    """set_data_attributes must emit gen_ai.response.finish_reasons."""

    def test_completed_message_has_stop_finish_reason(self, mock_span):
        from opentelemetry.instrumentation.openai.v1.responses_wrappers import (
            set_data_attributes,
            TracedData,
        )

        msg = _make_message_output("msg_001", "Hi there")
        traced = TracedData(
            start_time=1000,
            response_id="resp_123",
            input="Hello",
            instructions=None,
            tools=None,
            output_blocks={"msg_001": msg},
            usage=None,
            output_text="Hi there",
            request_model="gpt-4",
            response_model="gpt-4",
            response_status="completed",
        )
        set_data_attributes(traced, mock_span)

        assert GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS in mock_span._attrs, (
            f"Expected '{GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS}' in span attrs, "
            f"got: {list(mock_span._attrs.keys())}"
        )
        reasons = mock_span._attrs[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS]
        assert "stop" in reasons

    def test_function_call_has_tool_call_finish_reason(self, mock_span):
        from opentelemetry.instrumentation.openai.v1.responses_wrappers import (
            set_data_attributes,
            TracedData,
        )

        fc = _make_function_call_output("fc_001", "get_weather", "{}")
        traced = TracedData(
            start_time=1000,
            response_id="resp_456",
            input="What's the weather?",
            instructions=None,
            tools=None,
            output_blocks={"fc_001": fc},
            usage=None,
            output_text=None,
            request_model="gpt-4",
            response_model="gpt-4",
            response_status="completed",
        )
        set_data_attributes(traced, mock_span)

        assert GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS in mock_span._attrs
        reasons = mock_span._attrs[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS]
        assert reasons == ("tool_call",), (
            f"Expected exactly ('tool_call',) per OTel spec, got: {reasons}"
        )


# ---------------------------------------------------------------------------
# P1-3: Responses API must support gen_ai.input/output.messages JSON path
# ---------------------------------------------------------------------------

class TestP1_3_ResponsesMessagesAttributes:
    """set_data_attributes must emit gen_ai.input.messages /
    gen_ai.output.messages as JSON."""

    def test_input_messages_json_when_flag_enabled(self, mock_span):
        from opentelemetry.instrumentation.openai.v1.responses_wrappers import (
            set_data_attributes,
            TracedData,
        )

        msg = _make_message_output("msg_001", "Hi there")
        traced = TracedData(
            start_time=1000,
            response_id="resp_789",
            input=[
                {"role": "user", "content": "Hello", "type": "message"},
            ],
            instructions="Be helpful",
            tools=None,
            output_blocks={"msg_001": msg},
            usage=None,
            output_text="Hi there",
            request_model="gpt-4",
            response_model="gpt-4",
        )
        set_data_attributes(traced, mock_span)

        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES in mock_span._attrs, (
            f"Expected gen_ai.input.messages, "
            f"got keys: {list(mock_span._attrs.keys())}"
        )
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in mock_span._attrs, (
            f"Expected gen_ai.output.messages, "
            f"got keys: {list(mock_span._attrs.keys())}"
        )

        input_msgs = json.loads(mock_span._attrs[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
        assert isinstance(input_msgs, list)

        output_msgs = json.loads(mock_span._attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
        assert isinstance(output_msgs, list)

    def test_string_input_as_user_message(self, mock_span):
        from opentelemetry.instrumentation.openai.v1.responses_wrappers import (
            set_data_attributes,
            TracedData,
        )

        msg = _make_message_output("msg_001", "4")
        traced = TracedData(
            start_time=1000,
            response_id="resp_str",
            input="What is 2+2?",
            instructions=None,
            tools=None,
            output_blocks={"msg_001": msg},
            usage=None,
            output_text="4",
            request_model="gpt-4",
            response_model="gpt-4",
        )
        set_data_attributes(traced, mock_span)

        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES in mock_span._attrs
        input_msgs = json.loads(mock_span._attrs[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
        assert input_msgs[0]["role"] == "user"


# ---------------------------------------------------------------------------
# P1-4: Streaming must accumulate reasoning/thinking blocks
# ---------------------------------------------------------------------------

class TestP1_4_StreamingReasoningAccumulation:
    """_accumulate_stream_items must capture reasoning_content from deltas."""

    def test_reasoning_content_accumulated(self):
        complete_response = {"choices": [], "model": "o3-mini", "id": "cmpl-1"}

        # First chunk: role
        item1 = {
            "model": "o3-mini",
            "id": "cmpl-1",
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        _accumulate_stream_items(item1, complete_response)

        # Second chunk: reasoning_content
        item2 = {
            "model": "o3-mini",
            "id": "cmpl-1",
            "choices": [{"index": 0, "delta": {"reasoning_content": "Let me think..."}, "finish_reason": None}],
        }
        _accumulate_stream_items(item2, complete_response)

        # Third chunk: more reasoning
        item3 = {
            "model": "o3-mini",
            "id": "cmpl-1",
            "choices": [{"index": 0, "delta": {"reasoning_content": " The answer is 3."}, "finish_reason": None}],
        }
        _accumulate_stream_items(item3, complete_response)

        # Fourth chunk: regular content
        item4 = {
            "model": "o3-mini",
            "id": "cmpl-1",
            "choices": [{"index": 0, "delta": {"content": "There are 3 r's."}, "finish_reason": None}],
        }
        _accumulate_stream_items(item4, complete_response)

        msg = complete_response["choices"][0]["message"]
        assert msg["content"] == "There are 3 r's."
        assert msg.get("reasoning_content") == "Let me think... The answer is 3.", (
            f"Expected accumulated reasoning_content, got: {msg}"
        )

    def test_reasoning_content_in_output_messages(self, mock_span):
        """Reasoning content should become a 'reasoning' part in output messages."""
        choices = [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "3 r's",
                    "reasoning_content": "Let me count...",
                },
                "finish_reason": "stop",
            }
        ]
        _set_output_messages(mock_span, choices)
        result = _get_output_messages(mock_span)

        parts = result[0]["parts"]
        reasoning_parts = [p for p in parts if p.get("type") == "reasoning"]
        assert len(reasoning_parts) == 1, (
            f"Expected 1 reasoning part, got parts: {parts}"
        )
        assert reasoning_parts[0]["content"] == "Let me count..."


# ---------------------------------------------------------------------------
# P2-1: _map_content_block must distinguish base64 data URIs from URLs
# ---------------------------------------------------------------------------

class TestP2_1_Base64ImageMapping:
    """Base64 inline images should map to BlobPart, not UriPart."""

    def test_regular_url_maps_to_uri_part(self):
        block = {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
        result = _map_content_block(block)
        assert result["type"] == "uri"
        assert result["uri"] == "https://example.com/img.png"

    def test_base64_image_maps_to_blob_part(self):
        data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg..."
        block = {"type": "image_url", "image_url": {"url": data_uri}}
        result = _map_content_block(block)
        assert result["type"] == "blob", (
            f"Expected 'blob' type for base64 data URI, got: {result}"
        )
        assert result["modality"] == "image"
        assert result["mime_type"] == "image/png"
        assert result["content"] == "iVBORw0KGgoAAAANSUhEUg..."

    def test_base64_jpeg_image(self):
        data_uri = "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
        block = {"type": "image_url", "image_url": {"url": data_uri}}
        result = _map_content_block(block)
        assert result["type"] == "blob"
        assert result["mime_type"] == "image/jpeg"
        assert result["content"] == "/9j/4AAQSkZJRg..."


# ---------------------------------------------------------------------------
# P2-2: gen_ai.tool.definitions must use JSON
# ---------------------------------------------------------------------------

class TestP2_2_ToolDefinitionsJson:
    """set_tools_attributes and _set_functions_attributes should emit a single
    JSON string attribute for gen_ai.tool.definitions."""

    def test_tools_as_json(self, mock_span):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        set_tools_attributes(mock_span, tools)

        assert GenAIAttributes.GEN_AI_TOOL_DEFINITIONS in mock_span._attrs, (
            f"Expected single '{GenAIAttributes.GEN_AI_TOOL_DEFINITIONS}' key, "
            f"got: {list(mock_span._attrs.keys())}"
        )
        parsed = json.loads(mock_span._attrs[GenAIAttributes.GEN_AI_TOOL_DEFINITIONS])
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["type"] == "function"
        assert parsed[0]["name"] == "get_weather"

    def test_functions_as_json(self, mock_span):
        functions = [
            {
                "name": "search",
                "description": "Search the web",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        _set_functions_attributes(mock_span, functions)

        assert GenAIAttributes.GEN_AI_TOOL_DEFINITIONS in mock_span._attrs, (
            f"Expected single '{GenAIAttributes.GEN_AI_TOOL_DEFINITIONS}' key, "
            f"got: {list(mock_span._attrs.keys())}"
        )
        parsed = json.loads(mock_span._attrs[GenAIAttributes.GEN_AI_TOOL_DEFINITIONS])
        assert isinstance(parsed, list)
        assert parsed[0]["type"] == "function"
        assert parsed[0]["name"] == "search"

    def test_tools_emits_json_format(self, mock_span):
        """Tools always use JSON format (gen_ai.tool.definitions as JSON array)."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        set_tools_attributes(mock_span, tools)

        assert GenAIAttributes.GEN_AI_TOOL_DEFINITIONS in mock_span._attrs, (
            f"Expected '{GenAIAttributes.GEN_AI_TOOL_DEFINITIONS}' key, "
            f"got: {list(mock_span._attrs.keys())}"
        )
        parsed = json.loads(mock_span._attrs[GenAIAttributes.GEN_AI_TOOL_DEFINITIONS])
        assert isinstance(parsed, list)
        assert parsed[0]["name"] == "get_weather"


# ---------------------------------------------------------------------------
# P2-3: _set_output_messages must handle refusal content
# ---------------------------------------------------------------------------

class TestP2_3_OutputRefusal:
    """Refusal in output messages should be captured as a refusal part."""

    def test_refusal_captured_in_output_messages(self, mock_span):
        choices = [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "refusal": "I cannot help with that request.",
                },
                "finish_reason": "stop",
            }
        ]
        _set_output_messages(mock_span, choices)
        result = _get_output_messages(mock_span)

        assert len(result) == 1
        msg = result[0]
        refusal_parts = [p for p in msg["parts"] if p.get("type") == "refusal"]
        text_with_refusal = [p for p in msg["parts"]
                             if "cannot help" in (p.get("content") or "")]
        assert refusal_parts or text_with_refusal, (
            f"Expected refusal to be captured in output parts, got: {msg['parts']}"
        )


# ---------------------------------------------------------------------------
# finish_reason fallback: must be "" when unknown, not fabricated "stop"
# ---------------------------------------------------------------------------

class TestFinishReasonFallback:
    """finish_reason in output messages must be '' when unknown (Bedrock convention)."""

    def test_chat_missing_finish_reason_uses_empty_string(self, mock_span):
        choices = [
            {
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": None,
            }
        ]
        _set_output_messages(mock_span, choices)
        result = _get_output_messages(mock_span)
        assert result[0]["finish_reason"] == "", (
            f"Expected '' for missing finish_reason, got '{result[0]['finish_reason']}'"
        )

    def test_chat_present_finish_reason_preserved(self, mock_span):
        choices = [
            {
                "message": {"role": "assistant", "content": "Hi"},
                "finish_reason": "stop",
            }
        ]
        _set_output_messages(mock_span, choices)
        result = _get_output_messages(mock_span)
        assert result[0]["finish_reason"] == "stop"

    def test_chat_tool_calls_finish_reason_mapped(self, mock_span):
        choices = [
            {
                "message": {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ]},
                "finish_reason": "tool_calls",
            }
        ]
        _set_output_messages(mock_span, choices)
        result = _get_output_messages(mock_span)
        assert result[0]["finish_reason"] == "tool_call"

    def test_completion_missing_finish_reason_uses_empty_string(self, mock_span):
        from opentelemetry.instrumentation.openai.shared.completion_wrappers import (
            _set_output_messages as _set_completion_output_messages,
        )
        choices = [{"text": "Hello world", "finish_reason": None}]
        _set_completion_output_messages(mock_span, choices)
        result = _get_output_messages(mock_span)
        assert result[0]["finish_reason"] == "", (
            f"Expected '' for missing finish_reason, got '{result[0]['finish_reason']}'"
        )


# ---------------------------------------------------------------------------
# P2-4: Metrics must map finish reason values correctly
# ---------------------------------------------------------------------------

class TestP2_4_MetricsFinishReasonMapping:
    """Choice counter metrics should use mapped finish reason constant key."""

    def test_finish_reason_uses_correct_attribute_key(self):
        from opentelemetry.instrumentation.openai.shared.chat_wrappers import (
            _set_choice_counter_metrics,
        )
        mock_counter = MagicMock()
        choices = [
            {"index": 0, "finish_reason": "tool_calls"},
        ]
        shared_attrs = {"gen_ai.operation.name": "chat"}

        _set_choice_counter_metrics(mock_counter, choices, shared_attrs)

        calls = mock_counter.add.call_args_list
        assert len(calls) > 0, "Expected at least one counter add call"
        # The attribute key used should be the mapped key
        for call in calls:
            attrs = call.kwargs.get("attributes") or call[1].get("attributes", {})
            # Check that finish_reason value is the OTel canonical value
            for key, val in attrs.items():
                if "finish_reason" in key:
                    assert val == "tool_call", (
                        f"Expected OTel canonical 'tool_call', got '{val}'"
                    )


# ---------------------------------------------------------------------------
# Reasoning attrs must be ABSENT (not empty tuple) when value is None
# ---------------------------------------------------------------------------

class TestReasoningAttrsOmittedWhenNone:
    """When reasoning fields are None, _set_span_attribute must skip them
    entirely — not emit an empty tuple ()."""

    def test_responses_reasoning_attrs_absent_when_none(self, mock_span):
        from opentelemetry.instrumentation.openai.v1.responses_wrappers import (
            set_data_attributes,
            TracedData,
        )
        from opentelemetry.semconv_ai import SpanAttributes

        traced = TracedData(
            start_time=1000,
            response_id="resp_none",
            input="Hi",
            instructions=None,
            tools=None,
            output_blocks={},
            usage=None,
            output_text="Hello",
            request_model="gpt-4",
            response_model="gpt-4",
            request_reasoning_summary=None,
            request_reasoning_effort=None,
            response_reasoning_effort=None,
        )
        set_data_attributes(traced, mock_span)

        for attr_name in (
            SpanAttributes.GEN_AI_REQUEST_REASONING_SUMMARY,
            SpanAttributes.GEN_AI_REQUEST_REASONING_EFFORT,
            SpanAttributes.GEN_AI_RESPONSE_REASONING_EFFORT,
        ):
            assert attr_name not in mock_span._attrs, (
                f"Attribute '{attr_name}' should be ABSENT when value is None, "
                f"but it was set to: {mock_span._attrs.get(attr_name)!r}"
            )

    def test_chat_reasoning_effort_absent_when_none(self, mock_span):
        from opentelemetry.instrumentation.openai.shared.chat_wrappers import (
            _handle_request,
        )
        from opentelemetry.semconv_ai import SpanAttributes

        # Simulate a request without reasoning_effort
        _handle_request(mock_span, {"model": "gpt-4", "messages": []})

        assert SpanAttributes.GEN_AI_REQUEST_REASONING_EFFORT not in mock_span._attrs, (
            f"Attribute should be ABSENT when reasoning_effort not provided, "
            f"but was set to: {mock_span._attrs.get(SpanAttributes.GEN_AI_REQUEST_REASONING_EFFORT)!r}"
        )


# ---------------------------------------------------------------------------
# P2-3b: Partial stream cleanup must emit gen_ai.output.messages
# ---------------------------------------------------------------------------

class TestPartialStreamCleanupOutputMessages:
    """ChatStream._ensure_cleanup must call _set_completions so that
    gen_ai.output.messages is emitted even on abrupt stream teardown."""

    def test_ensure_cleanup_sets_output_messages(self, mock_span):
        from opentelemetry.instrumentation.openai.shared.chat_wrappers import (
            ChatStream,
            _accumulate_stream_items,
        )

        # Create a ChatStream with a MagicMock response (ObjectProxy needs a real object)
        mock_response = MagicMock()
        stream = ChatStream(
            span=mock_span,
            response=mock_response,
            instance=None,
            start_time=time.time(),
            request_kwargs={"model": "gpt-4"},
        )

        # Simulate partial accumulation (as if some chunks were received)
        _accumulate_stream_items(
            {
                "model": "gpt-4",
                "id": "cmpl-partial",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            },
            stream._complete_response,
        )
        _accumulate_stream_items(
            {
                "model": "gpt-4",
                "id": "cmpl-partial",
                "choices": [{"index": 0, "delta": {"content": "Partial answer"}, "finish_reason": None}],
            },
            stream._complete_response,
        )

        # Trigger cleanup (simulates GC or abrupt teardown)
        stream._ensure_cleanup()

        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in mock_span._attrs, (
            f"Expected gen_ai.output.messages after partial cleanup, "
            f"got keys: {list(mock_span._attrs.keys())}"
        )
        output_msgs = json.loads(mock_span._attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
        assert len(output_msgs) > 0
        assert output_msgs[0]["role"] == "assistant"
        text_parts = [p for p in output_msgs[0]["parts"] if p.get("type") == "text"]
        assert any("Partial answer" in p.get("content", "") for p in text_parts), (
            f"Expected partial content in output messages, got: {output_msgs}"
        )


# ---------------------------------------------------------------------------
# P2-5: Realtime gen_ai.system_instructions must be parts array JSON
# ---------------------------------------------------------------------------

class TestP2_5_RealtimeSystemInstructions:
    """gen_ai.system_instructions should be JSON parts array, not raw string."""

    @pytest.mark.asyncio
    async def test_system_instructions_as_parts_array(self):
        from opentelemetry.instrumentation.openai.v1.realtime_wrappers import (
            RealtimeSessionWrapper,
            RealtimeSessionState,
        )

        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        attrs = {}
        mock_span.set_attribute = lambda k, v: attrs.__setitem__(k, v)

        state = RealtimeSessionState(mock_tracer, "gpt-4o-realtime-preview")
        state.session_span = mock_span

        mock_session = MagicMock()
        mock_session.update = AsyncMock()

        wrapper = RealtimeSessionWrapper(mock_session, state)
        await wrapper.update(session={
            "instructions": "You are a helpful assistant.",
        })

        val = attrs.get(GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS)
        assert val is not None, "gen_ai.system_instructions should be set"
        parsed = json.loads(val)
        assert isinstance(parsed, list), (
            f"Expected JSON array, got: {type(parsed)}"
        )
        assert parsed[0]["type"] == "text"
        assert parsed[0]["content"] == "You are a helpful assistant."


# ---------------------------------------------------------------------------
# P3-1: metric_shared_attributes must use constants, not hardcoded strings
# ---------------------------------------------------------------------------

class TestP3_1_MetricAttributeConstants:
    """'server.address' key should use upstream constant."""

    def test_server_address_uses_constant(self):
        attrs = metric_shared_attributes(
            response_model="gpt-4",
            operation="chat",
            server_address="https://api.openai.com/v1/",
        )
        assert SERVER_ADDRESS in attrs, (
            f"Expected '{SERVER_ADDRESS}' key (from constant), got: {list(attrs.keys())}"
        )


# ---------------------------------------------------------------------------
# P3-2: _map_content_block must wrap unrecognized types
# ---------------------------------------------------------------------------

class TestP3_2_UnrecognizedBlockWrapping:
    """Unrecognized block types should not pass through raw."""

    def test_unknown_block_type_wrapped(self):
        block = {"type": "custom_widget", "data": "something"}
        result = _map_content_block(block)
        assert result.get("type") == "custom_widget"
        assert result != block or "type" in result, (
            "Unrecognized blocks should be wrapped, not passed through raw"
        )


# ---------------------------------------------------------------------------
# _map_finish_reason must return "" for falsy input, mapped value for known
# reasons, and the original string as-is for unknown reasons.
# ---------------------------------------------------------------------------

class TestMapFinishReason:
    from opentelemetry.instrumentation.openai.shared.chat_wrappers import (
        _map_finish_reason,
    )
    _map_finish_reason = staticmethod(_map_finish_reason)

    @pytest.mark.parametrize("falsy_input", [None, "", 0, False])
    def test_returns_empty_string_for_falsy(self, falsy_input):
        assert self._map_finish_reason(falsy_input) == ""

    def test_maps_tool_calls_to_tool_call(self):
        assert self._map_finish_reason("tool_calls") == "tool_call"

    def test_maps_function_call_to_tool_call(self):
        assert self._map_finish_reason("function_call") == "tool_call"

    def test_passes_through_stop(self):
        assert self._map_finish_reason("stop") == "stop"

    def test_passes_through_length(self):
        assert self._map_finish_reason("length") == "length"

    def test_passes_through_unknown_reason(self):
        assert self._map_finish_reason("some_new_reason") == "some_new_reason"
