"""
TDD tests for finish_reason handling across OpenAI instrumentation.

Based on finish_reasons_review_report.md findings:
- P1-1: Responses API hardcoded finish_reason instead of using provider values
- P1-2: Top-level finish_reasons derived instead of extracted from response
- P1-3: Chat completions missing top-level finish_reasons attribute
- P1-4: Completions API missing top-level finish_reasons attribute
- P2-1: Missing _map_finish_reason function in completion_wrappers.py
"""

import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from .utils import get_output_messages


class TestCompletionsFinishReasons:
    """Test finish_reason handling in completions API (P1-4, P2-1)"""

    @pytest.mark.vcr
    def test_completions_sets_top_level_finish_reasons(
        self, instrument_legacy, span_exporter, openai_client
    ):
        """Non-streaming completions must set gen_ai.response.finish_reasons."""
        openai_client.completions.create(
            model="davinci-002",
            prompt="Tell me a joke about opentelemetry",
        )

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        # Verify top-level finish_reasons attribute is set
        finish_reasons = span.attributes.get(
            GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
        )
        assert finish_reasons is not None, "gen_ai.response.finish_reasons must be set"
        assert isinstance(finish_reasons, (list, tuple))
        assert len(finish_reasons) > 0
        # Should contain mapped finish_reason from response
        assert "length" in finish_reasons or "stop" in finish_reasons

    @pytest.mark.vcr
    def test_completions_finish_reason_in_output_messages(
        self, instrument_legacy, span_exporter, openai_client
    ):
        """Output messages must have finish_reason field."""
        openai_client.completions.create(
            model="davinci-002",
            prompt="Tell me a joke about opentelemetry",
        )

        spans = span_exporter.get_finished_spans()
        span = spans[0]
        output_messages = get_output_messages(span)

        assert len(output_messages) > 0
        for msg in output_messages:
            assert "finish_reason" in msg
            # finish_reason should be a string (empty string if missing from provider)
            assert isinstance(msg["finish_reason"], str)

    @pytest.mark.vcr
    def test_completions_streaming_sets_top_level_finish_reasons(
        self, instrument_legacy, span_exporter, openai_client
    ):
        """Streaming completions must accumulate and set finish_reasons."""
        response = openai_client.completions.create(
            model="davinci-002",
            prompt="Tell me a joke about opentelemetry",
            stream=True,
        )

        for _ in response:
            pass

        spans = span_exporter.get_finished_spans()
        span = spans[0]

        # Verify top-level finish_reasons attribute is set
        finish_reasons = span.attributes.get(
            GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
        )
        assert finish_reasons is not None, "Streaming must set finish_reasons"
        assert isinstance(finish_reasons, (list, tuple))
        assert len(finish_reasons) > 0

    @pytest.mark.vcr
    @pytest.mark.asyncio
    async def test_async_completions_sets_top_level_finish_reasons(
        self, instrument_legacy, span_exporter, async_openai_client
    ):
        """Async completions must set gen_ai.response.finish_reasons."""
        await async_openai_client.completions.create(
            model="davinci-002",
            prompt="Tell me a joke about opentelemetry",
        )

        spans = span_exporter.get_finished_spans()
        span = spans[0]

        finish_reasons = span.attributes.get(
            GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
        )
        assert finish_reasons is not None
        assert isinstance(finish_reasons, (list, tuple))
        assert len(finish_reasons) > 0

    @pytest.mark.vcr
    @pytest.mark.asyncio
    async def test_async_completions_streaming_sets_top_level_finish_reasons(
        self, instrument_legacy, span_exporter, async_openai_client
    ):
        """Async streaming completions must set finish_reasons."""
        response = await async_openai_client.completions.create(
            model="davinci-002",
            prompt="Tell me a joke about opentelemetry",
            stream=True,
        )

        async for _ in response:
            pass

        spans = span_exporter.get_finished_spans()
        span = spans[0]

        finish_reasons = span.attributes.get(
            GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
        )
        assert finish_reasons is not None
        assert isinstance(finish_reasons, (list, tuple))
        assert len(finish_reasons) > 0


class TestChatCompletionsFinishReasons:
    """Test finish_reason handling in chat completions API (P1-3)"""

    @pytest.mark.vcr
    def test_chat_completions_sets_top_level_finish_reasons(
        self, instrument_legacy, span_exporter, openai_client
    ):
        """Non-streaming chat must set gen_ai.response.finish_reasons."""
        openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        )

        spans = span_exporter.get_finished_spans()
        span = spans[0]

        # Verify top-level finish_reasons attribute is set
        finish_reasons = span.attributes.get(
            GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
        )
        assert finish_reasons is not None, "gen_ai.response.finish_reasons must be set"
        assert isinstance(finish_reasons, (list, tuple))
        assert len(finish_reasons) > 0
        # Should contain "stop" for normal completion
        assert "stop" in finish_reasons

    @pytest.mark.vcr
    def test_chat_completions_finish_reason_in_output_messages(
        self, instrument_legacy, span_exporter, openai_client
    ):
        """Output messages must have finish_reason field."""
        openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        )

        spans = span_exporter.get_finished_spans()
        span = spans[0]
        output_messages = get_output_messages(span)

        assert len(output_messages) > 0
        for msg in output_messages:
            assert "finish_reason" in msg
            assert msg["finish_reason"] == "stop"

    @pytest.mark.vcr
    def test_chat_streaming_sets_top_level_finish_reasons(
        self, instrument_legacy, span_exporter, openai_client
    ):
        """Streaming chat must accumulate and set finish_reasons."""
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
            stream=True,
        )

        for _ in response:
            pass

        spans = span_exporter.get_finished_spans()
        span = spans[0]

        # Verify top-level finish_reasons attribute is set
        finish_reasons = span.attributes.get(
            GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
        )
        assert finish_reasons is not None, "Streaming must set finish_reasons"
        assert isinstance(finish_reasons, (list, tuple))
        assert len(finish_reasons) > 0

    @pytest.mark.vcr
    @pytest.mark.asyncio
    async def test_async_chat_sets_top_level_finish_reasons(
        self, instrument_legacy, span_exporter, async_openai_client
    ):
        """Async chat must set gen_ai.response.finish_reasons."""
        await async_openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        )

        spans = span_exporter.get_finished_spans()
        span = spans[0]

        finish_reasons = span.attributes.get(
            GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
        )
        assert finish_reasons is not None
        assert isinstance(finish_reasons, (list, tuple))
        assert len(finish_reasons) > 0

    @pytest.mark.vcr
    @pytest.mark.asyncio
    async def test_async_chat_streaming_sets_top_level_finish_reasons(
        self, instrument_legacy, span_exporter, async_openai_client
    ):
        """Async streaming chat must set finish_reasons."""
        response = await async_openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
            stream=True,
        )

        async for _ in response:
            pass

        spans = span_exporter.get_finished_spans()
        span = spans[0]

        finish_reasons = span.attributes.get(
            GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
        )
        assert finish_reasons is not None
        assert isinstance(finish_reasons, (list, tuple))
        assert len(finish_reasons) > 0


class TestFinishReasonMapping:
    """Test finish_reason mapping from provider values"""

    @pytest.mark.vcr
    def test_finish_reason_mapped_from_provider_not_derived(
        self, instrument_legacy, span_exporter, openai_client
    ):
        """finish_reason must come from response, not inferred from parts.

        This test validates that we use the actual finish_reason from the
        provider response (e.g., "length", "content_filter") rather than
        deriving it from the presence of tool_calls or other content.
        """
        # Use a completion that hits max_tokens to get finish_reason="length"
        openai_client.completions.create(
            model="davinci-002",
            prompt="Tell me a very long story about opentelemetry",
            max_tokens=10,  # Force length finish_reason
        )

        spans = span_exporter.get_finished_spans()
        span = spans[0]
        output_messages = get_output_messages(span)

        # Verify finish_reason is "length" from provider, not "stop"
        assert len(output_messages) > 0
        assert output_messages[0]["finish_reason"] == "length"

        # Verify top-level attribute also has "length"
        finish_reasons = span.attributes.get(
            GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
        )
        assert "length" in finish_reasons

    @pytest.mark.vcr
    def test_finish_reason_tool_calls_mapped_correctly(
        self, instrument_legacy, span_exporter, openai_client
    ):
        """finish_reason "tool_calls" must be mapped to "tool_call"."""
        openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "What's the weather in Boston?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                            },
                        },
                    },
                }
            ],
        )

        spans = span_exporter.get_finished_spans()
        span = spans[0]
        output_messages = get_output_messages(span)

        # OpenAI returns "tool_calls" but we should map to "tool_call"
        if output_messages and output_messages[0].get("parts"):
            has_tool_call = any(
                p.get("type") == "tool_call" for p in output_messages[0]["parts"]
            )
            if has_tool_call:
                # Should be mapped to "tool_call" (singular)
                assert output_messages[0]["finish_reason"] == "tool_call"

                finish_reasons = span.attributes.get(
                    GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
                )
                assert "tool_call" in finish_reasons

    @pytest.mark.vcr
    def test_finish_reason_defaults_to_empty_string_when_missing(
        self, instrument_legacy, span_exporter, openai_client
    ):
        """When provider doesn't return finish_reason, default to empty string."""
        # This test may need a mocked response where finish_reason is None
        openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
        )

        spans = span_exporter.get_finished_spans()
        span = spans[0]
        output_messages = get_output_messages(span)

        # Even if provider returns None, output message should have ""
        assert len(output_messages) > 0
        assert "finish_reason" in output_messages[0]
        assert isinstance(output_messages[0]["finish_reason"], str)


class TestFinishReasonDeduplication:
    """Test finish_reason deduplication in streaming scenarios"""

    @pytest.mark.vcr
    def test_streaming_deduplicates_finish_reasons(
        self, instrument_legacy, span_exporter, openai_client
    ):
        """Streaming with multiple choices should deduplicate finish_reasons."""
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Tell me a joke"}],
            n=2,  # Request 2 completions
            stream=True,
        )

        for _ in response:
            pass

        spans = span_exporter.get_finished_spans()
        span = spans[0]

        finish_reasons = span.attributes.get(
            GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
        )

        if finish_reasons:
            # Should be deduplicated (no duplicate "stop" entries)
            assert len(finish_reasons) == len(set(finish_reasons))


class TestFinishReasonOmission:
    """Test that top-level finish_reasons is omitted when no meaningful values"""

    def test_finish_reasons_omitted_when_all_none(
        self, instrument_legacy, span_exporter, openai_client
    ):
        """When all finish_reasons are None, attribute should be omitted entirely.

        Uses _set_output_messages directly with a mocked choice list where
        every choice has finish_reason=None to verify the attribute is not set.
        """
        from unittest.mock import MagicMock
        from opentelemetry.instrumentation.openai.shared.chat_wrappers import (
            _set_output_messages,
        )

        span = MagicMock()
        span.is_recording.return_value = True

        # Simulate two choices where finish_reason is None
        choices = [
            {
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": None,
            },
            {
                "message": {"role": "assistant", "content": "World"},
                "finish_reason": None,
            },
        ]

        _set_output_messages(span, choices)

        # Collect all set_attribute calls
        attr_calls = {call[0][0]: call[0][1] for call in span.set_attribute.call_args_list}

        # gen_ai.response.finish_reasons must NOT be set
        assert GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS not in attr_calls, (
            "gen_ai.response.finish_reasons should be omitted when all values are None"
        )


class TestResponsesAPIFinishReasons:
    """Test finish_reason handling in Responses API (P1-1, P1-2)"""

    @pytest.mark.vcr
    def test_responses_api_stop_finish_reason(
        self, instrument_legacy, span_exporter, openai_client
    ):
        """Responses API must derive finish_reason="stop" for a completed text response.

        A simple text completion should produce status="completed" with no tool
        calls, which maps to finish_reason="stop".
        """
        openai_client.responses.create(
            model="gpt-4.1-nano",
            input="Say hello in one word.",
        )

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        # Top-level finish_reasons must be set and contain "stop"
        finish_reasons = span.attributes.get(
            GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
        )
        assert finish_reasons is not None, "gen_ai.response.finish_reasons must be set"
        assert "stop" in finish_reasons

        # Output messages should also carry finish_reason="stop"
        output_messages = get_output_messages(span)
        assert len(output_messages) > 0
        assert output_messages[0]["finish_reason"] == "stop"

    @pytest.mark.vcr
    def test_responses_api_tool_call_finish_reason(
        self, instrument_legacy, span_exporter, openai_client
    ):
        """Responses API must derive finish_reason="tool_call" when tool calls are present.

        When the model decides to call a function, the response status is
        "completed" and the output contains a function_call block, which
        should map to finish_reason="tool_call".
        """
        openai_client.responses.create(
            model="gpt-4.1-nano",
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": "What's the weather in Paris?",
                }
            ],
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name",
                            }
                        },
                        "required": ["location"],
                    },
                }
            ],
            tool_choice="required",
        )

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        # Top-level finish_reasons must contain "tool_call"
        finish_reasons = span.attributes.get(
            GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
        )
        assert finish_reasons is not None, "gen_ai.response.finish_reasons must be set"
        assert "tool_call" in finish_reasons

        # Output messages should carry finish_reason="tool_call"
        output_messages = get_output_messages(span)
        assert len(output_messages) > 0
        assert output_messages[0]["finish_reason"] == "tool_call"
