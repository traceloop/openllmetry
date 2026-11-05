"""Tests for the async input-processing path (set_input_attributes, _handle_request_async)."""

import asyncio
import json
from unittest.mock import MagicMock

import pytest
from opentelemetry.instrumentation.google_generativeai import span_utils as su
from opentelemetry.instrumentation.google_generativeai.utils import dont_throw
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


def _recorded_input_messages(span_mock):
    for call in span_mock.set_attribute.call_args_list:
        args = call[0]
        if args[0] == GenAIAttributes.GEN_AI_INPUT_MESSAGES:
            return json.loads(args[1])
    return None


# ===========================================================================
# 1. dont_throw — async support
# ===========================================================================


class TestDontThrowAsync:
    def test_async_function_is_wrapped_as_coroutine(self):
        @dont_throw
        async def my_async():
            return 42

        assert asyncio.iscoroutinefunction(my_async)

    def test_async_wrapper_returns_value(self):
        @dont_throw
        async def my_async():
            return 42

        result = asyncio.get_event_loop().run_until_complete(my_async())
        assert result == 42

    def test_async_wrapper_swallows_exception(self):
        @dont_throw
        async def boom():
            raise ValueError("oops")

        result = asyncio.get_event_loop().run_until_complete(boom())
        assert result is None

    def test_sync_function_stays_sync(self):
        @dont_throw
        def my_sync():
            return 7

        assert not asyncio.iscoroutinefunction(my_sync)
        assert my_sync() == 7


# ===========================================================================
# 2. set_input_attributes (async) — basic text content
# ===========================================================================


class TestSetInputAttributesAsync:
    def _make_span(self):
        span = MagicMock()
        span.is_recording.return_value = True
        return span

    @pytest.mark.asyncio
    async def test_string_contents(self, monkeypatch):
        monkeypatch.setattr(su, "should_send_prompts", lambda: True)
        span = self._make_span()

        await su.set_input_attributes(span, (), {"contents": "Hello"}, "gemini-pro")

        messages = _recorded_input_messages(span)
        assert messages is not None
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["parts"][0] == {"type": "text", "content": "Hello"}

    @pytest.mark.asyncio
    async def test_list_of_strings(self, monkeypatch):
        monkeypatch.setattr(su, "should_send_prompts", lambda: True)
        span = self._make_span()

        await su.set_input_attributes(
            span, (), {"contents": ["first", "second"]}, "gemini-pro"
        )

        # A bare list of strings is one user turn with several parts, which is
        # how google.genai's t_contents sends it.
        messages = _recorded_input_messages(span)
        assert messages == [
            {
                "role": "user",
                "parts": [
                    {"type": "text", "content": "first"},
                    {"type": "text", "content": "second"},
                ],
            }
        ]

    @pytest.mark.asyncio
    async def test_single_content_object(self, monkeypatch):
        """A Content passed directly (not in a list) is one turn."""
        from google.genai import types

        monkeypatch.setattr(su, "should_send_prompts", lambda: True)
        span = self._make_span()

        content = types.Content(role="user", parts=[types.Part(text="Hi there")])
        await su.set_input_attributes(span, (), {"contents": content}, "gemini-pro")

        messages = _recorded_input_messages(span)
        assert messages == [
            {"role": "user", "parts": [{"type": "text", "content": "Hi there"}]}
        ]

    @pytest.mark.asyncio
    async def test_bare_parts_grouped_between_contents(self, monkeypatch):
        """Bare parts are grouped into turns; function calls form a model turn."""
        from google.genai import types

        monkeypatch.setattr(su, "should_send_prompts", lambda: True)
        span = self._make_span()

        contents = [
            "What is the weather?",
            types.Part(text="In Paris."),
            types.Part(
                function_call=types.FunctionCall(
                    id="c1", name="get_weather", args={"city": "Paris"}
                )
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            id="c1", name="get_weather", response={"temp": 20}
                        )
                    )
                ],
            ),
        ]
        await su.set_input_attributes(span, (), {"contents": contents}, "gemini-pro")

        messages = _recorded_input_messages(span)
        assert [m["role"] for m in messages] == ["user", "assistant", "user"]
        assert [p["content"] for p in messages[0]["parts"]] == [
            "What is the weather?",
            "In Paris.",
        ]
        assert messages[1]["parts"][0]["type"] == "tool_call"
        assert messages[1]["parts"][0]["name"] == "get_weather"
        assert messages[2]["parts"][0]["type"] == "tool_call_response"

    @pytest.mark.asyncio
    async def test_prompt_kwarg(self, monkeypatch):
        monkeypatch.setattr(su, "should_send_prompts", lambda: True)
        span = self._make_span()

        await su.set_input_attributes(span, (), {"prompt": "Summarize"}, "gemini-pro")

        messages = _recorded_input_messages(span)
        assert messages is not None
        assert len(messages) == 1
        assert messages[0]["parts"][0]["content"] == "Summarize"

    @pytest.mark.asyncio
    async def test_positional_args(self, monkeypatch):
        monkeypatch.setattr(su, "should_send_prompts", lambda: True)
        span = self._make_span()

        await su.set_input_attributes(span, ("Tell me a joke",), {}, "gemini-pro")

        messages = _recorded_input_messages(span)
        assert messages is not None
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_skipped_when_prompts_disabled(self, monkeypatch):
        monkeypatch.setattr(su, "should_send_prompts", lambda: False)
        span = self._make_span()

        await su.set_input_attributes(span, (), {"contents": "Hello"}, "gemini-pro")

        messages = _recorded_input_messages(span)
        assert messages is None

    @pytest.mark.asyncio
    async def test_content_object_with_role(self, monkeypatch):
        """Content objects with .role and .parts should be processed."""
        monkeypatch.setattr(su, "should_send_prompts", lambda: True)
        span = self._make_span()

        part = MagicMock()
        part.text = "Hello from model"
        part.thought = None
        part.function_call = None
        part.function_response = None
        part.inline_data = None
        part.executable_code = None
        part.code_execution_result = None

        content = MagicMock()
        content.role = "model"
        content.parts = [part]

        await su.set_input_attributes(
            span, (), {"contents": [content]}, "gemini-pro"
        )

        messages = _recorded_input_messages(span)
        assert messages is not None
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["parts"][0]["content"] == "Hello from model"


# ===========================================================================
# 3. Async produces identical format to sync
# ===========================================================================


class TestAsyncSyncParity:
    @pytest.mark.asyncio
    async def test_string_contents_match(self, monkeypatch):
        monkeypatch.setattr(su, "should_send_prompts", lambda: True)

        span_sync = MagicMock()
        span_sync.is_recording.return_value = True
        su.set_input_attributes_sync(
            span_sync, (), {"contents": "Hello world"}, "gemini-pro"
        )

        span_async = MagicMock()
        span_async.is_recording.return_value = True
        await su.set_input_attributes(
            span_async, (), {"contents": "Hello world"}, "gemini-pro"
        )

        sync_messages = _recorded_input_messages(span_sync)
        async_messages = _recorded_input_messages(span_async)
        assert sync_messages == async_messages

    @pytest.mark.asyncio
    async def test_list_contents_match(self, monkeypatch):
        monkeypatch.setattr(su, "should_send_prompts", lambda: True)

        span_sync = MagicMock()
        span_sync.is_recording.return_value = True
        su.set_input_attributes_sync(
            span_sync, (), {"contents": ["a", "b"]}, "gemini-pro"
        )

        span_async = MagicMock()
        span_async.is_recording.return_value = True
        await su.set_input_attributes(
            span_async, (), {"contents": ["a", "b"]}, "gemini-pro"
        )

        sync_messages = _recorded_input_messages(span_sync)
        async_messages = _recorded_input_messages(span_async)
        assert sync_messages == async_messages

    @pytest.mark.asyncio
    async def test_single_content_object_match(self, monkeypatch):
        from google.genai import types

        monkeypatch.setattr(su, "should_send_prompts", lambda: True)
        contents = types.Content(role="user", parts=[types.Part(text="Hi")])

        span_sync = MagicMock()
        span_sync.is_recording.return_value = True
        su.set_input_attributes_sync(span_sync, (), {"contents": contents}, "gemini-pro")

        span_async = MagicMock()
        span_async.is_recording.return_value = True
        await su.set_input_attributes(
            span_async, (), {"contents": contents}, "gemini-pro"
        )

        sync_messages = _recorded_input_messages(span_sync)
        assert sync_messages is not None
        assert sync_messages == _recorded_input_messages(span_async)

    @pytest.mark.asyncio
    async def test_prompt_kwarg_match(self, monkeypatch):
        monkeypatch.setattr(su, "should_send_prompts", lambda: True)

        span_sync = MagicMock()
        span_sync.is_recording.return_value = True
        su.set_input_attributes_sync(
            span_sync, (), {"prompt": "Explain quantum physics"}, "gemini-pro"
        )

        span_async = MagicMock()
        span_async.is_recording.return_value = True
        await su.set_input_attributes(
            span_async, (), {"prompt": "Explain quantum physics"}, "gemini-pro"
        )

        sync_messages = _recorded_input_messages(span_sync)
        async_messages = _recorded_input_messages(span_async)
        assert sync_messages == async_messages


# ===========================================================================
# 4. _handle_request_async wiring
# ===========================================================================


class TestHandleRequestAsync:
    @pytest.mark.asyncio
    async def test_calls_set_input_attributes_async(self, monkeypatch):
        from opentelemetry.instrumentation.google_generativeai import (
            _handle_request_async,
        )

        monkeypatch.setattr(su, "should_send_prompts", lambda: True)

        span = MagicMock()
        span.is_recording.return_value = True

        await _handle_request_async(
            span, (), {"contents": "test"}, "gemini-pro", None
        )

        messages = _recorded_input_messages(span)
        assert messages is not None
        assert messages[0]["parts"][0]["content"] == "test"

    @pytest.mark.asyncio
    async def test_sets_model_request_attributes(self, monkeypatch):
        from opentelemetry.instrumentation.google_generativeai import (
            _handle_request_async,
        )

        monkeypatch.setattr(su, "should_send_prompts", lambda: True)

        span = MagicMock()
        span.is_recording.return_value = True

        await _handle_request_async(
            span, (), {"contents": "x", "temperature": 0.5}, "gemini-pro", None
        )

        temp_calls = [
            c[0]
            for c in span.set_attribute.call_args_list
            if c[0][0] == GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE
        ]
        assert len(temp_calls) == 1
        assert temp_calls[0][1] == 0.5
