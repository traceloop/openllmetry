from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from litellm.types.utils import ModelResponse, TextCompletionResponse
from opentelemetry import context as context_api
from opentelemetry.instrumentation.fortifyroot import (
    SafetyFinding,
    SafetyLocation,
    SafetyResult,
    clear_safety_handlers,
    register_completion_safety_handler,
    register_prompt_safety_handler,
)
from opentelemetry.instrumentation.litellm import (
    LiteLLMInstrumentor,
    _WRAPPED_METHODS,
    _invoke_acompletion,
    _invoke_completion,
)
from opentelemetry.instrumentation.litellm.safety import (
    apply_completion_safety,
    apply_prompt_safety,
    extract_prompt_texts,
    extract_text_content,
    _get_messages,
    _resolve_masked_text,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)

pytestmark = pytest.mark.safety


def setup_function():
    clear_safety_handlers()


def teardown_function():
    clear_safety_handlers()


def _test_tracer():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)
    return exporter, tracer


def _prompt_result(masked_text, context):
    return SafetyResult(
        text=masked_text,
        overall_action="MASK",
        findings=[
            SafetyFinding(
                category="PII",
                severity="HIGH",
                action="MASK",
                rule_name="PII.secret",
                start=0,
                end=len(context.text),
            )
        ],
    )


def _completion_result(masked_text, context):
    return SafetyResult(
        text=masked_text,
        overall_action="MASK",
        findings=[
            SafetyFinding(
                category="SECRET",
                severity="HIGH",
                action="MASK",
                rule_name="SECRET.token",
                start=0,
                end=len(context.text),
            )
        ],
    )


def test_sync_completion_masks_prompt_response_and_sets_span_attributes():
    exporter, tracer = _test_tracer()
    register_prompt_safety_handler(
        lambda context: _prompt_result("[PII.email]", context)
        if context.location == SafetyLocation.PROMPT and context.text == "secret"
        else None
    )
    register_completion_safety_handler(
        lambda context: _completion_result("[SECRET.token]", context)
        if context.location == SafetyLocation.COMPLETION and context.text == "token-123"
        else None
    )

    messages = [{"role": "user", "content": "secret"}]

    def wrapped(*args, **kwargs):
        assert kwargs["messages"][0]["content"] == "[PII.email]"
        assert context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY) is True
        return ModelResponse(
            model="gpt-4o-mini",
            usage={"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
            choices=[
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "token-123"},
                }
            ],
        )

    response = _invoke_completion(
        tracer,
        wrapped,
        (),
        {"model": "gpt-4o-mini", "messages": messages},
    )

    assert messages[0]["content"] == "secret"
    assert response.choices[0].message.content == "[SECRET.token]"

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "litellm.completion"
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-4o-mini"
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "gpt-4o-mini"
    assert span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "[PII.email]"
    assert span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "[SECRET.token]"
    assert span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 8
    assert len(span.events) == 2


def test_sync_text_completion_masks_text_choices():
    exporter, tracer = _test_tracer()
    register_prompt_safety_handler(
        lambda context: _prompt_result("[PII.email]", context)
        if context.location == SafetyLocation.PROMPT and context.text == "secret"
        else None
    )
    register_completion_safety_handler(
        lambda context: _completion_result("[SECRET.token]", context)
        if context.location == SafetyLocation.COMPLETION and context.text == "token-123"
        else None
    )

    def wrapped(*args, **kwargs):
        assert args[0] == "[PII.email]"
        assert kwargs["model"] == "gpt-3.5-turbo-instruct"
        assert context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY) is True
        return TextCompletionResponse(
            model="gpt-3.5-turbo-instruct",
            choices=[{"text": "token-123"}],
        )

    response = _invoke_completion(
        tracer,
        wrapped,
        ("secret",),
        {"model": "gpt-3.5-turbo-instruct"},
        is_text_completion=True,
    )

    assert response.choices[0].text == "[SECRET.token]"

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "litellm.text_completion"
    assert span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-3.5-turbo-instruct"
    assert span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "[PII.email]"
    assert span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "[SECRET.token]"


@pytest.mark.asyncio
async def test_async_completion_masks_prompt_and_response():
    exporter, tracer = _test_tracer()
    register_prompt_safety_handler(
        lambda context: _prompt_result("[PII.email]", context)
        if context.location == SafetyLocation.PROMPT and context.text == "secret"
        else None
    )
    register_completion_safety_handler(
        lambda context: _completion_result("[SECRET.token]", context)
        if context.location == SafetyLocation.COMPLETION and context.text == "token-123"
        else None
    )

    async def wrapped(*args, **kwargs):
        assert kwargs["messages"][0]["content"] == "[PII.email]"
        return ModelResponse(
            model="gpt-4o-mini",
            choices=[
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "token-123"},
                }
            ],
        )

    response = await _invoke_acompletion(
        tracer,
        wrapped,
        (),
        {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "secret"}]},
    )

    assert response.choices[0].message.content == "[SECRET.token]"
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "[SECRET.token]"


@pytest.mark.asyncio
async def test_async_text_completion_masks_prompt_and_response():
    exporter, tracer = _test_tracer()
    register_prompt_safety_handler(
        lambda context: _prompt_result("[PII.email]", context)
        if context.location == SafetyLocation.PROMPT and context.text == "secret"
        else None
    )
    register_completion_safety_handler(
        lambda context: _completion_result("[SECRET.token]", context)
        if context.location == SafetyLocation.COMPLETION and context.text == "token-123"
        else None
    )

    async def wrapped(*args, **kwargs):
        assert args[0] == "[PII.email]"
        assert kwargs["model"] == "gpt-3.5-turbo-instruct"
        return TextCompletionResponse(
            model="gpt-3.5-turbo-instruct",
            choices=[{"text": "token-123"}],
        )

    response = await _invoke_acompletion(
        tracer,
        wrapped,
        ("secret",),
        {"model": "gpt-3.5-turbo-instruct"},
        is_text_completion=True,
    )

    assert response.choices[0].text == "[SECRET.token]"
    span = exporter.get_finished_spans()[0]
    assert span.name == "litellm.text_completion"
    assert span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "[PII.email]"
    assert span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "[SECRET.token]"


@pytest.mark.asyncio
async def test_sync_wrapper_handles_awaitable_response():
    exporter, tracer = _test_tracer()
    register_prompt_safety_handler(
        lambda context: _prompt_result("[PII.email]", context)
        if context.location == SafetyLocation.PROMPT and context.text == "secret"
        else None
    )
    register_completion_safety_handler(
        lambda context: _completion_result("[SECRET.token]", context)
        if context.location == SafetyLocation.COMPLETION and context.text == "token-123"
        else None
    )

    async def response_coro():
        assert context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY) is True
        return ModelResponse(
            model="gpt-4o-mini",
            choices=[
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "token-123"},
                }
            ],
        )

    def wrapped(*args, **kwargs):
        assert kwargs["messages"][0]["content"] == "[PII.email]"
        return response_coro()

    response = _invoke_completion(
        tracer,
        wrapped,
        (),
        {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "secret"}]},
    )

    assert inspect.iscoroutine(response)
    response = await response

    assert response.choices[0].message.content == "[SECRET.token]"
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "[SECRET.token]"


def test_streaming_completion_is_passthrough():
    exporter, tracer = _test_tracer()
    sentinel = SimpleNamespace(value="stream")
    calls = []

    def wrapped(*args, **kwargs):
        calls.append(kwargs)
        return sentinel

    response = _invoke_completion(
        tracer,
        wrapped,
        (),
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "secret"}],
            "stream": True,
        },
    )

    assert response is sentinel
    assert calls[0]["messages"][0]["content"] == "secret"
    assert not exporter.get_finished_spans()


def test_instrumentor_wraps_and_unwraps_all_methods():
    instrumentor = LiteLLMInstrumentor()
    unwrap_side_effect = [None, RuntimeError("boom")] + [None] * (len(_WRAPPED_METHODS) - 2)

    with patch("opentelemetry.instrumentation.litellm.wrap_function_wrapper") as wrap_mock, patch(
        "opentelemetry.instrumentation.litellm.unwrap",
        side_effect=unwrap_side_effect,
    ) as unwrap_mock, patch("opentelemetry.instrumentation.litellm.logger.debug") as debug_mock:
        instrumentor._instrument()
        instrumentor._uninstrument()

    assert instrumentor.instrumentation_dependencies() == ("litellm >= 1.71.2, < 2",)
    assert wrap_mock.call_count == len(_WRAPPED_METHODS)
    assert unwrap_mock.call_count == len(_WRAPPED_METHODS)
    debug_mock.assert_called_once()


def test_error_paths_record_error_status_and_request_attributes():
    exporter, tracer = _test_tracer()

    def wrapped(*args, **kwargs):
        raise ValueError("sync boom")

    with pytest.raises(ValueError, match="sync boom"):
        _invoke_completion(
            tracer,
            wrapped,
            ("gpt-4o-mini",),
            {"user": "alice", "custom_llm_provider": "openai"},
        )

    span = exporter.get_finished_spans()[0]
    assert span.status.status_code.name == "ERROR"
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-4o-mini"
    assert span.attributes[SpanAttributes.LLM_USER] == "alice"
    assert span.attributes["litellm.request.provider"] == "openai"


@pytest.mark.asyncio
async def test_async_skip_and_error_paths():
    exporter, tracer = _test_tracer()
    sentinel = SimpleNamespace(value="stream")

    async def skipped(*args, **kwargs):
        return sentinel

    response = await _invoke_acompletion(
        tracer,
        skipped,
        (),
        {"model": "gpt-4o-mini", "stream": True},
    )
    assert response is sentinel
    assert not exporter.get_finished_spans()

    async def raising(*args, **kwargs):
        raise RuntimeError("async boom")

    with pytest.raises(RuntimeError, match="async boom"):
        await _invoke_acompletion(
            tracer,
            raising,
            (),
            {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "secret"}]},
        )

    error_span = exporter.get_finished_spans()[0]
    assert error_span.status.status_code.name == "ERROR"


def test_safety_helpers_cover_args_blocks_and_text_extraction():
    register_prompt_safety_handler(
        lambda context: _prompt_result(f"[MASKED:{context.text}]", context)
        if context.location == SafetyLocation.PROMPT
        else None
    )
    register_completion_safety_handler(
        lambda context: _completion_result(f"[MASKED:{context.text}]", context)
        if context.location == SafetyLocation.COMPLETION
        else None
    )

    args = (
        "gpt-4o-mini",
        [
            {
                "role": "user",
                "content": [
                    "prompt-a",
                    {"type": "input_text", "text": "prompt-b"},
                    {"type": "image_url", "text": "ignored"},
                ],
            }
        ],
    )
    updated_args, unchanged_kwargs = apply_prompt_safety(
        None, args, {}, "chat", "litellm.completion"
    )
    assert unchanged_kwargs == {}
    assert updated_args[1][0]["content"][:2] == ["[MASKED:prompt-a]", {"type": "input_text", "text": "[MASKED:prompt-b]"}]
    assert _get_messages(updated_args, {})[1] == "args"
    assert apply_prompt_safety(None, (), {"messages": "invalid"}, "chat", "litellm.completion") == (
        (),
        {"messages": "invalid"},
    )
    unchanged_prompt_args, updated_prompt_kwargs = apply_prompt_safety(
        None,
        (),
        {"prompt": ["prompt-a", ["prompt-b", [1, 2, 3]]]},
        "completion",
        "litellm.text_completion",
    )
    assert unchanged_prompt_args == ()
    assert updated_prompt_kwargs["prompt"][0] == "[MASKED:prompt-a]"
    assert updated_prompt_kwargs["prompt"][1][0] == "[MASKED:prompt-b]"

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=["completion-a", {"type": "output_text", "text": "completion-b"}]
                )
            ),
            SimpleNamespace(
                message=SimpleNamespace(content="completion-c"),
                text="completion-c",
            ),
            SimpleNamespace(text="fallback-text"),
        ]
    )
    apply_completion_safety(None, response, "chat", "litellm.completion")
    assert response.choices[0].message.content[:2] == [
        "[MASKED:completion-a]",
        {"type": "output_text", "text": "[MASKED:completion-b]"},
    ]
    assert response.choices[1].message.content == "[MASKED:completion-c]"
    assert response.choices[1].text == "[MASKED:completion-c]"
    assert response.choices[2].text == "[MASKED:fallback-text]"
    assert extract_prompt_texts(["a", ["b", [1, 2, 3]]]) == ["a", "b"]
    assert extract_text_content(["a", {"type": "output_text", "text": "b"}]) == "a\nb"
    assert extract_text_content([{"type": "image_url", "text": "ignored"}]) is None
    assert _resolve_masked_text("same", None) == ("same", False)
    assert _resolve_masked_text("same", SafetyResult(text="same", overall_action="MASK", findings=[])) == (
        "same",
        False,
    )
