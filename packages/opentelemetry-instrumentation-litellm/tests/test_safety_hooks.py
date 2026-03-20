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
    register_completion_safety_stream_factory,
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
from opentelemetry.instrumentation.litellm.streaming_safety import (
    _accumulate_streaming_chunk,
    _mask_streaming_chunk,
    is_async_streaming_response,
    is_sync_streaming_response,
    wrap_async_streaming_response,
    wrap_sync_streaming_response,
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

pytestmark = pytest.mark.fr


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


def test_streaming_completion_masks_prompt_and_stream_chunks():
    exporter, tracer = _test_tracer()
    register_prompt_safety_handler(
        lambda context: _prompt_result("[PII.email]", context)
        if context.location == SafetyLocation.PROMPT and context.text == "secret"
        else None
    )
    register_completion_safety_stream_factory(
        lambda _: _FakeStreamSession(["masked-a"], flush_result="tail")
    )

    def wrapped(*args, **kwargs):
        assert kwargs["messages"][0]["content"] == "[PII.email]"

        def _stream():
            yield SimpleNamespace(
                model="gpt-4o-mini",
                choices=[
                    SimpleNamespace(
                        finish_reason=None,
                        message=SimpleNamespace(role="assistant", content="secret"),
                    )
                ],
            )
            yield SimpleNamespace(
                model="gpt-4o-mini",
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(role="assistant", content=None),
                    )
                ],
            )

        return _stream()

    response = list(
        _invoke_completion(
            tracer,
            wrapped,
            (),
            {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "secret"}],
                "stream": True,
            },
        )
    )

    assert response[0].choices[0].message.content == "masked-a"
    assert response[1].choices[0].message.content == "tail"
    span = exporter.get_finished_spans()[0]
    assert span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "[PII.email]"
    assert span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "masked-atail"


class _FakeStreamSession:
    def __init__(self, process_results, flush_result=""):
        self._process_results = list(process_results)
        self._flush_result = flush_result

    def process_chunk(self, text):
        return SafetyResult(text=self._process_results.pop(0), overall_action="allow", findings=[])

    def flush(self):
        return SafetyResult(text=self._flush_result, overall_action="allow", findings=[])


def test_streaming_text_completion_masks_chunks():
    exporter, tracer = _test_tracer()
    register_completion_safety_stream_factory(
        lambda _: _FakeStreamSession(["masked-a"], flush_result="tail")
    )

    def wrapped(*args, **kwargs):
        def _stream():
            yield SimpleNamespace(
                model="gpt-3.5-turbo-instruct",
                choices=[SimpleNamespace(text="secret", finish_reason=None)],
            )
            yield SimpleNamespace(
                model="gpt-3.5-turbo-instruct",
                choices=[SimpleNamespace(finish_reason="stop")],
            )

        return _stream()

    response = list(
        _invoke_completion(
            tracer,
            wrapped,
            ("secret",),
            {"model": "gpt-3.5-turbo-instruct", "stream": True},
            is_text_completion=True,
        )
    )

    assert response[0].choices[0].text == "masked-a"
    assert response[1].choices[0].text == "tail"
    assert exporter.get_finished_spans()[0].attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "masked-atail"


def test_streaming_completion_is_not_skipped():
    exporter, tracer = _test_tracer()

    def wrapped(*args, **kwargs):
        def _stream():
            if False:
                yield None
        return _stream()

    response = _invoke_completion(
        tracer,
        wrapped,
        (),
        {
            "model": "gpt-4o-mini",
            "stream": True,
        },
    )

    assert list(response) == []
    assert exporter.get_finished_spans()


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
    assert exporter.get_finished_spans()

    async def raising(*args, **kwargs):
        raise RuntimeError("async boom")

    with pytest.raises(RuntimeError, match="async boom"):
        await _invoke_acompletion(
            tracer,
            raising,
            (),
            {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "secret"}]},
        )

    error_span = exporter.get_finished_spans()[-1]
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


def test_streaming_helper_units_cover_text_mirroring_usage_and_detector_helpers():
    class _FakeStreamGroup:
        def process(self, key, text, **kwargs):
            return "masked"

        def flush(self, key):
            return "tail"

    chunk = SimpleNamespace(
        model="gpt-4o-mini",
        usage={"total_tokens": 3},
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(role="assistant", content="secret"),
                text="secret",
            )
        ],
    )
    complete_response = {"choices": [], "model": None, "usage": None}

    _mask_streaming_chunk(_FakeStreamGroup(), chunk)
    _accumulate_streaming_chunk(complete_response, chunk)

    assert chunk.choices[0].message.content == "maskedtail"
    assert chunk.choices[0].text == "maskedtail"
    assert complete_response["usage"] == {"total_tokens": 3}
    assert is_sync_streaming_response((item for item in ())) is True

    async def _agen():
        yield chunk

    assert is_async_streaming_response(_agen()) is True


def test_sync_streaming_wrapper_records_error_and_re_raises():
    _, tracer = _test_tracer()
    span = tracer.start_span("litellm.completion")

    def _response():
        yield SimpleNamespace(choices=[])
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        list(
            wrap_sync_streaming_response(
                span,
                _response(),
                "chat",
                "litellm.completion",
                lambda *_: None,
            )
        )

    assert span.status.status_code.name == "ERROR"


@pytest.mark.asyncio
async def test_async_streaming_wrapper_finalizes_span():
    _, tracer = _test_tracer()
    span = tracer.start_span("litellm.completion")
    responses = []

    async def _response():
        yield SimpleNamespace(
            model="gpt-4o-mini",
            usage={"total_tokens": 3},
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(role="assistant", content="secret"),
                    text="secret",
                )
            ],
        )

    yielded = [
        chunk
        async for chunk in wrap_async_streaming_response(
            span,
            _response(),
            "chat",
            "litellm.completion",
            lambda _span, response: responses.append(response),
        )
    ]

    assert yielded[0].choices[0].message.content == "secret"
    assert responses[0].usage == {"total_tokens": 3}
