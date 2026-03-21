from types import SimpleNamespace

import pytest

from opentelemetry.instrumentation.fortifyroot import (
    SafetyFinding,
    SafetyLocation,
    SafetyResult,
    clear_completion_safety_stream_factory,
    clear_safety_handlers,
    register_completion_safety_handler,
    register_completion_safety_stream_factory,
    register_prompt_safety_handler,
)
from opentelemetry.instrumentation.ollama import (
    _aaccumulate_streaming_response,
    _accumulate_streaming_response,
)
from opentelemetry.instrumentation.ollama.safety import (
    _apply_completion_safety,
    _apply_prompt_safety,
    _mask_completion_content,
    _mask_prompt_content,
    _resolve_masked_text,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import LLMRequestTypeValues

pytestmark = pytest.mark.fr


def setup_function():
    clear_safety_handlers()
    clear_completion_safety_stream_factory()


def teardown_function():
    clear_safety_handlers()
    clear_completion_safety_stream_factory()


def _test_span():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)
    return exporter, tracer


def test_prompt_safety_masks_chat_json_messages():
    _, tracer = _test_span()
    register_prompt_safety_handler(
        lambda context: SafetyResult(
            text="[PII.chat]",
            overall_action="MASK",
            findings=[SafetyFinding("PII", "HIGH", "MASK", "PII.chat", 0, len(context.text))],
        )
        if context.location == SafetyLocation.PROMPT and context.text == "secret"
        else None
    )

    kwargs = {"json": {"messages": [{"role": "user", "content": "secret"}]}}
    with tracer.start_as_current_span("ollama.chat") as span:
        updated_kwargs = _apply_prompt_safety(
            span, kwargs, LLMRequestTypeValues.CHAT, "ollama.chat"
        )

    assert updated_kwargs["json"]["messages"][0]["content"] == "[PII.chat]"


def test_completion_safety_masks_chat_message():
    exporter, tracer = _test_span()
    register_completion_safety_handler(
        lambda context: SafetyResult(
            text="[SECRET.chat]",
            overall_action="MASK",
            findings=[SafetyFinding("SECRET", "HIGH", "MASK", "SECRET.chat", 0, len(context.text))],
        )
        if context.location == SafetyLocation.COMPLETION and context.text == "secret"
        else None
    )

    response = {"message": {"content": "secret", "role": "assistant"}}
    with tracer.start_as_current_span("ollama.chat") as span:
        _apply_completion_safety(span, response, LLMRequestTypeValues.CHAT, "ollama.chat")

    assert response["message"]["content"] == "[SECRET.chat]"
    assert len(exporter.get_finished_spans()[0].events) == 1


def test_prompt_and_completion_cover_prompt_blocks_and_non_chat_response():
    _, tracer = _test_span()
    register_prompt_safety_handler(
        lambda context: SafetyResult(
            text=f"[MASKED:{context.text}]",
            overall_action="MASK",
            findings=[SafetyFinding("PII", "HIGH", "MASK", "PII.chat", 0, len(context.text))],
        )
        if context.location == SafetyLocation.PROMPT
        else None
    )
    register_completion_safety_handler(
        lambda context: SafetyResult(
            text=f"[MASKED:{context.text}]",
            overall_action="MASK",
            findings=[SafetyFinding("SECRET", "HIGH", "MASK", "SECRET.chat", 0, len(context.text))],
        )
        if context.location == SafetyLocation.COMPLETION
        else None
    )

    kwargs = {
        "json": {
            "prompt": "prompt-secret",
            "messages": [{"role": "user", "content": [{"text": "block-secret"}]}],
        }
    }
    response = {"response": "completion-secret"}
    with tracer.start_as_current_span("ollama.generate") as span:
        updated_kwargs = _apply_prompt_safety(
            span, kwargs, LLMRequestTypeValues.CHAT, "ollama.generate"
        )
        _apply_completion_safety(
            span, response, LLMRequestTypeValues.COMPLETION, "ollama.generate"
        )
        assert _mask_prompt_content(
            span,
            None,
            request_type="chat",
            span_name="ollama.generate",
            segment_index=0,
            segment_role="user",
        ) == (None, False)
        assert _mask_completion_content(
            span,
            None,
            request_type="completion",
            span_name="ollama.generate",
            segment_index=0,
        ) == (None, False)

    assert updated_kwargs["json"]["prompt"] == "[MASKED:prompt-secret]"
    assert updated_kwargs["json"]["messages"][0]["content"][0]["text"] == "[MASKED:block-secret]"
    assert response["response"] == "[MASKED:completion-secret]"
    assert _resolve_masked_text("same", None) == ("same", False)


def test_prompt_safety_ignores_non_dict_json_payload():
    _, tracer = _test_span()

    kwargs = {"json": "not-a-dict"}
    with tracer.start_as_current_span("ollama.chat") as span:
        updated_kwargs = _apply_prompt_safety(
            span, kwargs, LLMRequestTypeValues.CHAT, "ollama.chat"
        )

    assert updated_kwargs is kwargs


class _FakeStreamSession:
    def __init__(self, results, flush_result=""):
        self._results = list(results)
        self._flush_result = flush_result

    def process_chunk(self, text):
        return SafetyResult(text=self._results.pop(0), overall_action="MASK", findings=[])

    def flush(self):
        return SafetyResult(text=self._flush_result, overall_action="ALLOW", findings=[])


def test_streaming_completion_masks_sync_chunks(monkeypatch):
    _, tracer = _test_span()
    register_completion_safety_stream_factory(
        lambda _: _FakeStreamSession(["masked"], flush_result="tail")
    )
    handled = []
    monkeypatch.setattr(
        "opentelemetry.instrumentation.ollama._handle_response",
        lambda span, event_logger, llm_request_type, token_histogram, response: handled.append(response),
    )

    with tracer.start_as_current_span("ollama.generate") as span:
        yielded = list(
            _accumulate_streaming_response(
                span,
                None,
                None,
                LLMRequestTypeValues.COMPLETION,
                iter(
                    [
                        {"model": "llama3", "response": "secret", "done": False},
                        {"model": "llama3", "response": "", "done": True},
                    ]
                ),
            )
        )

    assert yielded[0]["response"] == "masked"
    assert yielded[1]["response"] == "tail"
    assert handled[0]["response"] == "maskedtail"


@pytest.mark.asyncio
async def test_streaming_completion_masks_async_chunks(monkeypatch):
    _, tracer = _test_span()
    register_completion_safety_stream_factory(
        lambda _: _FakeStreamSession(["masked"], flush_result="tail")
    )
    handled = []
    monkeypatch.setattr(
        "opentelemetry.instrumentation.ollama._handle_response",
        lambda span, event_logger, llm_request_type, token_histogram, response: handled.append(response),
    )

    class _AsyncResponse:
        def __init__(self, chunks):
            self._chunks = iter(chunks)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._chunks)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    with tracer.start_as_current_span("ollama.generate") as span:
        yielded = [
            item
            async for item in _aaccumulate_streaming_response(
                span,
                None,
                None,
                LLMRequestTypeValues.COMPLETION,
                _AsyncResponse(
                    [
                        {"model": "llama3", "response": "secret", "done": False},
                        {"model": "llama3", "response": "", "done": True},
                    ]
                ),
            )
        ]

    assert yielded[0]["response"] == "masked"
    assert yielded[1]["response"] == "tail"
    assert handled[0]["response"] == "maskedtail"
