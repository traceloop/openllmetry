from types import SimpleNamespace

import pytest

from opentelemetry.instrumentation.fortifyroot import (
    SafetyFinding,
    SafetyLocation,
    SafetyResult,
    clear_safety_handlers,
    register_completion_safety_handler,
    register_completion_safety_stream_factory,
    register_prompt_safety_handler,
)
from opentelemetry.instrumentation.groq import _awrap
from opentelemetry.instrumentation.groq.safety import (
    _apply_completion_safety,
    _apply_prompt_safety,
    _mask_completion_content,
    _mask_prompt_content,
    _resolve_masked_text,
)
from opentelemetry.instrumentation.groq.streaming_safety import (
    GroqStreamingSafety,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import SpanAttributes

pytestmark = pytest.mark.fr


def setup_function():
    clear_safety_handlers()


def teardown_function():
    clear_safety_handlers()


def _test_span():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)
    return exporter, tracer


def test_prompt_safety_masks_chat_message():
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

    kwargs = {"messages": [{"role": "user", "content": "secret"}]}
    with tracer.start_as_current_span("groq.chat") as span:
        updated_kwargs = _apply_prompt_safety(span, kwargs, "groq.chat")

    assert updated_kwargs["messages"][0]["content"] == "[PII.chat]"


def test_completion_safety_masks_chat_choice():
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

    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="secret"))])
    with tracer.start_as_current_span("groq.chat") as span:
        _apply_completion_safety(span, response, "groq.chat")

    assert response.choices[0].message.content == "[SECRET.chat]"
    assert len(exporter.get_finished_spans()[0].events) == 1


def test_prompt_safety_masks_supported_chat_block_content_only():
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

    kwargs = {
        "prompt": "secret-prompt",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "secret-block"},
                    {"type": "image_url", "text": "ignored"},
                ],
            }
        ],
    }
    with tracer.start_as_current_span("groq.chat") as span:
        updated_kwargs = _apply_prompt_safety(span, kwargs, "groq.chat")

    assert updated_kwargs["prompt"] == "secret-prompt"
    assert updated_kwargs["messages"][0]["content"][0]["text"] == "[MASKED:secret-block]"
    assert kwargs["messages"][0]["content"][0]["text"] == "secret-block"


def test_completion_helpers_cover_text_and_passthrough_paths():
    exporter, tracer = _test_span()
    register_completion_safety_handler(
        lambda context: SafetyResult(
            text=f"[MASKED:{context.text}]",
            overall_action="MASK",
            findings=[SafetyFinding("SECRET", "HIGH", "MASK", "SECRET.chat", 0, len(context.text))],
        )
        if context.location == SafetyLocation.COMPLETION
        else None
    )

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(text="secret-text"),
            SimpleNamespace(message=SimpleNamespace(content=[{"type": "output_text", "text": "secret-block"}])),
        ]
    )
    with tracer.start_as_current_span("groq.chat") as span:
        _apply_completion_safety(span, response, "groq.chat")
        assert _mask_prompt_content(span, 123, span_name="groq.chat", segment_index=0, segment_role="user") == (
            123,
            False,
        )
        assert _mask_completion_content(span, None, span_name="groq.chat", segment_index=0) == (
            None,
            False,
        )

    assert response.choices[0].text == "[MASKED:secret-text]"
    assert response.choices[1].message.content[0]["text"] == "[MASKED:secret-block]"
    assert _resolve_masked_text("same", None) == ("same", False)
    unchanged = SafetyResult(text="same", overall_action="MASK", findings=[])
    assert _resolve_masked_text("same", unchanged) == ("same", False)
    assert len(exporter.get_finished_spans()[0].events) >= 2


@pytest.mark.asyncio
async def test_async_wrapper_masks_completion_without_metrics_histogram():
    exporter, tracer = _test_span()
    register_completion_safety_handler(
        lambda context: SafetyResult(
            text="[MASKED:secret]",
            overall_action="MASK",
            findings=[SafetyFinding("SECRET", "HIGH", "MASK", "SECRET.chat", 0, len(context.text))],
        )
        if context.location == SafetyLocation.COMPLETION and context.text == "secret"
        else None
    )

    async def wrapped(*args, **kwargs):
        return {
            "model": "groq-test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "secret"},
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    wrapper = _awrap(tracer, None, None, None, None, {"span_name": "groq.chat"})
    response = await wrapper(
        wrapped,
        None,
        (),
        {"messages": [{"role": "user", "content": "prompt"}]},
    )

    assert response["choices"][0]["message"]["content"] == "[MASKED:secret]"
    span = exporter.get_finished_spans()[0]
    assert span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "[MASKED:secret]"


class _FakeStreamSession:
    def process_chunk(self, text):
        return SafetyResult(text="masked", overall_action="allow", findings=[])

    def flush(self):
        return SafetyResult(text="-tail", overall_action="allow", findings=[])


def test_streaming_helper_masks_delta_content_and_flushes_on_finish():
    _, tracer = _test_span()
    register_completion_safety_stream_factory(lambda _: _FakeStreamSession())
    chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content="secret"),
                finish_reason="stop",
            )
        ]
    )

    with tracer.start_as_current_span("groq.chat") as span:
        helper = GroqStreamingSafety(span, "groq.chat")
        helper.process_chunk(chunk)

    assert chunk.choices[0].delta.content == "masked-tail"


def test_streaming_helper_flushes_tail_when_finish_chunk_has_no_content():
    _, tracer = _test_span()
    register_completion_safety_stream_factory(lambda _: _FakeStreamSession())
    first_chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content="secret"),
                finish_reason=None,
            )
        ]
    )
    final_chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(),
                finish_reason="stop",
            )
        ]
    )

    with tracer.start_as_current_span("groq.chat") as span:
        helper = GroqStreamingSafety(span, "groq.chat")
        helper.process_chunk(first_chunk)
        helper.process_chunk(final_chunk)

    assert first_chunk.choices[0].delta.content == "masked"
    assert final_chunk.choices[0].delta.content == "-tail"


def test_streaming_helper_fail_opens_when_stream_session_raises():
    class _ExplodingSession:
        def process_chunk(self, text):
            raise RuntimeError("boom")

        def flush(self):
            raise RuntimeError("boom")

    _, tracer = _test_span()
    register_completion_safety_stream_factory(lambda _: _ExplodingSession())
    first_chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content="secret"),
                finish_reason=None,
            )
        ]
    )
    final_chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=None),
                finish_reason="stop",
            )
        ]
    )

    with tracer.start_as_current_span("groq.chat") as span:
        helper = GroqStreamingSafety(span, "groq.chat")
        helper.process_chunk(first_chunk)
        helper.process_chunk(final_chunk)

    assert first_chunk.choices[0].delta.content == "secret"
    assert final_chunk.choices[0].delta.content is None
