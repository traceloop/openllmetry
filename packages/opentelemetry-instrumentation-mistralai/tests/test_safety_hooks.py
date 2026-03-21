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
from opentelemetry.instrumentation.mistralai.safety import (
    _apply_completion_safety,
    _apply_prompt_safety,
    _mask_completion_content,
    _mask_prompt_content,
    _resolve_masked_text,
)
from opentelemetry.instrumentation.mistralai.streaming_safety import (
    MistralAIStreamingSafety,
)
from opentelemetry.instrumentation.mistralai import (
    _aaccumulate_streaming_response,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import LLMRequestTypeValues

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


def test_prompt_safety_masks_message_content():
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

    kwargs = {"messages": [SimpleNamespace(role="user", content="secret")]}
    with tracer.start_as_current_span("mistralai.chat") as span:
        updated_kwargs = _apply_prompt_safety(
            span, kwargs, LLMRequestTypeValues.CHAT, "mistralai.chat"
        )

    assert updated_kwargs["messages"][0].content == "[PII.chat]"


def test_completion_safety_masks_choice_message():
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
    with tracer.start_as_current_span("mistralai.chat") as span:
        _apply_completion_safety(span, response, LLMRequestTypeValues.CHAT, "mistralai.chat")

    assert response.choices[0].message.content == "[SECRET.chat]"
    assert len(exporter.get_finished_spans()[0].events) == 1


def test_prompt_safety_handles_non_chat_and_content_blocks():
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

    with tracer.start_as_current_span("mistralai.chat") as span:
        assert _apply_prompt_safety(
            span,
            {"messages": [SimpleNamespace(role="user", content="secret")]},
            LLMRequestTypeValues.COMPLETION,
            "mistralai.chat",
        )["messages"][0].content == "secret"
        updated_kwargs = _apply_prompt_safety(
            span,
            {"messages": [SimpleNamespace(role="user", content=[SimpleNamespace(text="secret-block")])]},
            LLMRequestTypeValues.CHAT,
            "mistralai.chat",
        )
        assert _mask_prompt_content(
            span, None, span_name="mistralai.chat", segment_index=0, segment_role="user"
        ) == (None, False)

    assert updated_kwargs["messages"][0].content[0].text == "[MASKED:secret-block]"
    with tracer.start_as_current_span("mistralai.chat") as span:
        image_kwargs = _apply_prompt_safety(
            span,
            {
                "messages": [
                    SimpleNamespace(
                        role="user",
                        content=[SimpleNamespace(type="image_url", text="leave-alone")],
                    )
                ]
            },
            LLMRequestTypeValues.CHAT,
            "mistralai.chat",
        )
    assert image_kwargs["messages"][0].content[0].text == "leave-alone"


def test_completion_helpers_cover_passthrough_and_block_paths():
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
            SimpleNamespace(message=SimpleNamespace(content=None)),
            SimpleNamespace(message=SimpleNamespace(content=[SimpleNamespace(text="secret-block")])),
        ]
    )
    with tracer.start_as_current_span("mistralai.chat") as span:
        _apply_completion_safety(span, response, LLMRequestTypeValues.CHAT, "mistralai.chat")
        _apply_completion_safety(span, response, LLMRequestTypeValues.COMPLETION, "mistralai.chat")
        assert _mask_completion_content(
            span, None, span_name="mistralai.chat", segment_index=0
        ) == (None, False)

    assert response.choices[1].message.content[0].text == "[MASKED:secret-block]"
    assert _resolve_masked_text("same", None) == ("same", False)
    unchanged = SafetyResult(text="same", overall_action="MASK", findings=[])
    assert _resolve_masked_text("same", unchanged) == ("same", False)
    assert exporter.get_finished_spans()


class _FakeStreamSession:
    def process_chunk(self, text):
        return SafetyResult(text="masked", overall_action="allow", findings=[])

    def flush(self):
        return SafetyResult(text="-tail", overall_action="allow", findings=[])


def test_streaming_helper_masks_delta_content_and_flushes_on_finish():
    _, tracer = _test_span()
    register_completion_safety_stream_factory(lambda _: _FakeStreamSession())
    item = SimpleNamespace(
        data=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="secret"),
                    finish_reason="stop",
                )
            ]
        )
    )

    with tracer.start_as_current_span("mistralai.chat") as span:
        helper = MistralAIStreamingSafety(span, "mistralai.chat")
        helper.process_chunk(item)

    assert item.data.choices[0].delta.content == "masked-tail"


def test_streaming_helper_flushes_tail_when_finish_chunk_has_no_content():
    _, tracer = _test_span()
    register_completion_safety_stream_factory(lambda _: _FakeStreamSession())
    first_item = SimpleNamespace(
        data=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="secret"),
                    finish_reason=None,
                )
            ]
        )
    )
    final_item = SimpleNamespace(
        data=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(),
                    finish_reason="stop",
                )
            ]
        )
    )

    with tracer.start_as_current_span("mistralai.chat") as span:
        helper = MistralAIStreamingSafety(span, "mistralai.chat")
        helper.process_chunk(first_item)
        helper.process_chunk(final_item)

    assert first_item.data.choices[0].delta.content == "masked"
    assert final_item.data.choices[0].delta.content == "-tail"


@pytest.mark.asyncio
async def test_async_streaming_accumulator_masks_tail_before_final_response(monkeypatch):
    _, tracer = _test_span()
    register_completion_safety_stream_factory(lambda _: _FakeStreamSession())
    handled = []
    monkeypatch.setattr(
        "opentelemetry.instrumentation.mistralai._handle_response",
        lambda span, event_logger, llm_request_type, response: handled.append(response),
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

    first = SimpleNamespace(
        data=SimpleNamespace(
            id="resp-1",
            model="mistral-test",
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason=None,
                    delta=SimpleNamespace(content="secret", role="assistant"),
                )
            ],
        )
    )
    final = SimpleNamespace(
        data=SimpleNamespace(
            id="resp-1",
            model="mistral-test",
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    delta=SimpleNamespace(content=None, role="assistant"),
                )
            ],
        )
    )

    with tracer.start_as_current_span("mistralai.chat") as span:
        yielded = [
            item
            async for item in _aaccumulate_streaming_response(
                span,
                None,
                LLMRequestTypeValues.CHAT,
                _AsyncResponse([first, final]),
            )
        ]

    assert yielded[0].data.choices[0].delta.content == "masked"
    assert yielded[1].data.choices[0].delta.content == "-tail"
    assert handled[0].choices[0].message.content == "masked-tail"
