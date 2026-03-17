from types import SimpleNamespace

import pytest

from opentelemetry.instrumentation.fortifyroot import (
    SafetyFinding,
    SafetyLocation,
    SafetyResult,
    clear_safety_handlers,
    register_completion_safety_handler,
    register_prompt_safety_handler,
)
from opentelemetry.instrumentation.mistralai.safety import (
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

pytestmark = pytest.mark.safety


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
