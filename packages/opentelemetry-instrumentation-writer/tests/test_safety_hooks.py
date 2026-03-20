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
from opentelemetry.instrumentation.writer.safety import (
    _apply_completion_safety,
    _apply_prompt_safety,
    _resolve_masked_text,
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
    with tracer.start_as_current_span("writerai.chat") as span:
        updated_kwargs = _apply_prompt_safety(
            span, kwargs, LLMRequestTypeValues.CHAT, "writerai.chat"
        )

    assert updated_kwargs["messages"][0]["content"] == "[PII.chat]"


def test_completion_safety_masks_choice_text():
    exporter, tracer = _test_span()
    register_completion_safety_handler(
        lambda context: SafetyResult(
            text="[SECRET.output]",
            overall_action="MASK",
            findings=[SafetyFinding("SECRET", "HIGH", "MASK", "SECRET.output", 0, len(context.text))],
        )
        if context.location == SafetyLocation.COMPLETION and context.text == "secret"
        else None
    )

    response = SimpleNamespace(choices=[SimpleNamespace(text="secret")])
    with tracer.start_as_current_span("writerai.completions") as span:
        _apply_completion_safety(
            span, response, LLMRequestTypeValues.COMPLETION, "writerai.completions"
        )

    assert response.choices[0].text == "[SECRET.output]"
    assert len(exporter.get_finished_spans()[0].events) == 1


def test_prompt_and_completion_cover_prompt_and_message_paths():
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
            findings=[SafetyFinding("SECRET", "HIGH", "MASK", "SECRET.output", 0, len(context.text))],
        )
        if context.location == SafetyLocation.COMPLETION
        else None
    )

    with tracer.start_as_current_span("writerai.chat") as span:
        updated_kwargs = _apply_prompt_safety(
            span,
            {"prompt": "prompt-secret", "messages": [{"role": "user", "content": "message-secret"}]},
            LLMRequestTypeValues.CHAT,
            "writerai.chat",
        )
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="message-secret"))])
        _apply_completion_safety(span, response, LLMRequestTypeValues.CHAT, "writerai.chat")
        unchanged = _apply_prompt_safety(
            span, {"messages": "invalid"}, LLMRequestTypeValues.CHAT, "writerai.chat"
        )

    assert updated_kwargs["prompt"] == "[MASKED:prompt-secret]"
    assert updated_kwargs["messages"][0]["content"] == "[MASKED:message-secret]"
    assert response.choices[0].message.content == "[MASKED:message-secret]"
    assert unchanged["messages"] == "invalid"
    assert _resolve_masked_text("same", None) == ("same", False)
    assert _resolve_masked_text("same", SafetyResult(text="same", overall_action="MASK", findings=[])) == (
        "same",
        False,
    )


def test_prompt_and_completion_mask_mixed_content_blocks():
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
            findings=[SafetyFinding("SECRET", "HIGH", "MASK", "SECRET.output", 0, len(context.text))],
        )
        if context.location == SafetyLocation.COMPLETION
        else None
    )

    kwargs = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "prompt-secret"},
                    {"type": "image_url", "text": "leave-image-alone"},
                ],
            }
        ]
    }
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=[
                        SimpleNamespace(type="text", text="completion-secret"),
                        SimpleNamespace(type="image_url", text="leave-image-alone"),
                    ]
                )
            )
        ]
    )

    with tracer.start_as_current_span("writerai.chat") as span:
        updated_kwargs = _apply_prompt_safety(
            span,
            kwargs,
            LLMRequestTypeValues.CHAT,
            "writerai.chat",
        )
        _apply_completion_safety(
            span,
            response,
            LLMRequestTypeValues.CHAT,
            "writerai.chat",
        )

    assert updated_kwargs["messages"][0]["content"][0]["text"] == "[MASKED:prompt-secret]"
    assert updated_kwargs["messages"][0]["content"][1]["text"] == "leave-image-alone"
    assert response.choices[0].message.content[0].text == "[MASKED:completion-secret]"
    assert response.choices[0].message.content[1].text == "leave-image-alone"
