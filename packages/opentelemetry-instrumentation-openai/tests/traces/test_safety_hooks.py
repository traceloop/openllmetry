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
from opentelemetry.instrumentation.openai.shared.chat_wrappers import (
    _handle_request as _handle_chat_request,
    _apply_prompt_safety,
)
from opentelemetry.instrumentation.openai.shared.completion_wrappers import (
    _apply_completion_safety,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

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


def test_chat_prompt_safety_masks_message_content_without_mutating_input():
    exporter, tracer = _test_span()
    register_prompt_safety_handler(
        lambda context: SafetyResult(
            text="[PII.email]",
            overall_action="MASK",
            findings=[
                SafetyFinding(
                    category="PII",
                    severity="HIGH",
                    action="MASK",
                    rule_name="PII.email",
                    start=0,
                    end=len(context.text),
                )
            ],
        )
        if context.location == SafetyLocation.PROMPT and context.text == "secret"
        else None
    )

    kwargs = {"messages": [{"role": "user", "content": "secret"}]}
    with tracer.start_as_current_span("openai.chat") as span:
        updated_kwargs = _apply_prompt_safety(span, kwargs)

    assert kwargs["messages"][0]["content"] == "secret"
    assert updated_kwargs["messages"][0]["content"] == "[PII.email]"
    spans = exporter.get_finished_spans()
    assert len(spans[0].events) == 1
    assert spans[0].events[0].attributes["fortifyroot.safety.location"] == SafetyLocation.PROMPT.value


@pytest.mark.asyncio
async def test_chat_prompt_safety_masks_span_prompt_attributes():
    exporter, tracer = _test_span()
    register_prompt_safety_handler(
        lambda context: SafetyResult(
            text="[PII.email]",
            overall_action="MASK",
            findings=[
                SafetyFinding(
                    category="PII",
                    severity="HIGH",
                    action="MASK",
                    rule_name="PII.email",
                    start=0,
                    end=len(context.text),
                )
            ],
        )
        if context.location == SafetyLocation.PROMPT and context.text == "secret"
        else None
    )

    kwargs = {"messages": [{"role": "user", "content": "secret"}]}
    instance = SimpleNamespace(_client=None)
    with tracer.start_as_current_span("openai.chat") as span:
        updated_kwargs = _apply_prompt_safety(span, kwargs)
        await _handle_chat_request(span, updated_kwargs, instance)

    spans = exporter.get_finished_spans()
    assert spans[0].attributes["gen_ai.prompt.0.content"] == "[PII.email]"
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"


def test_completion_response_safety_masks_choice_text():
    exporter, tracer = _test_span()
    register_completion_safety_handler(
        lambda context: SafetyResult(
            text="[SECRET.api_key]",
            overall_action="MASK",
            findings=[
                SafetyFinding(
                    category="SECRET",
                    severity="HIGH",
                    action="MASK",
                    rule_name="SECRET.api_key",
                    start=0,
                    end=len(context.text),
                )
            ],
        )
        if context.location == SafetyLocation.COMPLETION and context.text == "secret"
        else None
    )

    response = SimpleNamespace(choices=[SimpleNamespace(text="secret")])
    with tracer.start_as_current_span("openai.completion") as span:
        _apply_completion_safety(span, response)

    assert response.choices[0].text == "[SECRET.api_key]"
    spans = exporter.get_finished_spans()
    assert len(spans[0].events) == 1
    assert spans[0].events[0].attributes["fortifyroot.safety.action"] == "MASK"
