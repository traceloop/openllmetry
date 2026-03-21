from types import SimpleNamespace

import pytest
from aleph_alpha_client import Prompt

from opentelemetry.instrumentation.alephalpha import safety as alephalpha_safety
from opentelemetry.instrumentation.alephalpha.safety import (
    _apply_completion_safety,
    _apply_prompt_safety,
    _mask_prompt_value,
    _resolve_masked_text,
)
from opentelemetry.instrumentation.fortifyroot import (
    SafetyFinding,
    SafetyLocation,
    SafetyResult,
    clear_safety_handlers,
    register_completion_safety_handler,
    register_prompt_safety_handler,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

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


def test_prompt_safety_masks_request_prompt():
    exporter, tracer = _test_span()
    register_prompt_safety_handler(
        lambda context: SafetyResult(
            text=f"[PII:{context.text}]",
            overall_action="MASK",
            findings=[SafetyFinding("PII", "HIGH", "MASK", "PII.prompt", 0, len(context.text))],
        )
        if context.location == SafetyLocation.PROMPT
        else None
    )

    request = SimpleNamespace(
        prompt=Prompt.from_json(
            [
                {"type": "text", "data": "secret-a"},
                {"type": "text", "data": "secret-b"},
                {"type": "token_ids", "data": [1, 2, 3]},
            ]
        )
    )
    with tracer.start_as_current_span("alephalpha.completion") as span:
        _apply_prompt_safety(span, (request,), {}, "alephalpha.completion")

    prompt_json = request.prompt.to_json()
    assert prompt_json[0]["data"] == "[PII:secret-a]"
    assert prompt_json[1]["data"] == "[PII:secret-b]"
    assert prompt_json[2]["data"] == [1, 2, 3]
    assert len(exporter.get_finished_spans()[0].events) == 2


def test_completion_safety_masks_completion_text():
    exporter, tracer = _test_span()
    register_completion_safety_handler(
        lambda context: SafetyResult(
            text=f"[SECRET:{context.text}]",
            overall_action="MASK",
            findings=[SafetyFinding("SECRET", "HIGH", "MASK", "SECRET.output", 0, len(context.text))],
        )
        if context.location == SafetyLocation.COMPLETION
        else None
    )

    response = SimpleNamespace(
        completions=[
            SimpleNamespace(completion="secret-a"),
            SimpleNamespace(completion="secret-b"),
        ]
    )
    with tracer.start_as_current_span("alephalpha.completion") as span:
        _apply_completion_safety(span, response, "alephalpha.completion")

    assert response.completions[0].completion == "[SECRET:secret-a]"
    assert response.completions[1].completion == "[SECRET:secret-b]"
    assert len(exporter.get_finished_spans()[0].events) == 2


def test_helpers_cover_passthrough_and_prompt_extraction_branches():
    _, tracer = _test_span()

    class BrokenRequest:
        @property
        def prompt(self):
            raise RuntimeError("boom")

    class BrokenResponse:
        @property
        def completions(self):
            raise RuntimeError("boom")

    request = SimpleNamespace(prompt=Prompt.from_text("plain"))
    broken_request = BrokenRequest()
    with tracer.start_as_current_span("alephalpha.completion") as span:
        assert _apply_prompt_safety(span, (request,), {}, "alephalpha.completion") == (
            (request,),
            {},
        )
        _apply_completion_safety(
            span,
            SimpleNamespace(completions=[]),
            "alephalpha.completion",
        )
        _apply_completion_safety(
            span,
            SimpleNamespace(completions=[SimpleNamespace(completion=None)]),
            "alephalpha.completion",
        )
        assert _apply_prompt_safety(
            span, (broken_request,), {}, "alephalpha.completion"
        ) == ((broken_request,), {})
        _apply_completion_safety(span, BrokenResponse(), "alephalpha.completion")
        dict_request = SimpleNamespace(prompt={"type": "text", "data": "plain"})
        assert _apply_prompt_safety(
            span,
            (dict_request,),
            {},
            "alephalpha.completion",
        ) == ((dict_request,), {})

    assert _resolve_masked_text("same", None) == ("same", False)
    unchanged = SafetyResult(text="same", overall_action="MASK", findings=[])
    assert _resolve_masked_text("same", unchanged) == ("same", False)


def test_prompt_safety_keeps_plain_string_prompts_as_strings():
    _, tracer = _test_span()
    register_prompt_safety_handler(
        lambda context: SafetyResult(
            text="[PII:secret]",
            overall_action="MASK",
            findings=[SafetyFinding("PII", "HIGH", "MASK", "PII.prompt", 0, len(context.text))],
        )
        if context.location == SafetyLocation.PROMPT and context.text == "secret"
        else None
    )

    request = SimpleNamespace(prompt="secret")
    with tracer.start_as_current_span("alephalpha.completion") as span:
        _apply_prompt_safety(span, (request,), {}, "alephalpha.completion")

    assert request.prompt == "[PII:secret]"
    assert isinstance(request.prompt, str)


def test_mask_prompt_value_falls_back_to_json_when_prompt_class_is_unavailable(monkeypatch):
    register_prompt_safety_handler(
        lambda context: SafetyResult(
            text="[PII:secret]",
            overall_action="MASK",
            findings=[SafetyFinding("PII", "HIGH", "MASK", "PII.prompt", 0, len(context.text))],
        )
        if context.location == SafetyLocation.PROMPT and context.text == "secret"
        else None
    )
    monkeypatch.setattr(alephalpha_safety, "AlephAlphaPrompt", None)

    updated_prompt, changed = _mask_prompt_value(
        None,
        [{"type": "text", "data": "secret"}],
        span_name="alephalpha.completion",
    )

    assert changed is True
    assert updated_prompt == [{"type": "text", "data": "[PII:secret]"}]
