from types import SimpleNamespace

import pytest

from opentelemetry.instrumentation.fortifyroot import SafetyDecision, SafetyResult
from opentelemetry.instrumentation.google_generativeai import safety

pytestmark = pytest.mark.fr


def test_apply_prompt_safety_masks_args_kwargs_and_parts(monkeypatch):
    monkeypatch.setattr(safety, "run_prompt_safety", lambda **kwargs: SafetyResult(text=f"masked:{kwargs['text']}", overall_action="MASK"))
    args = ("secret",)
    kwargs = {
        "contents": [
            "list-secret",
            SimpleNamespace(parts=[SimpleNamespace(text="part-secret")]),
        ]
    }

    updated_args, updated_kwargs = safety._apply_prompt_safety(None, args, kwargs, "gemini.generate_content")

    assert updated_args == ("masked:secret",)
    assert updated_kwargs["contents"][0] == "masked:list-secret"
    assert updated_kwargs["contents"][1].parts[0].text == "masked:part-secret"


def test_google_apply_prompt_safety_noop_and_exception_paths(monkeypatch):
    monkeypatch.setattr(safety, "run_prompt_safety", lambda **kwargs: SafetyResult(text=kwargs["text"], overall_action="MASK"))
    args = ("keep",)
    kwargs = {"contents": "keep"}
    updated_args, updated_kwargs = safety._apply_prompt_safety(None, args, kwargs, "gemini.generate_content")
    assert updated_args == args
    assert updated_kwargs is kwargs

    monkeypatch.setattr(safety, "_mask_prompt_value", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    assert safety._apply_prompt_safety(None, args, kwargs, "gemini.generate_content") == (args, kwargs)


def test_apply_completion_safety_masks_text_and_candidate_parts(monkeypatch):
    monkeypatch.setattr(safety, "run_completion_safety", lambda **kwargs: SafetyResult(text=f"masked:{kwargs['text']}", overall_action="MASK"))
    response = SimpleNamespace(
        text="secret",
        candidates=[
            SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="part-secret")])),
            SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(binary=b"x")])),
        ],
    )

    safety._apply_completion_safety(None, response, "gemini.generate_content")

    assert response.text == "masked:secret"
    assert response.candidates[0].content.parts[0].text == "masked:part-secret"


def test_google_prompt_and_completion_helpers_cover_noop_branches(monkeypatch):
    monkeypatch.setattr(safety, "run_prompt_safety", lambda **kwargs: SafetyResult(text=kwargs["text"], overall_action="MASK"))
    assert safety._mask_prompt_value(None, {"x": 1}, span_name="gemini.generate_content", segment_index=0, segment_role="user") == ({"x": 1}, False)
    updated, changed = safety._mask_prompt_value(
        None,
        ["keep", SimpleNamespace(parts=[SimpleNamespace(binary=b"x")])],
        span_name="gemini.generate_content",
        segment_index=0,
        segment_role="user",
    )
    assert changed is False
    assert updated[0] == "keep"

    monkeypatch.setattr(safety, "run_completion_safety", lambda **kwargs: SafetyResult(text=kwargs["text"], overall_action="MASK"))
    response = SimpleNamespace(text="keep", candidates="not-a-list")
    assert safety._apply_completion_safety(None, response, "gemini.generate_content") is None

    response = SimpleNamespace(text="keep", candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(binary=b"x")]))])
    safety._apply_completion_safety(None, response, "gemini.generate_content")
    assert response.text == "keep"

    monkeypatch.setattr(safety, "get_object_value", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    assert safety._apply_completion_safety(None, SimpleNamespace(text="secret"), "gemini.generate_content") is None


def test_google_resolve_masked_text_and_fail_open(monkeypatch):
    assert safety._resolve_masked_text("x", None) == ("x", False)
    assert safety._resolve_masked_text("x", SafetyResult(text="x", overall_action=SafetyDecision.MASK.value)) == ("x", False)
    assert safety._resolve_masked_text("x", SafetyResult(text="y", overall_action=SafetyDecision.ALLOW.value)) == ("x", False)
    assert safety._resolve_masked_text("x", SafetyResult(text="y", overall_action=SafetyDecision.MASK.value)) == ("y", True)

    args = ("secret",)
    monkeypatch.setattr(safety, "clone_value", lambda *_: (_ for _ in ()).throw(RuntimeError("boom")))
    assert safety._apply_prompt_safety(None, args, {}, "gemini.generate_content") == (args, {})
