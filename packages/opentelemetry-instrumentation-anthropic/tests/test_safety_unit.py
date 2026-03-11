from types import SimpleNamespace

import pytest

from opentelemetry.instrumentation.anthropic import safety
from opentelemetry.instrumentation.fortifyroot import SafetyDecision, SafetyResult

pytestmark = pytest.mark.safety


def test_apply_prompt_safety_masks_prompt_system_and_messages(monkeypatch):
    monkeypatch.setattr(safety, "run_prompt_safety", lambda **kwargs: SafetyResult(text=f"masked:{kwargs['text']}", overall_action="MASK"))

    kwargs = {
        "prompt": "secret",
        "system": [{"type": "text", "text": "sys-secret"}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "msg-secret"}]}],
    }
    updated = safety._apply_prompt_safety(None, kwargs, "anthropic.chat")

    assert kwargs["prompt"] == "secret"
    assert updated["prompt"] == "masked:secret"
    assert updated["system"][0]["text"] == "masked:sys-secret"
    assert updated["messages"][0]["content"][0]["text"] == "masked:msg-secret"


def test_apply_prompt_safety_returns_partial_update_when_messages_missing():
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(safety, "run_prompt_safety", lambda **kwargs: SafetyResult(text=f"masked:{kwargs['text']}", overall_action="MASK"))
    kwargs = {"prompt": "secret"}
    updated = safety._apply_prompt_safety(None, kwargs, "anthropic.completion")
    assert updated is not kwargs
    assert updated["prompt"] == "masked:secret"
    monkeypatch.undo()


def test_apply_completion_safety_masks_completion_and_content(monkeypatch):
    monkeypatch.setattr(safety, "run_completion_safety", lambda **kwargs: SafetyResult(text=f"masked:{kwargs['text']}", overall_action="MASK"))
    response = SimpleNamespace(
        completion="secret",
        content=[
            {"type": "text", "text": "text-secret"},
            {"type": "thinking", "thinking": "thought-secret"},
            {"type": "tool_use", "name": "ignored"},
        ],
    )

    safety._apply_completion_safety(None, response, "anthropic.chat")

    assert response.completion == "masked:secret"
    assert response.content[0]["text"] == "masked:text-secret"
    assert response.content[1]["thinking"] == "masked:thought-secret"


def test_anthropic_prompt_and_completion_helpers_cover_noop_branches(monkeypatch):
    monkeypatch.setattr(safety, "run_prompt_safety", lambda **kwargs: SafetyResult(text=kwargs["text"], overall_action="MASK"))
    updated, changed = safety._mask_prompt_content(
        None,
        [{"type": "tool_use", "text": "ignored"}, {"type": "text", "text": 1}],
        span_name="anthropic.chat",
        request_type="chat",
        segment_index=0,
        segment_role="user",
    )
    assert changed is False
    assert updated[0]["text"] == "ignored"

    monkeypatch.setattr(safety, "run_completion_safety", lambda **kwargs: SafetyResult(text=kwargs["text"], overall_action="MASK"))
    response = SimpleNamespace(completion="keep", content=[{"type": "text", "text": 1}, {"type": "tool_use", "name": "ignored"}])
    safety._apply_completion_safety(None, response, "anthropic.chat")
    assert response.completion == "keep"

    response = SimpleNamespace(completion="keep", content="not-a-list")
    assert safety._apply_completion_safety(None, response, "anthropic.chat") is None


def test_anthropic_message_only_change_path(monkeypatch):
    def _prompt(**kwargs):
        return SafetyResult(text=kwargs["text"], overall_action="MASK") if kwargs["text"] == "keep" else SafetyResult(text="masked:secret", overall_action="MASK")

    monkeypatch.setattr(safety, "run_prompt_safety", _prompt)
    kwargs = {"messages": [{"role": "user", "content": "secret"}]}
    updated = safety._apply_prompt_safety(None, kwargs, "anthropic.chat")
    assert updated["messages"][0]["content"] == "masked:secret"


def test_anthropic_request_type_and_resolve_masked_text():
    assert safety._request_type("anthropic.completion") == "completion"
    assert safety._request_type("anthropic.chat") == "chat"
    assert safety._resolve_masked_text("x", None) == ("x", False)
    assert safety._resolve_masked_text(
        "x",
        SafetyResult(text="x", overall_action=SafetyDecision.MASK.value),
    ) == ("x", False)
    assert safety._resolve_masked_text(
        "x",
        SafetyResult(text="y", overall_action=SafetyDecision.ALLOW.value),
    ) == ("x", False)
    assert safety._resolve_masked_text(
        "x",
        SafetyResult(text="y", overall_action=SafetyDecision.MASK.value),
    ) == ("y", True)


def test_anthropic_fail_opens_on_internal_error(monkeypatch):
    kwargs = {"messages": [{"role": "user", "content": "secret"}]}
    monkeypatch.setattr(safety, "_request_type", lambda *_: (_ for _ in ()).throw(RuntimeError("boom")))
    assert safety._apply_prompt_safety(None, kwargs, "anthropic.chat") is kwargs

    response = SimpleNamespace(completion="secret")
    monkeypatch.setattr(safety, "_request_type", lambda *_: (_ for _ in ()).throw(RuntimeError("boom")))
    assert safety._apply_completion_safety(None, response, "anthropic.chat") is None
