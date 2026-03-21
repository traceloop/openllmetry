from types import SimpleNamespace

import pytest

from opentelemetry.instrumentation.cohere import safety
from opentelemetry.instrumentation.fortifyroot import SafetyDecision, SafetyResult
from opentelemetry.semconv_ai import LLMRequestTypeValues

pytestmark = pytest.mark.fr


def test_apply_prompt_safety_masks_chat_fields(monkeypatch):
    monkeypatch.setattr(safety, "run_prompt_safety", lambda **kwargs: SafetyResult(text=f"masked:{kwargs['text']}", overall_action="MASK"))
    kwargs = {
        "preamble": "pre",
        "message": "msg",
        "messages": [{"role": "user", "content": "thread"}],
    }

    updated = safety._apply_prompt_safety(None, kwargs, LLMRequestTypeValues.CHAT, "cohere.chat")

    assert kwargs["message"] == "msg"
    assert updated["preamble"] == "masked:pre"
    assert updated["message"] == "masked:msg"
    assert updated["messages"][0]["content"] == "masked:thread"


def test_apply_prompt_safety_masks_completion_prompt(monkeypatch):
    monkeypatch.setattr(safety, "run_prompt_safety", lambda **kwargs: SafetyResult(text=f"masked:{kwargs['text']}", overall_action="MASK"))
    updated = safety._apply_prompt_safety(None, {"prompt": "secret"}, LLMRequestTypeValues.COMPLETION, "cohere.completion")
    assert updated["prompt"] == "masked:secret"


def test_apply_prompt_safety_returns_original_for_unchanged_completion_prompt(monkeypatch):
    monkeypatch.setattr(safety, "run_prompt_safety", lambda **kwargs: SafetyResult(text=kwargs["text"], overall_action="MASK"))
    kwargs = {"prompt": "keep"}
    assert safety._apply_prompt_safety(None, kwargs, LLMRequestTypeValues.COMPLETION, "cohere.completion") is kwargs


def test_apply_prompt_safety_returns_original_for_non_string_completion_prompt():
    kwargs = {"prompt": 123}
    assert safety._apply_prompt_safety(None, kwargs, LLMRequestTypeValues.COMPLETION, "cohere.completion") is kwargs


def test_apply_prompt_safety_ignores_unsupported_type():
    kwargs = {"prompt": "secret"}
    assert safety._apply_prompt_safety(None, kwargs, SimpleNamespace(value="embed"), "cohere.embed") is kwargs


def test_cohere_prompt_helper_and_noop_message_paths(monkeypatch):
    monkeypatch.setattr(safety, "run_prompt_safety", lambda **kwargs: SafetyResult(text=kwargs["text"], overall_action="MASK"))
    assert safety._mask_prompt_content(None, {"x": 1}, request_type="chat", span_name="cohere.chat", segment_index=0, segment_role="user") == ({"x": 1}, False)

    kwargs = {"preamble": "keep", "message": "keep", "messages": [{"role": "user", "content": "keep"}]}
    assert safety._apply_prompt_safety(None, kwargs, LLMRequestTypeValues.CHAT, "cohere.chat") is kwargs


def test_cohere_message_only_and_messages_only_prompt_paths(monkeypatch):
    def _prompt(**kwargs):
        if kwargs["text"] == "secret":
            return SafetyResult(text="masked:secret", overall_action="MASK")
        return SafetyResult(text=kwargs["text"], overall_action="MASK")

    monkeypatch.setattr(safety, "run_prompt_safety", _prompt)

    updated = safety._apply_prompt_safety(
        None,
        {"preamble": "keep", "message": "secret"},
        LLMRequestTypeValues.CHAT,
        "cohere.chat",
    )
    assert updated["message"] == "masked:secret"

    updated = safety._apply_prompt_safety(
        None,
        {"messages": [{"role": "user", "content": "secret"}]},
        LLMRequestTypeValues.CHAT,
        "cohere.chat",
    )
    assert updated["messages"][0]["content"] == "masked:secret"


def test_apply_completion_safety_masks_chat_and_completion_shapes(monkeypatch):
    monkeypatch.setattr(safety, "run_completion_safety", lambda **kwargs: SafetyResult(text=f"masked:{kwargs['text']}", overall_action="MASK"))

    chat_response = SimpleNamespace(
        text="chat-secret",
        message=SimpleNamespace(content=[{"text": "content-secret"}]),
    )
    safety._apply_completion_safety(None, chat_response, LLMRequestTypeValues.CHAT, "cohere.chat")
    assert chat_response.text == "masked:chat-secret"
    assert chat_response.message.content[0]["text"] == "masked:content-secret"

    generation = SimpleNamespace(text="gen-secret")
    safety._apply_completion_safety(None, [generation], LLMRequestTypeValues.COMPLETION, "cohere.completion")
    assert generation.text == "masked:gen-secret"


def test_apply_completion_safety_syncs_response_text_only_when_derived_from_message_content(monkeypatch):
    monkeypatch.setattr(
        safety,
        "run_completion_safety",
        lambda **kwargs: SafetyResult(text=f"masked:{kwargs['text']}", overall_action="MASK"),
    )

    chat_response = SimpleNamespace(
        text="content-secret",
        message=SimpleNamespace(content=[{"text": "content-secret"}]),
    )

    safety._apply_completion_safety(None, chat_response, LLMRequestTypeValues.CHAT, "cohere.chat")

    assert chat_response.text == "masked:content-secret"
    assert chat_response.message.content[0]["text"] == "masked:content-secret"


def test_cohere_completion_helper_noop_branches(monkeypatch):
    monkeypatch.setattr(safety, "run_completion_safety", lambda **kwargs: SafetyResult(text=kwargs["text"], overall_action="MASK"))
    assert safety._apply_completion_safety(None, SimpleNamespace(text="x"), SimpleNamespace(value="embed"), "cohere.embed") is None

    chat_response = SimpleNamespace(text="keep", message=SimpleNamespace(content="keep"))
    safety._apply_completion_safety(None, chat_response, LLMRequestTypeValues.CHAT, "cohere.chat")
    assert chat_response.text == "keep"
    assert chat_response.message.content == "keep"

    assert safety._mask_completion_content(None, {"x": 1}, request_type="chat", span_name="cohere.chat", segment_index=0) == ({"x": 1}, False)
    updated, changed = safety._mask_completion_content(
        None,
        [{"text": "keep"}, {"binary": b"x"}],
        request_type="chat",
        span_name="cohere.chat",
        segment_index=0,
    )
    assert changed is False
    assert updated[0]["text"] == "keep"

    response = SimpleNamespace(generations=[SimpleNamespace(text=None), SimpleNamespace(text="keep")])
    safety._apply_completion_safety(None, response, LLMRequestTypeValues.COMPLETION, "cohere.completion")
    assert response.generations[1].text == "keep"

    monkeypatch.setattr(safety, "get_object_value", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    assert safety._apply_completion_safety(None, SimpleNamespace(text="secret"), LLMRequestTypeValues.COMPLETION, "cohere.completion") is None


def test_cohere_mask_completion_content_changes_list(monkeypatch):
    monkeypatch.setattr(safety, "run_completion_safety", lambda **kwargs: SafetyResult(text=f"masked:{kwargs['text']}", overall_action="MASK"))
    updated, changed = safety._mask_completion_content(
        None,
        [{"text": "secret"}],
        request_type="chat",
        span_name="cohere.chat",
        segment_index=0,
    )
    assert changed is True
    assert updated[0]["text"] == "masked:secret"


def test_cohere_resolve_masked_text_and_fail_open(monkeypatch):
    assert safety._resolve_masked_text("x", None) == ("x", False)
    assert safety._resolve_masked_text("x", SafetyResult(text="x", overall_action=SafetyDecision.MASK.value)) == ("x", False)
    assert safety._resolve_masked_text("x", SafetyResult(text="y", overall_action=SafetyDecision.ALLOW.value)) == ("x", False)
    assert safety._resolve_masked_text("x", SafetyResult(text="y", overall_action=SafetyDecision.MASK.value)) == ("y", True)

    kwargs = {"messages": [{"role": "user", "content": "secret"}]}
    monkeypatch.setattr(safety, "get_object_value", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    assert safety._apply_prompt_safety(None, kwargs, LLMRequestTypeValues.CHAT, "cohere.chat") is kwargs
