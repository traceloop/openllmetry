from types import SimpleNamespace

import pytest

from opentelemetry.instrumentation.openai.shared import chat_safety, completion_safety, safety_common
from opentelemetry.instrumentation.fortifyroot import SafetyDecision, SafetyResult

pytestmark = pytest.mark.safety


def test_chat_prompt_safety_masks_string_and_block_content(monkeypatch):
    def _mask(span, text, **kwargs):
        return (f"masked:{text}", True)

    monkeypatch.setattr(chat_safety, "mask_prompt_text", _mask)

    kwargs = {
        "messages": [
            {"role": "user", "content": "secret"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "secret-2"},
                    {"type": "image", "image": "ignored"},
                ],
            },
        ]
    }

    updated = chat_safety._apply_prompt_safety(None, kwargs)

    assert kwargs["messages"][0]["content"] == "secret"
    assert updated["messages"][0]["content"] == "masked:secret"
    assert updated["messages"][1]["content"][0]["text"] == "masked:secret-2"
    assert updated["messages"][1]["content"][1]["image"] == "ignored"


def test_chat_prompt_safety_returns_original_when_messages_are_not_a_list(monkeypatch):
    monkeypatch.setattr(chat_safety, "mask_prompt_text", lambda *args, **kwargs: ("x", True))
    kwargs = {"messages": "secret"}
    assert chat_safety._apply_prompt_safety(None, kwargs) is kwargs


def test_chat_prompt_safety_skips_unchanged_messages(monkeypatch):
    def _mask(span, text, **kwargs):
        return (text, False) if text == "keep" else (f"masked:{text}", True)

    monkeypatch.setattr(chat_safety, "mask_prompt_text", _mask)
    kwargs = {"messages": [{"role": "user", "content": "keep"}, {"role": "user", "content": "secret"}]}
    updated = chat_safety._apply_prompt_safety(None, kwargs)

    assert updated["messages"][0]["content"] == "keep"
    assert updated["messages"][1]["content"] == "masked:secret"


def test_chat_completion_safety_masks_string_and_output_blocks(monkeypatch):
    monkeypatch.setattr(chat_safety, "mask_completion_text", lambda span, text, **kwargs: (f"done:{text}", True))
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content="secret")),
            SimpleNamespace(
                message=SimpleNamespace(
                    content=[
                        {"type": "output_text", "text": "secret-2"},
                        {"type": "tool", "text": "ignored"},
                    ]
                )
            ),
        ]
    )

    chat_safety._apply_completion_safety(None, response)

    assert response.choices[0].message.content == "done:secret"
    assert response.choices[1].message.content[0]["text"] == "done:secret-2"
    assert response.choices[1].message.content[1]["text"] == "ignored"


def test_chat_completion_safety_ignores_empty_choices_and_missing_message(monkeypatch):
    monkeypatch.setattr(chat_safety, "mask_completion_text", lambda span, text, **kwargs: (text, False))
    assert chat_safety._apply_completion_safety(None, SimpleNamespace(choices=[])) is None
    response = SimpleNamespace(choices=[SimpleNamespace(message=None)])
    assert chat_safety._apply_completion_safety(None, response) is None


def test_chat_and_completion_content_helpers_cover_non_list_and_unchanged(monkeypatch):
    monkeypatch.setattr(chat_safety, "mask_prompt_text", lambda span, text, **kwargs: (text, False))
    assert chat_safety._mask_prompt_content(None, {"x": 1}, message_index=0, message_role="user") == ({"x": 1}, False)

    content = [{"type": "text", "text": "keep"}]
    updated, changed = chat_safety._mask_prompt_content(None, content, message_index=0, message_role="user")
    assert updated == content
    assert changed is False

    monkeypatch.setattr(chat_safety, "mask_completion_text", lambda span, text, **kwargs: (text, False))
    assert chat_safety._mask_completion_content(None, {"x": 1}, choice_index=0) == ({"x": 1}, False)
    updated, changed = chat_safety._mask_completion_content(None, [{"type": "text", "text": "keep"}], choice_index=0)
    assert changed is False
    assert updated[0]["text"] == "keep"


def test_completion_prompt_safety_masks_string_and_list(monkeypatch):
    monkeypatch.setattr(completion_safety, "mask_prompt_text", lambda span, text, **kwargs: (f"masked:{text}", True))

    assert completion_safety._apply_prompt_safety(None, {"prompt": "secret"})["prompt"] == "masked:secret"
    updated = completion_safety._apply_prompt_safety(None, {"prompt": ["a", "b"]})
    assert updated["prompt"] == ["masked:a", "masked:b"]
    same = {"prompt": {"unsupported": True}}
    assert completion_safety._apply_prompt_safety(None, same) is same


def test_completion_prompt_safety_returns_original_for_unchanged_or_non_string_items(monkeypatch):
    monkeypatch.setattr(completion_safety, "mask_prompt_text", lambda span, text, **kwargs: (text, False))
    kwargs = {"prompt": "keep"}
    assert completion_safety._apply_prompt_safety(None, kwargs) is kwargs
    kwargs = {"prompt": [1, "keep"]}
    assert completion_safety._apply_prompt_safety(None, kwargs) is kwargs


def test_completion_safety_masks_choice_text(monkeypatch):
    monkeypatch.setattr(completion_safety, "mask_completion_text", lambda span, text, **kwargs: (f"done:{text}", True))
    response = SimpleNamespace(choices=[SimpleNamespace(text="secret"), SimpleNamespace(text=None)])
    completion_safety._apply_completion_safety(None, response)
    assert response.choices[0].text == "done:secret"


def test_completion_safety_skips_non_text_choices_and_fail_opens():
    response = SimpleNamespace(choices=[SimpleNamespace(text=None)])
    assert completion_safety._apply_completion_safety(None, response) is None

    class BrokenKwargs:
        def get(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    broken = BrokenKwargs()
    assert completion_safety._apply_prompt_safety(None, broken) is broken


def test_safety_common_request_type_and_resolve_masked_text():
    assert safety_common.request_type(safety_common.CHAT_SPAN_NAME) == "chat"
    assert safety_common.request_type(safety_common.COMPLETION_SPAN_NAME) == "completion"

    assert safety_common.resolve_masked_text("x", None) == ("x", False)
    assert safety_common.resolve_masked_text(
        "x",
        SafetyResult(text="x", findings=(), overall_action=SafetyDecision.MASK.value),
    ) == ("x", False)
    assert safety_common.resolve_masked_text(
        "x",
        SafetyResult(text="y", findings=(), overall_action=SafetyDecision.ALLOW.value),
    ) == ("x", False)
    assert safety_common.resolve_masked_text(
        "x",
        SafetyResult(text="y", findings=(), overall_action=SafetyDecision.MASK.value),
    ) == ("y", True)


def test_openai_safety_helpers_fail_open_on_internal_error(monkeypatch):
    monkeypatch.setattr(chat_safety, "get_object_value", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    kwargs = {"messages": [{"role": "user", "content": "secret"}]}
    assert chat_safety._apply_prompt_safety(None, kwargs) is kwargs

    monkeypatch.setattr(completion_safety, "get_object_value", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    response = SimpleNamespace(choices=[SimpleNamespace(text="secret")])
    assert completion_safety._apply_completion_safety(None, response) is None
