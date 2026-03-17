from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult, Generation, LLMResult
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from opentelemetry.instrumentation.fortifyroot import (
    SafetyFinding,
    SafetyLocation,
    SafetyResult,
    clear_safety_handlers,
    register_completion_safety_handler,
    register_prompt_safety_handler,
)
from opentelemetry.instrumentation.langchain.safety import (
    _apply_chat_prompt_safety,
    _apply_chat_result_completion_safety,
    _apply_llm_prompt_safety,
    _apply_llm_result_completion_safety,
    _content_text,
    _message_role,
    _set_content_text,
    base_chat_model_agenerate_with_cache_wrapper,
    base_chat_model_agenerate_wrapper,
    base_chat_model_generate_with_cache_wrapper,
    base_chat_model_generate_wrapper,
    base_llm_agenerate_helper_wrapper,
    base_llm_agenerate_wrapper,
    base_llm_generate_helper_wrapper,
    base_llm_generate_wrapper,
    instrument_safety_wrappers,
    uninstrument_safety_wrappers,
    _provider_name,
    _resolve_masked_text,
)

pytestmark = pytest.mark.safety


class _FakeChatModel:
    pass


class _FakeLLM:
    pass


def setup_function():
    clear_safety_handlers()


def teardown_function():
    clear_safety_handlers()


def test_chat_prompt_safety_masks_messages_before_wrapped_call():
    captured = {}
    register_prompt_safety_handler(
        lambda context: SafetyResult(
            text="[PII.langchain]",
            overall_action="MASK",
            findings=[
                SafetyFinding(
                    category="PII",
                    severity="MEDIUM",
                    action="MASK",
                    rule_name="PII.langchain",
                    start=0,
                    end=len(context.text),
                )
            ],
        )
        if context.location == SafetyLocation.PROMPT and context.text == "secret"
        else None
    )

    def wrapped(messages, **kwargs):
        captured["messages"] = messages
        return "ok"

    messages = [[HumanMessage(content="secret")]]
    result = base_chat_model_generate_wrapper(
        wrapped,
        _FakeChatModel(),
        (messages,),
        {},
    )

    assert result == "ok"
    assert captured["messages"][0][0].content == "[PII.langchain]"
    assert messages[0][0].content == "secret"


def test_llm_prompt_safety_masks_prompts_before_wrapped_call():
    captured = {}
    register_prompt_safety_handler(
        lambda context: SafetyResult(
            text="[PII.langchain]",
            overall_action="MASK",
            findings=[
                SafetyFinding(
                    category="PII",
                    severity="MEDIUM",
                    action="MASK",
                    rule_name="PII.langchain",
                    start=0,
                    end=len(context.text),
                )
            ],
        )
        if context.location == SafetyLocation.PROMPT and context.text == "secret"
        else None
    )

    def wrapped(prompts, *args, **kwargs):
        captured["prompts"] = prompts
        return "ok"

    result = base_llm_generate_wrapper(
        wrapped,
        _FakeLLM(),
        (["secret"], None, None),
        {},
    )

    assert result == "ok"
    assert captured["prompts"] == ["[PII.langchain]"]


def test_chat_completion_safety_masks_chat_result_before_callbacks():
    register_completion_safety_handler(
        lambda context: SafetyResult(
            text="[SECRET.langchain]",
            overall_action="MASK",
            findings=[
                SafetyFinding(
                    category="SECRET",
                    severity="HIGH",
                    action="MASK",
                    rule_name="SECRET.langchain",
                    start=0,
                    end=len(context.text),
                )
            ],
        )
        if context.location == SafetyLocation.COMPLETION and context.text == "secret"
        else None
    )

    def wrapped(*args, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="secret"))])

    response = base_chat_model_generate_with_cache_wrapper(
        wrapped,
        _FakeChatModel(),
        ([],),
        {},
    )

    assert response.generations[0].message.content == "[SECRET.langchain]"
    assert response.generations[0].text == "[SECRET.langchain]"


def test_llm_completion_safety_masks_llm_result_before_callbacks():
    register_completion_safety_handler(
        lambda context: SafetyResult(
            text="[SECRET.langchain]",
            overall_action="MASK",
            findings=[
                SafetyFinding(
                    category="SECRET",
                    severity="HIGH",
                    action="MASK",
                    rule_name="SECRET.langchain",
                    start=0,
                    end=len(context.text),
                )
            ],
        )
        if context.location == SafetyLocation.COMPLETION and context.text == "secret"
        else None
    )

    def wrapped(*args, **kwargs):
        return LLMResult(generations=[[Generation(text="secret")]])

    response = base_llm_generate_helper_wrapper(
        wrapped,
        _FakeLLM(),
        (["prompt"], None, []),
        {},
    )

    assert response.generations[0][0].text == "[SECRET.langchain]"


@pytest.mark.asyncio
async def test_async_wrappers_cover_kwargs_and_cache_paths():
    register_prompt_safety_handler(
        lambda context: SafetyResult(
            text=f"[MASKED:{context.text}]",
            overall_action="MASK",
            findings=[SafetyFinding("PII", "MEDIUM", "MASK", "PII.langchain", 0, len(context.text))],
        )
        if context.location == SafetyLocation.PROMPT
        else None
    )
    register_completion_safety_handler(
        lambda context: SafetyResult(
            text=f"[MASKED:{context.text}]",
            overall_action="MASK",
            findings=[SafetyFinding("SECRET", "HIGH", "MASK", "SECRET.langchain", 0, len(context.text))],
        )
        if context.location == SafetyLocation.COMPLETION
        else None
    )

    async def wrapped_chat(messages, **kwargs):
        return messages

    async def wrapped_cache(*args, **kwargs):
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content=[{"type": "text", "text": "secret-block"}]))
            ]
        )

    async def wrapped_llm(prompts, *args, **kwargs):
        return prompts

    async def wrapped_llm_cache(*args, **kwargs):
        return LLMResult(generations=[[Generation(text="secret-llm")]])

    chat_messages = [[HumanMessage(content=[{"type": "text", "text": "secret-block"}])]]
    masked_messages = await base_chat_model_agenerate_wrapper(
        wrapped_chat,
        _FakeChatModel(),
        (),
        {"messages": chat_messages},
    )
    chat_response = await base_chat_model_agenerate_with_cache_wrapper(
        wrapped_cache,
        _FakeChatModel(),
        ([],),
        {},
    )
    masked_prompts = await base_llm_agenerate_wrapper(
        wrapped_llm,
        _FakeLLM(),
        (),
        {"prompts": ["secret-llm"]},
    )
    llm_response = await base_llm_agenerate_helper_wrapper(
        wrapped_llm_cache,
        _FakeLLM(),
        (["prompt"], None, []),
        {},
    )

    assert masked_messages[0][0].content[0]["text"] == "[MASKED:secret-block]"
    assert chat_response.generations[0].message.content[0]["text"] == "[MASKED:secret-block]"
    assert chat_response.generations[0].text == "[MASKED:secret-block]"
    assert masked_prompts == ["[MASKED:secret-llm]"]
    assert llm_response.generations[0][0].text == "[MASKED:secret-llm]"


def test_registration_and_helper_fallbacks():
    with patch("opentelemetry.instrumentation.langchain.safety.wrap_function_wrapper") as wrap_mock, patch(
        "opentelemetry.instrumentation.langchain.safety.unwrap"
    ) as unwrap_mock:
        instrument_safety_wrappers()
        uninstrument_safety_wrappers()

    assert wrap_mock.call_count == 8
    assert unwrap_mock.call_count == 8
    assert _provider_name(_FakeChatModel()) == "Langchain"
    assert _resolve_masked_text("same", None) == ("same", False)
    assert _resolve_masked_text("same", SafetyResult(text="same", overall_action="MASK", findings=[])) == (
        "same",
        False,
    )


def test_internal_helpers_cover_kwargs_passthrough_and_block_updates():
    register_prompt_safety_handler(
        lambda context: SafetyResult(
            text=f"[MASKED:{context.text}]",
            overall_action="MASK",
            findings=[SafetyFinding("PII", "MEDIUM", "MASK", "PII.langchain", 0, len(context.text))],
        )
        if context.location == SafetyLocation.PROMPT
        else None
    )
    register_completion_safety_handler(
        lambda context: SafetyResult(
            text=f"[MASKED:{context.text}]",
            overall_action="MASK",
            findings=[SafetyFinding("SECRET", "HIGH", "MASK", "SECRET.langchain", 0, len(context.text))],
        )
        if context.location == SafetyLocation.COMPLETION
        else None
    )

    chat_messages = [[HumanMessage(content=[{"type": "text", "text": "secret-block"}, "secret-inline"])]]
    _, updated_kwargs = _apply_chat_prompt_safety(_FakeChatModel(), (), {"messages": chat_messages})
    assert updated_kwargs["messages"][0][0].content[0]["text"] == "[MASKED:secret-block]"
    assert updated_kwargs["messages"][0][0].content[1] == "[MASKED:secret-inline]"
    assert _apply_chat_prompt_safety(_FakeChatModel(), (), {"messages": "invalid"}) == ((), {"messages": "invalid"})

    _, updated_llm_kwargs = _apply_llm_prompt_safety(_FakeLLM(), (), {"prompts": ["secret", 1]})
    assert updated_llm_kwargs["prompts"][0] == "[MASKED:secret]"
    assert _apply_llm_prompt_safety(_FakeLLM(), (), {"prompts": "invalid"}) == ((), {"prompts": "invalid"})

    chat_response = ChatResult(
        generations=[
            ChatGeneration(message=AIMessage(content=[{"type": "text", "text": "secret-block"}])),
            ChatGeneration(message=AIMessage(content="secret-text"), text="secret-text"),
        ]
    )
    _apply_chat_result_completion_safety(_FakeChatModel(), chat_response)
    assert chat_response.generations[0].message.content[0]["text"] == "[MASKED:secret-block]"
    assert chat_response.generations[0].text == "[MASKED:secret-block]"
    assert chat_response.generations[1].text == "[MASKED:secret-text]"

    llm_response = SimpleNamespace(
        generations=[
            [SimpleNamespace(text="secret-text", message=SimpleNamespace(content="secret-text"))],
            "ignore",
        ]
    )
    _apply_llm_result_completion_safety(_FakeLLM(), llm_response)
    assert llm_response.generations[0][0].text == "[MASKED:secret-text]"
    assert llm_response.generations[0][0].message.content == "[MASKED:secret-text]"

    assert _message_role(SimpleNamespace(type="system")) == "system"
    assert _message_role(SimpleNamespace(type="ai")) == "assistant"
    assert _message_role(SimpleNamespace(type="tool")) == "tool"
    block = {"type": "text", "text": "value"}
    assert _content_text(block) == "value"
    assert _set_content_text(block, "new") is True
    assert block["text"] == "new"
