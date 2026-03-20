from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    MessageRole,
)
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMCompletionEndEvent,
    LLMCompletionStartEvent,
)
from types import SimpleNamespace
from unittest.mock import patch
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

import pytest

from opentelemetry.instrumentation.fortifyroot import (
    SafetyFinding,
    SafetyLocation,
    SafetyResult,
    clear_safety_handlers,
    register_completion_safety_handler,
    register_prompt_safety_handler,
)
from opentelemetry.instrumentation.llamaindex.safety import (
    _apply_chat_prompt_safety,
    _apply_completion_prompt_safety,
    _block_text,
    _set_block_text,
    apply_chat_end_safety,
    apply_completion_end_safety,
    apply_completion_start_attributes,
    apply_predict_end_safety,
    instrument_llm_safety_wrappers,
    llm_acomplete_wrapper,
    llm_achat_wrapper,
    llm_complete_wrapper,
    llm_chat_wrapper,
    _mask_chat_message,
    _resolve_masked_text,
)

pytestmark = pytest.mark.fr


class _FakeLLM:
    pass


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


def test_chat_prompt_safety_masks_messages_before_wrapped_call():
    captured = {}
    register_prompt_safety_handler(
        lambda context: SafetyResult(
            text="[PII.llamaindex]",
            overall_action="MASK",
            findings=[
                SafetyFinding(
                    category="PII",
                    severity="MEDIUM",
                    action="MASK",
                    rule_name="PII.llamaindex",
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

    messages = [ChatMessage(content="secret", role=MessageRole.USER)]
    result = llm_chat_wrapper(wrapped, _FakeLLM(), (messages,), {})

    assert result == "ok"
    assert captured["messages"][0].content == "[PII.llamaindex]"
    assert messages[0].content == "secret"


def test_completion_prompt_safety_masks_prompt_before_wrapped_call():
    captured = {}
    register_prompt_safety_handler(
        lambda context: SafetyResult(
            text="[PII.llamaindex]",
            overall_action="MASK",
            findings=[
                SafetyFinding(
                    category="PII",
                    severity="MEDIUM",
                    action="MASK",
                    rule_name="PII.llamaindex",
                    start=0,
                    end=len(context.text),
                )
            ],
        )
        if context.location == SafetyLocation.PROMPT and context.text == "secret"
        else None
    )

    def wrapped(prompt, **kwargs):
        captured["prompt"] = prompt
        return CompletionResponse(text="ok")

    llm_complete_wrapper(wrapped, _FakeLLM(), ("secret",), {})

    assert captured["prompt"] == "[PII.llamaindex]"


def test_completion_start_sets_masked_prompt_attributes():
    exporter, tracer = _test_span()

    with tracer.start_as_current_span("llamaindex.completion") as span:
        apply_completion_start_attributes(
            LLMCompletionStartEvent(
                prompt="[PII.llamaindex]",
                additional_kwargs={},
                model_dict={"model": "test-model", "temperature": 0.1},
                span_id="span-1",
            ),
            span,
        )

    spans = exporter.get_finished_spans()
    assert spans[0].attributes["gen_ai.prompt.0.content"] == "[PII.llamaindex]"
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"


def test_completion_end_safety_masks_response_and_span_attributes():
    exporter, tracer = _test_span()
    register_completion_safety_handler(
        lambda context: SafetyResult(
            text="[SECRET.llamaindex]",
            overall_action="MASK",
            findings=[
                SafetyFinding(
                    category="SECRET",
                    severity="HIGH",
                    action="MASK",
                    rule_name="SECRET.llamaindex",
                    start=0,
                    end=len(context.text),
                )
            ],
        )
        if context.location == SafetyLocation.COMPLETION and context.text == "secret"
        else None
    )

    response = CompletionResponse(text="secret")
    with tracer.start_as_current_span("llamaindex.completion") as span:
        apply_completion_end_safety(
            LLMCompletionEndEvent(prompt="prompt", response=response, span_id="span-1"),
            span,
        )

    spans = exporter.get_finished_spans()
    assert response.text == "[SECRET.llamaindex]"
    assert spans[0].attributes["gen_ai.completion.0.content"] == "[SECRET.llamaindex]"


@pytest.mark.asyncio
async def test_async_wrappers_and_chat_end_safety_cover_block_paths():
    exporter, tracer = _test_span()
    register_prompt_safety_handler(
        lambda context: SafetyResult(
            text=f"[MASKED:{context.text}]",
            overall_action="MASK",
            findings=[SafetyFinding("PII", "MEDIUM", "MASK", "PII.llamaindex", 0, len(context.text))],
        )
        if context.location == SafetyLocation.PROMPT
        else None
    )
    register_completion_safety_handler(
        lambda context: SafetyResult(
            text=f"[MASKED:{context.text}]",
            overall_action="MASK",
            findings=[SafetyFinding("SECRET", "HIGH", "MASK", "SECRET.llamaindex", 0, len(context.text))],
        )
        if context.location == SafetyLocation.COMPLETION
        else None
    )

    async def wrapped_messages(messages, **kwargs):
        return messages

    async def wrapped_prompt(prompt, **kwargs):
        return CompletionResponse(text="ok")

    messages = [
        SimpleNamespace(
            role=MessageRole.USER,
            blocks=[SimpleNamespace(block_type="text", text="secret")],
        )
    ]
    masked_messages = await llm_achat_wrapper(wrapped_messages, _FakeLLM(), (messages,), {})
    await llm_acomplete_wrapper(wrapped_prompt, _FakeLLM(), ("secret",), {})

    with tracer.start_as_current_span("llamaindex.chat") as span:
        _mask_chat_message(
            messages[0],
            span=span,
            span_name="llamaindex.chat",
            request_type="chat",
            segment_index=0,
            segment_role="assistant",
        )
        response = ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content="secret"))
        apply_chat_end_safety(
            LLMChatEndEvent(messages=[], response=response, span_id="span-1"),
            span,
        )

    assert masked_messages[0].blocks[0].text == "[MASKED:secret]"
    assert messages[0].blocks[0].text == "[MASKED:secret]"
    assert response.message.content == "[MASKED:secret]"
    assert exporter.get_finished_spans()


def test_predict_and_registration_helpers_cover_non_emit_paths():
    exporter, tracer = _test_span()
    register_completion_safety_handler(
        lambda context: SafetyResult(
            text=f"[MASKED:{context.text}]",
            overall_action="MASK",
            findings=[SafetyFinding("SECRET", "HIGH", "MASK", "SECRET.llamaindex", 0, len(context.text))],
        )
        if context.location == SafetyLocation.COMPLETION
        else None
    )

    event = SimpleNamespace(output="secret")
    with patch(
        "opentelemetry.instrumentation.llamaindex.safety.wrap_function_wrapper"
    ) as wrap_mock, patch(
        "opentelemetry.instrumentation.llamaindex.safety.pkgutil.iter_modules",
        return_value=[SimpleNamespace(name="fake_module")],
    ), patch(
        "opentelemetry.instrumentation.llamaindex.safety.importlib.import_module"
    ) as import_mock, patch(
        "opentelemetry.instrumentation.llamaindex.safety._WRAPPERS_INSTALLED", False
    ):
        fake_module = SimpleNamespace()

        class FakeWrappedLLM(_FakeLLM.__class__):
            pass

        class DummyLLM:
            def chat(self):  # pragma: no cover - signature only
                return None

            def complete(self):  # pragma: no cover - signature only
                return None

        import llama_index.core.base.llms.base as base_module

        DummyLLM = type(
            "DummyLLM",
            (base_module.BaseLLM,),
            {
                "__module__": "fake.module",
                "chat": lambda self: None,
                "complete": lambda self: None,
            },
        )
        fake_module.DummyLLM = DummyLLM
        import_mock.side_effect = [SimpleNamespace(__path__=["fake"]), fake_module, Exception("nope")]

        with tracer.start_as_current_span("llamaindex.predict") as span:
            apply_predict_end_safety(event, span)
        instrument_llm_safety_wrappers()

    assert event.output == "[MASKED:secret]"
    wrap_mock.assert_any_call(
        "llama_index.core.base.llms.base",
        "BaseLLM.chat",
        llm_chat_wrapper,
    )
    wrap_mock.assert_any_call(
        "llama_index.core.base.llms.base",
        "BaseLLM.complete",
        llm_complete_wrapper,
    )
    wrap_mock.assert_any_call(
        "llama_index.core.base.llms.base",
        "BaseLLM.achat",
        llm_achat_wrapper,
    )
    wrap_mock.assert_any_call(
        "llama_index.core.base.llms.base",
        "BaseLLM.acomplete",
        llm_acomplete_wrapper,
    )
    assert wrap_mock.call_count >= 2
    assert _resolve_masked_text("same", None) == ("same", False)


def test_internal_helpers_cover_kwargs_and_non_recording_paths():
    register_prompt_safety_handler(
        lambda context: SafetyResult(
            text=f"[MASKED:{context.text}]",
            overall_action="MASK",
            findings=[SafetyFinding("PII", "MEDIUM", "MASK", "PII.llamaindex", 0, len(context.text))],
        )
        if context.location == SafetyLocation.PROMPT
        else None
    )
    register_completion_safety_handler(
        lambda context: SafetyResult(
            text=f"[MASKED:{context.text}]",
            overall_action="MASK",
            findings=[SafetyFinding("SECRET", "HIGH", "MASK", "SECRET.llamaindex", 0, len(context.text))],
        )
        if context.location == SafetyLocation.COMPLETION
        else None
    )

    message = SimpleNamespace(
        role=MessageRole.USER,
        blocks=[SimpleNamespace(block_type="thinking", content="secret")],
    )
    _, updated_kwargs = _apply_chat_prompt_safety(_FakeLLM(), (), {"messages": [message]})
    assert updated_kwargs["messages"][0].blocks[0].content == "[MASKED:secret]"
    assert _apply_chat_prompt_safety(_FakeLLM(), (), {"messages": "invalid"}) == ((), {"messages": "invalid"})

    _, updated_prompt_kwargs = _apply_completion_prompt_safety(
        _FakeLLM(), (), {"prompt": "secret"}
    )
    assert updated_prompt_kwargs["prompt"] == "[MASKED:secret]"
    assert _apply_completion_prompt_safety(_FakeLLM(), (), {"prompt": 123}) == ((), {"prompt": 123})

    exporter, tracer = _test_span()
    with patch("opentelemetry.instrumentation.llamaindex.safety.should_send_prompts", return_value=False):
        with tracer.start_as_current_span("llamaindex.completion") as span:
            apply_completion_start_attributes(
                LLMCompletionStartEvent(
                    prompt="prompt",
                    additional_kwargs={},
                    model_dict={"llm": {"model": "test-model", "temperature": 0.1}},
                    span_id="span-2",
                ),
                span,
            )
            apply_completion_end_safety(
                LLMCompletionEndEvent(
                    prompt="prompt",
                    response=CompletionResponse(
                        text="secret",
                        raw={"model": "test-model", "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}},
                    ),
                    span_id="span-2",
                ),
                span,
            )

    spans = exporter.get_finished_spans()
    assert spans[0].attributes["gen_ai.response.model"] == "test-model"
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 1
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 2
    assert spans[0].attributes["llm.usage.total_tokens"] == 3
    thinking_block = SimpleNamespace(block_type="thinking", content="idea")
    assert _block_text(thinking_block) == "idea"
    assert _set_block_text(thinking_block, "updated") is True
    assert thinking_block.content == "updated"
