"""Nesting / suppression tests.

Proves that while a litellm call is being traced, nested LLM calls do not produce
duplicate spans. This covers both real-world nesting cases:

* litellm internally delegating (e.g. ``acompletion`` -> ``completion`` for sync-only
  providers), and
* the Azure/OpenAI-routed internal provider-SDK call (the OpenAI instrumentor honors
  the same ``SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY``).

Done offline by having a custom provider re-enter ``litellm.completion``; if suppression
works, only the outer span is emitted.
"""

import litellm
import pytest
from litellm import CustomLLM, ModelResponse
from litellm.types.utils import Choices, Message, Usage

PROVIDER = "nestprov"
MESSAGES = [{"role": "user", "content": "ping"}]


class _ReentrantCustomLLM(CustomLLM):
    """A provider whose completion re-enters litellm (like a real router would)."""

    def completion(self, *args, **kwargs) -> ModelResponse:
        # Nested litellm call — must NOT create its own span while suppressed.
        litellm.completion(
            model="gpt-3.5-turbo",
            messages=MESSAGES,
            mock_response="nested inner response",
        )
        return ModelResponse(
            id="chatcmpl-nested-1",
            model=kwargs.get("model", f"{PROVIDER}/test"),
            choices=[
                Choices(
                    index=0,
                    finish_reason="stop",
                    message=Message(role="assistant", content="outer response"),
                )
            ],
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


@pytest.fixture
def reentrant_llm():
    previous = litellm.custom_provider_map
    litellm.custom_provider_map = [
        {"provider": PROVIDER, "custom_handler": _ReentrantCustomLLM()}
    ]
    yield
    litellm.custom_provider_map = previous


def test_nested_call_emits_single_span(instrument_legacy, span_exporter, reentrant_llm):
    litellm.completion(model=f"{PROVIDER}/test", messages=MESSAGES)

    spans = span_exporter.get_finished_spans()
    # Exactly one span despite the nested litellm.completion inside the provider.
    assert [span.name for span in spans] == ["litellm.chat"]


def test_both_openai_and_litellm_instrumented(span_exporter, tracer_provider, meter_provider):
    """Instrumenting both must not raise and must not double-count."""
    pytest.importorskip("opentelemetry.instrumentation.openai")
    from opentelemetry.instrumentation.litellm import LiteLLMInstrumentor
    from opentelemetry.instrumentation.openai import OpenAIInstrumentor

    openai_instrumentor = OpenAIInstrumentor()
    litellm_instrumentor = LiteLLMInstrumentor()
    try:
        openai_instrumentor.instrument(
            tracer_provider=tracer_provider, meter_provider=meter_provider
        )
        litellm_instrumentor.instrument(
            tracer_provider=tracer_provider, meter_provider=meter_provider
        )
        litellm.completion(
            model="gpt-3.5-turbo",
            messages=MESSAGES,
            mock_response="hello",
        )
        spans = span_exporter.get_finished_spans()
        assert [span.name for span in spans] == ["litellm.chat"]
    finally:
        litellm_instrumentor.uninstrument()
        openai_instrumentor.uninstrument()
