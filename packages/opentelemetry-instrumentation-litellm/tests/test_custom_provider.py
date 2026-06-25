"""Custom-provider tracing tests.

This is the key test for the agent-orchestrator use-case: it registers a
``litellm.CustomLLM`` (the same mechanism the orchestrator's GAIC→Mosaic provider
uses) and proves that a call routed through it is captured as a ``litellm.chat`` span
with the resolved provider name — without any network or credentials.

It also verifies that ``SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY`` is active in
context while the provider executes, which is what prevents a nested provider-SDK
instrumentor (e.g. OpenAI/Azure) from emitting a duplicate span.
"""

import litellm
import pytest
from litellm import CustomLLM, ModelResponse
from litellm.types.utils import Choices, Message, Usage
from opentelemetry import context as context_api
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    LLMRequestTypeValues,
    SpanAttributes,
)

PROVIDER = "myprov"
MESSAGES = [{"role": "user", "content": "ping"}]


def _build_response(model):
    return ModelResponse(
        id="chatcmpl-custom-1",
        model=model,
        choices=[
            Choices(
                index=0,
                finish_reason="stop",
                message=Message(role="assistant", content="pong from custom provider"),
            )
        ],
        usage=Usage(prompt_tokens=3, completion_tokens=4, total_tokens=7),
    )


class _MockCustomLLM(CustomLLM):
    def __init__(self):
        super().__init__()
        # Records whether nested instrumentation was suppressed during the call.
        self.suppress_during_call = "unset"

    def _record_suppression(self):
        self.suppress_during_call = context_api.get_value(
            SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
        )

    def completion(self, *args, **kwargs) -> ModelResponse:
        self._record_suppression()
        return _build_response(kwargs.get("model", f"{PROVIDER}/test"))

    async def acompletion(self, *args, **kwargs) -> ModelResponse:
        self._record_suppression()
        return _build_response(kwargs.get("model", f"{PROVIDER}/test"))


@pytest.fixture
def custom_llm():
    handler = _MockCustomLLM()
    previous = litellm.custom_provider_map
    litellm.custom_provider_map = [
        {"provider": PROVIDER, "custom_handler": handler}
    ]
    yield handler
    litellm.custom_provider_map = previous


def test_custom_provider_completion(instrument_legacy, span_exporter, custom_llm):
    litellm.completion(model=f"{PROVIDER}/test", messages=MESSAGES)

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["litellm.chat"]
    attrs = spans[0].attributes

    assert attrs[GenAIAttributes.GEN_AI_SYSTEM] == PROVIDER
    assert attrs[SpanAttributes.LLM_REQUEST_TYPE] == LLMRequestTypeValues.CHAT.value
    assert (
        attrs[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"]
        == "pong from custom provider"
    )
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 7

    # Suppression was active while the provider ran -> nested instrumentation skips.
    assert custom_llm.suppress_during_call is True


@pytest.mark.asyncio
async def test_custom_provider_acompletion(instrument_legacy, span_exporter, custom_llm):
    await litellm.acompletion(model=f"{PROVIDER}/test", messages=MESSAGES)

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["litellm.chat"]
    assert spans[0].attributes[GenAIAttributes.GEN_AI_SYSTEM] == PROVIDER
    assert custom_llm.suppress_during_call is True
