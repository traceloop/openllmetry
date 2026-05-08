import json
from unittest.mock import MagicMock, patch

import dspy
import pytest
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAIAttributes
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GenAiSystemValues,
)
from opentelemetry.semconv_ai import GenAISystem


def _fake_litellm_result(model="openai/gpt-4o", prompt_tokens=10, completion_tokens=5,
                         cache_hit=False, content="test response", tool_calls=None):
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    msg = MagicMock()
    msg.content = content
    msg.role = "assistant"
    msg.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    result = MagicMock()
    result.model = model
    result.usage = usage
    result.choices = [choice]
    result.cache_hit = cache_hit
    result.__getitem__ = lambda self, k: getattr(self, k)
    result.get = lambda k, d=None: getattr(result, k, d)
    return result


def test_lm_forward_chat_span(span_exporter):
    lm = dspy.LM("openai/gpt-4o", cache=False)
    fake = _fake_litellm_result()

    with patch("dspy.clients.lm.litellm_completion", return_value=fake):
        lm.forward(messages=[{"role": "user", "content": "hello"}])

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "chat openai/gpt-4o"
    assert span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "chat"
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "openai/gpt-4o"
    assert span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == GenAISystem.OPENAI.value
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 10
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 5
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES in span.attributes
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in span.attributes


def test_lm_forward_prompt_kwarg(span_exporter):
    lm = dspy.LM("openai/gpt-4o", cache=False)
    fake = _fake_litellm_result()

    with patch("dspy.clients.lm.litellm_completion", return_value=fake):
        lm.forward(prompt="hello")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES in spans[0].attributes


def test_lm_forward_cache_hit_zero_tokens(span_exporter):
    lm = dspy.LM("openai/gpt-4o", cache=False)
    fake = _fake_litellm_result(prompt_tokens=0, completion_tokens=0, cache_hit=True)

    with patch("dspy.clients.lm.litellm_completion", return_value=fake):
        lm.forward(messages=[{"role": "user", "content": "hello"}])

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes.get("dspy.cache_hit") is True
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 0
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 0


def test_lm_forward_exception_keeps_input_messages(span_exporter):
    """Input messages must be set before the wrapped call so they survive failures."""
    lm = dspy.LM("openai/gpt-4o", cache=False)

    with patch("dspy.clients.lm.litellm_completion", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            lm.forward(messages=[{"role": "user", "content": "hello"}])

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code.name == "ERROR"
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES in span.attributes
    payload = json.loads(span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    assert payload[0]["role"] == "user"


def test_lm_forward_tool_calls_in_output(span_exporter):
    lm = dspy.LM("openai/gpt-4o", cache=False)
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
    }
    fake = _fake_litellm_result(content=None, tool_calls=[tool_call])

    with patch("dspy.clients.lm.litellm_completion", return_value=fake):
        lm.forward(messages=[{"role": "user", "content": "weather?"}])

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    output = json.loads(spans[0].attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    parts = output[0]["parts"]
    tool_parts = [p for p in parts if p["type"] == "tool_call"]
    assert len(tool_parts) == 1
    assert tool_parts[0]["name"] == "get_weather"
    assert tool_parts[0]["arguments"] == {"city": "Paris"}


@pytest.mark.parametrize("model,expected_provider", [
    ("openai/gpt-4o",                     GenAISystem.OPENAI.value),
    ("anthropic/claude-3-5-sonnet",       GenAISystem.ANTHROPIC.value),
    ("bedrock/anthropic.claude-3-haiku",  GenAISystem.AWS.value),
    ("azure/gpt-4o",                      GenAiSystemValues.AZURE_AI_OPENAI.value),
    ("gemini/gemini-1.5-flash",           GenAiSystemValues.GCP_GEMINI.value),
    ("vertex_ai/gemini-pro",              GenAiSystemValues.GCP_VERTEX_AI.value),
    ("groq/llama-3",                      GenAISystem.GROQ.value),
    ("mistral/mistral-large",             GenAISystem.MISTRALAI.value),
    ("cohere/command-r",                  GenAISystem.COHERE.value),
    ("ollama/llama3",                     GenAISystem.OLLAMA.value),
    ("gpt-4o",                            GenAISystem.OPENAI.value),
    ("claude-3-5-sonnet",                 GenAISystem.ANTHROPIC.value),
])
def test_provider_inference_otel_spec_compliant(span_exporter, model, expected_provider):
    lm = dspy.LM(model, cache=False)
    fake = _fake_litellm_result(model=model)

    with patch("dspy.clients.lm.litellm_completion", return_value=fake):
        lm.forward(messages=[{"role": "user", "content": "hi"}])

    spans = span_exporter.get_finished_spans()
    assert spans[-1].attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == expected_provider


async def test_lm_aforward_chat_span(span_exporter):
    lm = dspy.LM("openai/gpt-4o", cache=False)
    fake = _fake_litellm_result()

    async def fake_acompletion(*a, **kw):
        return fake

    with patch("dspy.clients.lm.alitellm_completion", side_effect=fake_acompletion):
        await lm.aforward(messages=[{"role": "user", "content": "hello"}])

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "chat"
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 10


def test_predict_forward_span_name_and_no_chat_op(span_exporter):
    lm = dspy.LM("openai/gpt-4o", cache=False)
    fake = _fake_litellm_result()
    fake.choices[0].message.content = "[[ ## answer ## ]]\n4\n\n[[ ## completed ## ]]"

    with dspy.settings.context(lm=lm), patch("dspy.clients.lm.litellm_completion", return_value=fake):
        predict = dspy.Predict("question -> answer")
        predict(question="What is 2+2?")

    spans = span_exporter.get_finished_spans()
    predict_spans = [s for s in spans if s.name.endswith(".predict")]
    assert len(predict_spans) == 1
    span = predict_spans[0]
    assert span.status.status_code.name == "OK"
    assert span.attributes.get("dspy.signature") is not None
    # Predict is an INTERNAL task, not a chat operation — child LM span owns the chat op.
    assert GenAIAttributes.GEN_AI_OPERATION_NAME not in span.attributes


def test_predict_is_parent_of_lm_span(span_exporter):
    lm = dspy.LM("openai/gpt-4o", cache=False)
    fake = _fake_litellm_result()
    fake.choices[0].message.content = "[[ ## answer ## ]]\n4\n\n[[ ## completed ## ]]"

    with dspy.settings.context(lm=lm), patch("dspy.clients.lm.litellm_completion", return_value=fake):
        predict = dspy.Predict("question -> answer")
        predict(question="What is 2+2?")

    spans = span_exporter.get_finished_spans()
    predict_span = next(s for s in spans if s.name.endswith(".predict"))
    lm_span = next(s for s in spans if s.name == "chat openai/gpt-4o")
    assert lm_span.parent is not None
    assert lm_span.parent.span_id == predict_span.context.span_id


def test_instrumentor_reinstrument():
    from opentelemetry.instrumentation.dspy import DSPyInstrumentor
    i = DSPyInstrumentor()
    i.instrument()
    i.uninstrument()
    i.instrument()
    i.uninstrument()
