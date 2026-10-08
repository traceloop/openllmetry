from contextlib import contextmanager
from unittest.mock import AsyncMock, Mock, patch

import dspy
import pytest
from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import suppress_instrumentation
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAIAttributes
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY

from opentelemetry.instrumentation.dspy import instrumentation
from .test_dspy_instrumentation import _fake_litellm_result


@contextmanager
def _context_value(key, value):
    token = context_api.attach(context_api.set_value(key, value))
    try:
        yield
    finally:
        context_api.detach(token)


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("input_form", ["messages", "positional_messages", "prompt", "positional_prompt"])
@pytest.mark.parametrize(
    "env_value,override,record_content",
    [
        (None, None, True),
        ("", None, True),
        ("TrUe", False, True),
        ("false", None, False),
        ("FALSE", False, False),
        ("false", True, True),
        ("true", False, True),
    ],
)
async def test_content_policy(monkeypatch, span_exporter, is_async, input_form, env_value, override, record_content):
    if env_value is None:
        monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
    else:
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", env_value)
    lm = dspy.LM("openai/gpt-4o", cache=False)
    fake = _fake_litellm_result(content="synthetic output", cache_hit=True)
    messages = [{"role": "user", "content": "synthetic input"}]
    args, kwargs = {
        "messages": ((), {"messages": messages}),
        "positional_messages": ((None, messages), {}),
        "prompt": ((), {"prompt": "synthetic input"}),
        "positional_prompt": (("synthetic input",), {}),
    }[input_form]
    completion = AsyncMock(return_value=fake) if is_async else Mock(return_value=fake)
    target = "alitellm_completion" if is_async else "litellm_completion"
    with (
        _context_value("override_enable_content_tracing", override),
        patch(f"dspy.clients.lm.{target}", completion),
        patch.object(
            instrumentation, "messages_to_otel_input", wraps=instrumentation.messages_to_otel_input
        ) as encode_input,
        patch.object(
            instrumentation, "response_to_otel_output", wraps=instrumentation.response_to_otel_output
        ) as encode_output,
    ):
        result = await lm.aforward(*args, **kwargs) if is_async else lm.forward(*args, **kwargs)
    assert result is fake
    completion.assert_called_once()
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code.name == "OK"
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "openai/gpt-4o"
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "openai/gpt-4o"
    assert span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "openai"
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 10
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 5
    assert span.attributes["dspy.cache_hit"] is True
    if record_content:
        assert "synthetic input" in span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
        assert "synthetic output" in span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
        encode_input.assert_called_once()
        encode_output.assert_called_once()
    else:
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes
        encode_input.assert_not_called()
        encode_output.assert_not_called()


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_content_opt_out_preserves_errors(monkeypatch, span_exporter, is_async):
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    lm = dspy.LM("openai/gpt-4o", cache=False)
    error = RuntimeError("synthetic failure")
    completion = AsyncMock(side_effect=error) if is_async else Mock(side_effect=error)
    target = "alitellm_completion" if is_async else "litellm_completion"
    with _context_value("override_enable_content_tracing", False), patch(f"dspy.clients.lm.{target}", completion):
        with pytest.raises(RuntimeError) as caught:
            if is_async:
                await lm.aforward(prompt="synthetic input")
            else:
                lm.forward(prompt="synthetic input")
    assert caught.value is error
    completion.assert_called_once()
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code.name == "ERROR"
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in spans[0].attributes
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in spans[0].attributes


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("kind", ["lm", "predict"])
@pytest.mark.parametrize("raises", [False, True], ids=["return", "error"])
async def test_suppressed_wrappers_are_transparent(is_async, kind, raises):
    tracer, duration, tokens = Mock(), Mock(), Mock()
    result, arg = object(), object()
    error = RuntimeError("synthetic failure")
    wrapped = AsyncMock(return_value=result) if is_async else Mock(return_value=result)
    if raises:
        wrapped.side_effect = error
    name = f"wrap_{kind}_{'aforward' if is_async else 'forward'}"
    factory = getattr(instrumentation, name)
    wrapper = factory(tracer, duration, tokens) if kind == "lm" else factory(tracer)

    class UnreadableInstance:
        def __getattribute__(self, name):
            raise AssertionError(f"suppressed wrapper inspected {name}")

    async def invoke():
        value = wrapper(wrapped, UnreadableInstance(), (arg,), {"option": arg})
        return await value if is_async else value

    with suppress_instrumentation():
        if raises:
            with pytest.raises(RuntimeError) as caught:
                await invoke()
            assert caught.value is error
        else:
            assert await invoke() is result
    wrapped.assert_called_once_with(arg, option=arg)
    if is_async:
        wrapped.assert_awaited_once_with(arg, option=arg)
    assert tracer.mock_calls == []
    assert duration.mock_calls == []
    assert tokens.mock_calls == []


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("kind", ["lm", "predict"])
@pytest.mark.parametrize("suppression", ["all", "language_model"])
async def test_dspy_suppression_and_restoration(span_exporter, is_async, kind, suppression):
    lm = dspy.LM("openai/gpt-4o", cache=False)
    fake = _fake_litellm_result(content="[[ ## answer ## ]]\nsynthetic answer\n\n[[ ## completed ## ]]")
    completion = AsyncMock(return_value=fake) if is_async else Mock(return_value=fake)
    target = "alitellm_completion" if is_async else "litellm_completion"
    predict = dspy.Predict("question -> answer")

    async def invoke():
        if kind == "predict":
            return (
                await predict.acall(question="synthetic question")
                if is_async
                else predict(question="synthetic question")
            )
        return await lm.aforward(prompt="synthetic question") if is_async else lm.forward(prompt="synthetic question")

    context = (
        suppress_instrumentation()
        if suppression == "all"
        else _context_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True)
    )
    with dspy.settings.context(lm=lm), patch(f"dspy.clients.lm.{target}", completion):
        with context:
            result = await invoke()
        assert result.answer == "synthetic answer" if kind == "predict" else result is fake
        suppressed_spans = span_exporter.get_finished_spans()
        if suppression == "language_model" and kind == "predict":
            assert len(suppressed_spans) == 1
            assert suppressed_spans[0].name.endswith(".predict")
        else:
            assert len(suppressed_spans) == 0
        span_exporter.clear()
        await invoke()
    assert completion.call_count == 2
    spans = span_exporter.get_finished_spans()
    assert len(spans) == (2 if kind == "predict" else 1)
    lm_span = next(span for span in spans if span.name == "chat openai/gpt-4o")
    assert lm_span.status.status_code.name == "OK"
    if kind == "predict":
        predict_span = next(span for span in spans if span.name.endswith(".predict"))
        assert predict_span.attributes["dspy.signature"]
        assert lm_span.parent.span_id == predict_span.context.span_id
