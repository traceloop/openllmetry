"""LM telemetry regressions with normal pytest-asyncio scheduling."""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from opentelemetry.instrumentation.dspy import instrumentation as inst

from .test_dspy_instrumentation import _fake_litellm_result


class BrokenHandler(logging.Handler):
    def emit(self, record):
        raise RuntimeError("synthetic logging failure")


@pytest.fixture
def broken_debug_logger():
    logger = inst.logger
    handler = BrokenHandler()
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("outcome", ["success", "error"])
@pytest.mark.parametrize("failure", ["none", "input", "duration", "both"])
async def test_lm_telemetry_failures_preserve_application(
    tracer_provider, span_exporter, monkeypatch, broken_debug_logger, is_async, outcome, failure
):
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
    duration, tokens = Mock(), Mock()
    if failure in ("duration", "both"):
        duration.record.side_effect = RuntimeError("synthetic metrics failure")
    fake = _fake_litellm_result(model="response-model")
    error = ValueError("synthetic application failure")
    calls = []

    def application(**kwargs):
        calls.append(kwargs)
        if outcome == "error":
            raise error
        return fake

    async def async_application(**kwargs):
        # A real yield, not a manually driven coroutine body.
        await asyncio.sleep(0)
        return application(**kwargs)

    wrapper = (inst.wrap_lm_aforward if is_async else inst.wrap_lm_forward)(
        tracer_provider.get_tracer(__name__), duration, tokens
    )
    encode = inst.messages_to_otel_input
    with patch.object(inst, "messages_to_otel_input", wraps=encode) as encode_input:
        if failure in ("input", "both"):
            encode_input.side_effect = TypeError("synthetic serialization failure")

        async def invoke():
            value = wrapper(
                async_application if is_async else application,
                SimpleNamespace(model="openai/request-model"),
                (),
                {"prompt": "synthetic input"},
            )
            return await value if is_async else value

        if outcome == "error":
            with pytest.raises(ValueError) as caught:
                await invoke()
            assert caught.value is error
        else:
            assert await invoke() is fake
    assert calls == [{"prompt": "synthetic input"}]
    duration.record.assert_called_once()
    args, kwargs = duration.record.call_args
    assert args[0] >= 0
    assert kwargs["attributes"]["gen_ai.provider.name"] == "openai"
    assert kwargs["attributes"]["gen_ai.response.model"] == (
        "response-model" if outcome == "success" else "openai/request-model"
    )
    if outcome == "error":
        tokens.record.assert_not_called()
    else:
        assert tokens.record.call_count == 2
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code.name == ("ERROR" if outcome == "error" else "OK")


@pytest.mark.parametrize("broken_duration", [False, True])
async def test_task_cancellation_survives_telemetry_failure(
    tracer_provider, monkeypatch, broken_debug_logger, broken_duration
):
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
    duration, tokens = Mock(), Mock()
    if broken_duration:
        duration.record.side_effect = RuntimeError("synthetic metrics failure")
    entered = asyncio.Event()
    cleaned = asyncio.Event()

    async def application(**kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    wrapper = inst.wrap_lm_aforward(tracer_provider.get_tracer(__name__), duration, tokens)
    with patch.object(inst, "messages_to_otel_input", side_effect=TypeError("synthetic input failure")):
        task = asyncio.create_task(wrapper(application, SimpleNamespace(model="openai/gpt-4o"), (), {"prompt": "x"}))
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            task.cancel("synthetic cancellation")
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            assert caught.value.args == ("synthetic cancellation",)
            assert cleaned.is_set()
            assert task.cancelled()
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    duration.record.assert_called_once()
    tokens.record.assert_not_called()
