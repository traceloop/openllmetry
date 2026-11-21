import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from opentelemetry.instrumentation.dspy import instrumentation as inst
from opentelemetry.sdk.trace import TracerProvider


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("failure", ["output", "tokens"])
async def test_output_telemetry_and_logger_failures_preserve_result(monkeypatch, is_async, failure):
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
    result = SimpleNamespace(
        model="openai/gpt-4o",
        choices=[],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        cache_hit=False,
    )
    duration, tokens = Mock(), Mock()
    if failure == "output":
        monkeypatch.setattr(
            inst, "response_to_otel_output", Mock(side_effect=ValueError("output serialization failed"))
        )
    else:
        tokens.record.side_effect = ValueError("token metrics failed")

    class BrokenHandler(logging.Handler):
        def emit(self, record):
            raise RuntimeError("logger failed")

    async def async_call(**kwargs):
        return result

    sync_call = Mock(return_value=result)
    logger = inst.logger
    handler = BrokenHandler()
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        factory = inst.wrap_lm_aforward if is_async else inst.wrap_lm_forward
        wrapper = factory(TracerProvider().get_tracer(__name__), duration, tokens)
        value = wrapper(async_call if is_async else sync_call, SimpleNamespace(model="openai/gpt-4o"), (), {})
        assert (await value if is_async else value) is result
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    duration.record.assert_called_once()
