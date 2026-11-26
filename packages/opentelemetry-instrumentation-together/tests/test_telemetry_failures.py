"""Failures in the telemetry pipeline (samplers, span processors) must not change
what the app sees: the provider's response is returned and the provider's
exceptions propagate unchanged."""

import asyncio

import pytest
from opentelemetry.instrumentation.together import WRAPPED_METHODS, _awrap, _Telemetry, _wrap
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import Sampler

CHAT_METHOD = next(m for m in WRAPPED_METHODS if m["object"] == "ChatCompletions")
KWARGS = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
RESPONSE = {
    "id": "r",
    "model": "m",
    "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "hello"}}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


class TelemetryError(Exception):
    pass


class ProviderError(Exception):
    pass


class RaisingOnEndProcessor(SpanProcessor):
    def on_end(self, span):
        raise TelemetryError("span processor failed")


class RaisingSampler(Sampler):
    def should_sample(self, *args, **kwargs):
        raise TelemetryError("sampler failed")

    def get_description(self):
        return "RaisingSampler"


def _call(is_async, tracer_provider, provider_call):
    """Runs the together wrapper around `provider_call`, sync or async."""
    telemetry = _Telemetry(tracer_provider.get_tracer(__name__), None, None, None, None)

    if not is_async:
        return _wrap(telemetry, CHAT_METHOD)(provider_call, None, (), dict(KWARGS))

    async def async_provider_call(*args, **kwargs):
        return provider_call(*args, **kwargs)

    return asyncio.run(_awrap(telemetry, CHAT_METHOD)(async_provider_call, None, (), dict(KWARGS)))


def _respond(*args, **kwargs):
    return RESPONSE


def _fail(*args, **kwargs):
    raise ProviderError("provider failed")


@pytest.fixture(params=["sync", "async"])
def is_async(request):
    return request.param == "async"


def test_failing_span_processor_does_not_break_a_successful_call(is_async):
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(RaisingOnEndProcessor())
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    assert _call(is_async, tracer_provider, _respond) is RESPONSE


def test_failing_span_processor_does_not_replace_the_provider_error(is_async):
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(RaisingOnEndProcessor())

    with pytest.raises(ProviderError, match="provider failed"):
        _call(is_async, tracer_provider, _fail)


def test_failing_sampler_does_not_break_the_call(is_async):
    tracer_provider = TracerProvider(sampler=RaisingSampler())
    calls = []

    def provider_call(*args, **kwargs):
        calls.append(kwargs)
        return RESPONSE

    assert _call(is_async, tracer_provider, provider_call) is RESPONSE
    # The provider is called exactly once, with the app's arguments.
    assert calls == [KWARGS]


def test_failing_sampler_does_not_replace_the_provider_error(is_async):
    tracer_provider = TracerProvider(sampler=RaisingSampler())

    with pytest.raises(ProviderError, match="provider failed"):
        _call(is_async, tracer_provider, _fail)
