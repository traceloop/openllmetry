from types import SimpleNamespace

import pytest
from opentelemetry.instrumentation.vertexai import _awrap, _wrap
from opentelemetry.instrumentation.vertexai.config import Config
from opentelemetry.instrumentation.vertexai.utils import TRACELOOP_TRACE_CONTENT
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from vertexai.generative_models import GenerationResponse

MODEL = "gemini-2.5-flash"
TO_WRAP = {"span_name": "vertexai.generate_content"}
FINISH_REASONS = GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS


@pytest.fixture(autouse=True)
def legacy_attributes(monkeypatch):
    monkeypatch.setattr(Config, "use_legacy_attributes", True)


def _instance():
    return SimpleNamespace(_model_name=f"publishers/google/models/{MODEL}")


def _response(*finish_reasons, text="ok"):
    candidates = []
    for reason in finish_reasons:
        candidate = {"content": {"role": "model", "parts": [{"text": text}]}}
        if reason is not None:
            candidate["finish_reason"] = reason
        candidates.append(candidate)
    return GenerationResponse.from_dict({"candidates": candidates})


def _finished_attributes(span_exporter):
    (finished,) = span_exporter.get_finished_spans()
    return finished.attributes


def _wrapper(tracer_provider, wrap):
    return wrap(tracer_provider.get_tracer(__name__), None, TO_WRAP)


def _consume_sync(tracer_provider, chunks):
    wrapper = _wrapper(tracer_provider, _wrap)

    def stream():
        yield from chunks

    list(wrapper(lambda *a, **k: stream(), _instance(), ("hi",), {}))


async def _consume_async(tracer_provider, chunks):
    wrapper = _wrapper(tracer_provider, _awrap)

    async def stream():
        for chunk in chunks:
            yield chunk

    async def wrapped(*args, **kwargs):
        return stream()

    result = await wrapper(wrapped, _instance(), ("hi",), {})
    async for _ in result:
        pass


@pytest.mark.parametrize(
    "vertex_reason,expected",
    [
        ("STOP", "stop"),
        ("MAX_TOKENS", "length"),
        ("SAFETY", "content_filter"),
        ("RECITATION", "content_filter"),
        ("BLOCKLIST", "content_filter"),
        ("PROHIBITED_CONTENT", "content_filter"),
        ("SPII", "content_filter"),
        ("MODEL_ARMOR", "content_filter"),
        ("OTHER", "error"),
        ("MALFORMED_FUNCTION_CALL", "error"),
    ],
)
def test_finish_reasons_sync_mapped(tracer_provider, span_exporter, vertex_reason, expected):
    wrapper = _wrapper(tracer_provider, _wrap)

    wrapper(lambda *a, **k: _response(vertex_reason), _instance(), ("hi",), {})

    assert list(_finished_attributes(span_exporter)[FINISH_REASONS]) == [expected]


def test_finish_reasons_multiple_candidates(tracer_provider, span_exporter):
    wrapper = _wrapper(tracer_provider, _wrap)

    wrapper(lambda *a, **k: _response("STOP", "MAX_TOKENS", "SAFETY"), _instance(), ("hi",), {})

    attrs = _finished_attributes(span_exporter)
    assert list(attrs[FINISH_REASONS]) == ["stop", "length", "content_filter"]


def test_finish_reasons_sync_without_content(tracer_provider, span_exporter, monkeypatch):
    monkeypatch.setenv(TRACELOOP_TRACE_CONTENT, "false")
    wrapper = _wrapper(tracer_provider, _wrap)

    wrapper(lambda *a, **k: _response("STOP"), _instance(), ("hi",), {})

    attrs = _finished_attributes(span_exporter)
    assert list(attrs[FINISH_REASONS]) == ["stop"]
    assert f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content" not in attrs


def test_finish_reasons_missing_not_recorded(tracer_provider, span_exporter):
    wrapper = _wrapper(tracer_provider, _wrap)

    wrapper(lambda *a, **k: _response(None), _instance(), ("hi",), {})

    assert FINISH_REASONS not in _finished_attributes(span_exporter)


def test_finish_reasons_unspecified_not_recorded(tracer_provider, span_exporter):
    wrapper = _wrapper(tracer_provider, _wrap)

    wrapper(lambda *a, **k: _response("FINISH_REASON_UNSPECIFIED"), _instance(), ("hi",), {})

    assert FINISH_REASONS not in _finished_attributes(span_exporter)


async def test_finish_reasons_async(tracer_provider, span_exporter):
    wrapper = _wrapper(tracer_provider, _awrap)

    async def wrapped(*args, **kwargs):
        return _response("MAX_TOKENS")

    await wrapper(wrapped, _instance(), ("hi",), {})

    assert list(_finished_attributes(span_exporter)[FINISH_REASONS]) == ["length"]


def test_finish_reasons_streaming(tracer_provider, span_exporter):
    _consume_sync(tracer_provider, [_response(None, text="a"), _response("STOP", text="b")])

    assert list(_finished_attributes(span_exporter)[FINISH_REASONS]) == ["stop"]


def test_finish_reasons_streaming_reason_not_on_last_chunk(tracer_provider, span_exporter):
    _consume_sync(tracer_provider, [_response("STOP", text="a"), _response(None, text="b")])

    assert list(_finished_attributes(span_exporter)[FINISH_REASONS]) == ["stop"]


def test_finish_reasons_streaming_unspecified_last_chunk(tracer_provider, span_exporter):
    chunks = [_response("MAX_TOKENS", text="a"), _response("FINISH_REASON_UNSPECIFIED", text="b")]

    _consume_sync(tracer_provider, chunks)

    assert list(_finished_attributes(span_exporter)[FINISH_REASONS]) == ["length"]


async def test_finish_reasons_async_streaming(tracer_provider, span_exporter):
    await _consume_async(tracer_provider, [_response(None, text="a"), _response("MAX_TOKENS", text="b")])

    assert list(_finished_attributes(span_exporter)[FINISH_REASONS]) == ["length"]


async def test_finish_reasons_async_streaming_reason_not_on_last_chunk(tracer_provider, span_exporter):
    await _consume_async(tracer_provider, [_response("STOP", text="a"), _response(None, text="b")])

    assert list(_finished_attributes(span_exporter)[FINISH_REASONS]) == ["stop"]


def test_get_finish_reasons_skips_none_and_unmapped_values():
    from types import SimpleNamespace

    from opentelemetry.instrumentation.vertexai import get_finish_reasons

    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(finish_reason=None),
            SimpleNamespace(finish_reason=99),
            SimpleNamespace(finish_reason=SimpleNamespace(name="IMAGE_SAFETY")),
            SimpleNamespace(finish_reason=SimpleNamespace(name="MAX_TOKENS")),
        ]
    )
    assert get_finish_reasons(response) == ["length"]
