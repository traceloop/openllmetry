"""Tests for gen_ai.request.seed, gen_ai.request.stop_sequences and
gen_ai.output.type on the OpenAI instrumentation.

These three sampling parameters were not recorded at all, so a trace could not
reproduce the request it came from (determinism, output format and stop
behaviour were all lost). See
https://github.com/traceloop/openllmetry/issues/4538

Every test runs against an in-process httpx.MockTransport, so no API key and no
recorded cassette is needed.
"""

import httpx
import pydantic
import pytest
from openai import AsyncOpenAI, OpenAI
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


class _Profile(pydantic.BaseModel):
    name: str
    age: int


def _chat_completion_body(content="{}"):
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4.1-nano",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=_chat_completion_body())


def _client():
    return OpenAI(
        api_key="test-key",
        base_url="http://mock.invalid/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(_handler)),
    )


def _async_client():
    return AsyncOpenAI(
        api_key="test-key",
        base_url="http://mock.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
    )


def _attrs(span_exporter: InMemorySpanExporter):
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"expected exactly one span, got {len(spans)}"
    return spans[0].attributes


def test_seed_stop_sequences_and_output_type_are_recorded(
    instrument_legacy, span_exporter: InMemorySpanExporter
):
    """The issue's repro: all three attributes must be present alongside
    temperature, which was already recorded."""
    _client().chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "hi"}],
        seed=4242,
        stop=["ZZSTOP"],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "S", "schema": {"type": "object"}},
        },
        temperature=0.37,
    )

    attributes = _attrs(span_exporter)
    assert attributes["gen_ai.request.seed"] == 4242
    assert attributes["gen_ai.request.stop_sequences"] == ("ZZSTOP",)
    assert attributes["gen_ai.output.type"] == "json"
    # temperature must keep working alongside the new attributes.
    assert attributes["gen_ai.request.temperature"] == 0.37


@pytest.mark.asyncio
async def test_seed_stop_sequences_and_output_type_are_recorded_async(
    instrument_legacy, span_exporter: InMemorySpanExporter
):
    await _async_client().chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "hi"}],
        seed=7,
        stop=["A", "B"],
        response_format={"type": "json_object"},
    )

    attributes = _attrs(span_exporter)
    assert attributes["gen_ai.request.seed"] == 7
    assert attributes["gen_ai.request.stop_sequences"] == ("A", "B")
    assert attributes["gen_ai.output.type"] == "json"


def test_stop_accepts_a_bare_string(
    instrument_legacy, span_exporter: InMemorySpanExporter
):
    """The SDK allows `stop="END"` as well as `stop=["END"]`. The semconv
    attribute is a sequence either way."""
    _client().chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "hi"}],
        stop="END",
    )

    assert _attrs(span_exporter)["gen_ai.request.stop_sequences"] == ("END",)


def test_seed_of_zero_is_recorded(
    instrument_legacy, span_exporter: InMemorySpanExporter
):
    """seed=0 is falsy but meaningful; it must not be dropped."""
    _client().chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "hi"}],
        seed=0,
    )

    assert _attrs(span_exporter)["gen_ai.request.seed"] == 0


def test_attributes_absent_when_not_requested(
    instrument_legacy, span_exporter: InMemorySpanExporter
):
    """Conditionally Required attributes must stay off the span entirely when
    the caller did not set them, rather than being emitted as None."""
    _client().chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "hi"}],
    )

    attributes = _attrs(span_exporter)
    assert "gen_ai.request.seed" not in attributes
    assert "gen_ai.request.stop_sequences" not in attributes
    assert "gen_ai.output.type" not in attributes


@pytest.mark.parametrize(
    "response_format,expected",
    [
        ({"type": "text"}, "text"),
        ({"type": "json_object"}, "json"),
        (
            {
                "type": "json_schema",
                "json_schema": {"name": "S", "schema": {"type": "object"}},
            },
            "json",
        ),
    ],
)
def test_output_type_maps_response_format_variants(
    response_format, expected, instrument_legacy, span_exporter: InMemorySpanExporter
):
    _client().chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "hi"}],
        response_format=response_format,
    )

    assert _attrs(span_exporter)["gen_ai.output.type"] == expected


def test_output_type_for_pydantic_model(
    instrument_legacy, span_exporter: InMemorySpanExporter
):
    """A pydantic model is sent as a JSON schema by the SDK, so it maps to json."""
    _client().chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "hi"}],
        response_format=_Profile,
    )

    assert _attrs(span_exporter)["gen_ai.output.type"] == "json"