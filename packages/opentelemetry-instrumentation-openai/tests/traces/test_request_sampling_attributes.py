"""Tests for gen_ai.request.seed, gen_ai.request.stop_sequences and
gen_ai.output.type on the OpenAI instrumentation.

These three sampling parameters were not recorded at all, so a trace could not
reproduce the request it came from (determinism, output format and stop
behaviour were all lost). See
https://github.com/traceloop/openllmetry/issues/4538

Every test runs against an in-process httpx.MockTransport, so no API key and no
recorded cassette is needed.
"""

from unittest.mock import MagicMock

import httpx
import pydantic
import pytest
from openai import AsyncOpenAI, OpenAI
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from opentelemetry.instrumentation.openai.shared import _set_request_attributes
from opentelemetry.instrumentation.openai.v1.responses_wrappers import (
    prepare_kwargs_for_shared_attributes,
)


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


def _client(content="{}"):
    """Build a client whose transport always answers with `content`.

    chat.completions.parse() needs the mocked message to actually satisfy the
    pydantic model, so the body has to be configurable per test.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_completion_body(content))

    return OpenAI(
        api_key="test-key",
        base_url="http://mock.invalid/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
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
    """A pydantic model passed as response_format is the output schema itself, so
    it maps to json.

    Uses chat.completions.parse(), which is the SDK-supported way to pass a model
    class as response_format (chat.completions.create() does not accept one).
    Completions.parse is instrumented, so the span is still produced.
    """
    _client(content='{"name": "Ada", "age": 36}').chat.completions.parse(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "hi"}],
        response_format=_Profile,
    )

    assert _attrs(span_exporter)["gen_ai.output.type"] == "json"


def _mock_span():
    """Minimal recording-span stand-in, following tests/traces/test_semconv_compliance.py."""
    span = MagicMock()
    span.is_recording.return_value = True
    attributes = {}
    span.set_attribute = lambda name, value: attributes.__setitem__(name, value)
    span.attributes = attributes
    return span


def _responses_request_attributes(**kwargs):
    """Run the Responses API request kwargs through the shared attribute setter.

    The Responses API declares the output format as text.format rather than
    response_format, so this asserts on the attribute setter directly instead of
    standing up a full mocked HTTP round-trip for responses.create().
    """
    span = _mock_span()
    _set_request_attributes(span, prepare_kwargs_for_shared_attributes(dict(kwargs)))
    return span.attributes


def test_responses_api_text_format_json_schema_sets_output_type():
    """responses.create(text={"format": {"type": "json_schema", ...}}) requests JSON
    output, but has no response_format parameter, so it must be read from text.format.

    Note the Responses API nests the schema flat rather than under a "json_schema"
    key, which is why this is not routed through the response_format branch.
    """
    attributes = _responses_request_attributes(
        model="gpt-4.1-nano",
        input="hi",
        text={
            "format": {
                "type": "json_schema",
                "strict": True,
                "name": "Person",
                "schema": {"type": "object"},
            }
        },
    )

    assert attributes["gen_ai.output.type"] == "json"


def test_responses_api_text_format_text_sets_output_type():
    assert (
        _responses_request_attributes(
            model="gpt-4.1-nano", input="hi", text={"format": {"type": "text"}}
        )["gen_ai.output.type"]
        == "text"
    )


def test_responses_api_without_text_format_leaves_output_type_unset():
    """A Responses call with no text.format must not gain an output.type attribute."""
    attributes = _responses_request_attributes(model="gpt-4.1-nano", input="hi")

    assert "gen_ai.output.type" not in attributes


def test_responses_api_structured_output_schema_is_not_faked():
    """Guard against regressing gen_ai.request.structured_output_schema.

    The Responses API's flat format shape does not carry a nested "json_schema"
    key, so the structured-output-schema branch must leave the attribute unset
    rather than recording a placeholder schema derived from the dict itself.
    """
    attributes = _responses_request_attributes(
        model="gpt-4.1-nano",
        input="hi",
        text={
            "format": {
                "type": "json_schema",
                "strict": True,
                "name": "Person",
                "schema": {"type": "object"},
            }
        },
    )

    assert "gen_ai.request.structured_output_schema" not in attributes


def test_schema_model_with_a_format_field_is_still_reported_as_json():
    """A user schema passed to parse() may legitimately have a field named `format`.

    Only the declared `type` is consulted, so such a model must still map to
    json rather than being probed for a nested format declaration.
    """

    class _ConfigWithFormat(pydantic.BaseModel):
        format: str
        strict: bool

    span = _mock_span()
    _set_request_attributes(
        span,
        {
            "model": "gpt-4.1-nano",
            "response_format": _ConfigWithFormat,
        },
    )

    assert span.attributes["gen_ai.output.type"] == "json"
