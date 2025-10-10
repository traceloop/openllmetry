import hashlib
import json
from typing import Any, Dict

import pytest

from opentelemetry.semconv_ai import SpanAttributes

import opentelemetry.instrumentation.openai.shared as shared
from opentelemetry.instrumentation.openai.shared.chat_wrappers import (
    _accumulate_stream_items,
    _set_prompts as set_chat_prompts,
)
from opentelemetry.instrumentation.openai.shared.completion_wrappers import (
    _accumulate_streaming_response,
    _set_prompts as set_completion_prompts,
)


class RecordingSpan:
    def __init__(self) -> None:
        self.attributes: Dict[str, Any] = {}

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def is_recording(self) -> bool:
        return True

@pytest.fixture(autouse=True)
def reset_shared_helpers(monkeypatch):
    monkeypatch.setattr(shared, "_set_api_attributes", lambda *args, **kwargs: None)
    monkeypatch.setattr(shared, "_get_openai_base_url", lambda instance=None: "")
    monkeypatch.setattr(shared, "_get_vendor_from_url", lambda base_url: "openai")
    monkeypatch.setattr(shared, "_cross_region_check", lambda value: value)
    monkeypatch.setattr(shared, "_extract_model_name_from_provider_format", lambda value: value)
    monkeypatch.setattr(
        "opentelemetry.instrumentation.openai.shared.is_openai_v1", lambda: False
    )
    yield


def test_accumulate_stream_items_preserves_model(monkeypatch):
    monkeypatch.setattr(
        "opentelemetry.instrumentation.openai.shared.chat_wrappers.is_openai_v1",
        lambda: False,
    )

    complete = {"choices": [], "model": "", "id": ""}
    first = {
        "model": "gpt-4o",
        "id": "resp-123",
        "usage": None,
        "choices": [
            {"index": 0, "delta": {"content": "Hello"}, "content_filter_results": None},
        ],
    }
    second = {
        "choices": [
            {"index": 0, "delta": {"content": " there"}, "content_filter_results": None},
        ],
    }

    _accumulate_stream_items(first, complete)
    _accumulate_stream_items(second, complete)

    assert complete["model"] == "gpt-4o"
    assert complete["id"] == "resp-123"
    assert complete["choices"][0]["message"]["content"] == "Hello there"


@pytest.mark.asyncio
async def test_set_prompts_truncates_and_hashes():
    span = RecordingSpan()
    long_prompt = "A" * 200

    await set_chat_prompts(span, [{"role": "user", "content": long_prompt}])

    preview = span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
    digest = span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.hash"]

    assert len(preview) == 128
    assert preview == long_prompt[:128]
    assert digest == hashlib.sha256(long_prompt.encode("utf-8")).hexdigest()


def test_set_completion_prompts_hashes_each_prompt():
    span = RecordingSpan()
    prompts = ["foo", "bar"]

    set_completion_prompts(span, prompts)

    first_hash = span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.hash"]
    second_hash = span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.hash"]

    assert first_hash == hashlib.sha256("foo".encode("utf-8")).hexdigest()
    assert second_hash == hashlib.sha256("bar".encode("utf-8")).hexdigest()


def test_request_headers_are_scrubbed_and_sorted():
    span = RecordingSpan()
    kwargs = {
        "model": "gpt-4",
        "headers": {"Authorization": "Bearer 123", "X-Request-Id": "abc"},
        "seed": 42,
    }

    shared._set_request_attributes(span, kwargs)

    headers_json = span.attributes[SpanAttributes.LLM_HEADERS]
    headers = json.loads(headers_json)

    assert headers["Authorization"] == "<redacted>"
    assert headers["X-Request-Id"] == "abc"
    assert span.attributes[SpanAttributes.LLM_REPRODUCIBLE_RUN] is True


def test_response_receipt_id_is_set():
    span = RecordingSpan()
    response = {"id": "resp-456", "model": "gpt-4-turbo"}

    shared._set_response_attributes(span, response)

    assert span.attributes[SpanAttributes.LLM_RECEIPT_ID] == "resp-456"
    assert span.attributes[SpanAttributes.LLM_RESPONSE_MODEL] == "gpt-4-turbo"


def test_accumulate_streaming_response_preserves_model():
    complete = {"choices": [], "model": "", "id": ""}
    first = {
        "model": "text-davinci",
        "id": "resp-789",
        "choices": [{"index": 0, "text": "Hi"}],
    }
    second = {"choices": [{"index": 0, "text": "!*"}]}

    _accumulate_streaming_response(complete, first)
    _accumulate_streaming_response(complete, second)

    assert complete["model"] == "text-davinci"
    assert complete["id"] == "resp-789"
    assert complete["choices"][0]["text"] == "Hi!*"
