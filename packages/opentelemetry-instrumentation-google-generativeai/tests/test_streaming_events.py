"""Streaming builders must emit choice events from the last chunk, not the drained generator."""

from unittest.mock import MagicMock

import pytest
from opentelemetry.instrumentation.google_generativeai import (
    _abuild_from_streaming_response,
    _build_from_streaming_response,
)


class _Part:
    def __init__(self, text):
        self.text = text


class _Content:
    role = "model"

    def __init__(self, text):
        self.parts = [_Part(text)]


class _Candidate:
    finish_reason = None

    def __init__(self, text):
        self.content = _Content(text)


class _Chunk:
    """Stands in for a GenerateContentResponse chunk."""

    def __init__(self, text):
        self.text = text
        self.candidates = [_Candidate(text)]
        self.usage_metadata = None


def _streaming_setup(monkeypatch):
    seen = {}

    def fake_emit_choice_events(response, event_logger):
        seen["response"] = response

    monkeypatch.setattr(
        "opentelemetry.instrumentation.google_generativeai.emit_choice_events",
        fake_emit_choice_events,
    )
    monkeypatch.setattr(
        "opentelemetry.instrumentation.google_generativeai.should_emit_events",
        lambda: True,
    )
    monkeypatch.setattr(
        "opentelemetry.instrumentation.google_generativeai.set_model_response_attributes",
        lambda *a, **k: None,
    )
    return seen


def test_sync_streaming_emits_choice_events_from_last_chunk(monkeypatch):
    seen = _streaming_setup(monkeypatch)
    chunks = [_Chunk("hello "), _Chunk("world")]

    generator = _build_from_streaming_response(
        MagicMock(), iter(chunks), "gemini-2.0-flash", MagicMock(), None
    )
    list(generator)

    assert seen["response"] is chunks[-1], (
        "choice events must come from the last chunk, the generator has been drained"
    )


@pytest.mark.asyncio
async def test_async_streaming_emits_choice_events_from_last_chunk(monkeypatch):
    seen = _streaming_setup(monkeypatch)
    chunks = [_Chunk("hello "), _Chunk("world")]

    async def agen():
        for c in chunks:
            yield c

    generator = _abuild_from_streaming_response(
        MagicMock(), agen(), "gemini-2.0-flash", MagicMock(), None
    )
    async for _ in generator:
        pass

    assert seen["response"] is chunks[-1]


def test_sync_streaming_with_no_chunks_does_not_emit(monkeypatch):
    seen = _streaming_setup(monkeypatch)

    generator = _build_from_streaming_response(
        MagicMock(), iter([]), "gemini-2.0-flash", MagicMock(), None
    )
    list(generator)

    assert "response" not in seen, "an empty stream has no candidates to emit"
