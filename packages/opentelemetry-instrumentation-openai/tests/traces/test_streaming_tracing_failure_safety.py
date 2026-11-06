"""
Regression test for the @dont_throw / generator gap.

_build_from_streaming_response (and its async twin) are generator functions.
@dont_throw only wraps the instant they're *called*, which just creates a
paused generator object and runs none of the body -- it never protects the
`for item in response:` loop, which only actually executes later, when the
caller iterates via `for chunk in stream:`. Before this fix, a real bug in
the tracing bookkeeping (_accumulate_stream_items) during that loop was
uncaught and crashed straight into the caller's own streaming loop, silently
dropping every remaining chunk of the real response -- despite the function
being decorated with @dont_throw.

This test forces exactly that: one chunk in the middle of the stream is
missing its "choices" key, which makes `for choice in item.get("choices"):`
inside _accumulate_stream_items raise TypeError (`NoneType not iterable`).
"""

import logging
from unittest.mock import MagicMock

from opentelemetry.instrumentation.openai.shared.chat_wrappers import (
    _build_from_streaming_response,
)


def test_tracing_failure_on_one_chunk_does_not_drop_later_chunks(monkeypatch, caplog):
    # Skip the openai-v1 model_as_dict(...) conversion inside
    # _accumulate_stream_items -- our fake chunks are plain dicts already.
    monkeypatch.setattr(
        "opentelemetry.instrumentation.openai.shared.chat_wrappers.is_openai_v1",
        lambda: False,
    )

    good_chunk_1 = {
        "id": "c1",
        "model": "gpt-4",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}}],
    }
    # Malformed: no "choices" key at all -> item.get("choices") is None ->
    # `for choice in None:` raises TypeError inside _accumulate_stream_items.
    malformed_chunk = {"id": "c2", "model": "gpt-4"}
    good_chunk_3 = {
        "id": "c3",
        "model": "gpt-4",
        "choices": [{"index": 0, "delta": {"content": "lo"}}],
    }

    fake_response = [good_chunk_1, malformed_chunk, good_chunk_3]
    fake_span = MagicMock()

    gen = _build_from_streaming_response(fake_span, fake_response)

    with caplog.at_level(logging.WARNING):
        yielded = list(gen)

    # The actual point of the fix: nothing gets dropped just because tracing
    # broke on one chunk. All three real chunks still reach the caller, in order.
    assert yielded == [good_chunk_1, malformed_chunk, good_chunk_3]

    # The failure was logged, not silently swallowed and not left to crash.
    assert any(
        "failed to trace a streaming chunk" in record.getMessage()
        for record in caplog.records
    )
