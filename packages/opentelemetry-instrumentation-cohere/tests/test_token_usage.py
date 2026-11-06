"""Regression test: embed spans must carry token usage.

Cohere's ApiMetaBilledUnits is a pydantic model whose unused counters are
present-but-None, so `.get("output_tokens", 0)` yields None rather than 0 and
`input_tokens + output_tokens` raises TypeError. set_span_response_attributes
is @dont_throw-wrapped, so the raise is swallowed and EVERY token attribute on
the span is lost, not merely the missing one.
"""

from unittest.mock import Mock

from cohere.types import (
    ApiMeta,
    ApiMetaBilledUnits,
    EmbedByTypeResponse,
    EmbedByTypeResponseEmbeddings,
)
from opentelemetry.instrumentation.cohere.span_utils import (
    set_span_response_attributes,
)
from opentelemetry.semconv_ai import SpanAttributes


def _recording_span():
    span = Mock()
    span.is_recording.return_value = True
    captured = {}
    span.set_attribute = lambda key, value: captured.__setitem__(key, value)
    return span, captured


def _embed_response(**billed):
    return EmbedByTypeResponse(
        id="embed-1",
        embeddings=EmbedByTypeResponseEmbeddings(),
        meta=ApiMeta(billed_units=ApiMetaBilledUnits(**billed)),
    )


def test_embed_response_records_prompt_tokens():
    """An embed reply bills input_tokens only; output_tokens comes back None."""
    span, captured = _recording_span()

    set_span_response_attributes(span, _embed_response(input_tokens=208))

    assert captured.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 208
    assert captured.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS) == 208


def test_response_with_both_counters_still_records():
    """Control: when both counters are present the attributes are recorded."""
    span, captured = _recording_span()

    set_span_response_attributes(
        span, _embed_response(input_tokens=7, output_tokens=88)
    )

    assert captured.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 7
    assert captured.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS) == 95


def test_embed_response_emits_a_choice_event(logger_provider, log_exporter, monkeypatch):
    """An EMBEDDING response must reach the exporter as a `gen_ai.choice` event.

    The branch was written `elif a == CHAT or COMPLETION`, whose right operand is a truthy enum
    member rather than a comparison, so every type that is not RERANK read `response.text`. The
    object below raises if it is touched. With the branch fixed, the event carries a message without
    a role, and `_emit_choice_event` must still emit it rather than raise KeyError.
    """
    from opentelemetry.instrumentation.cohere.config import Config
    from opentelemetry.instrumentation.cohere.event_emitter import emit_response_events
    from opentelemetry.semconv_ai import LLMRequestTypeValues

    class EmbedResponseWithoutText:
        """Stands in for EmbedByTypeResponse, which has no `text` and no `finish_reason`."""

        def __getattr__(self, name):
            raise AssertionError(
                f"the embedding branch read {name!r} off an embed response"
            )

    monkeypatch.setattr(Config, "use_legacy_attributes", False)
    emit_response_events(
        logger_provider.get_logger(__name__),
        LLMRequestTypeValues.EMBEDDING,
        EmbedResponseWithoutText(),
    )

    logs = log_exporter.get_finished_logs()
    assert [log.log_record.event_name for log in logs] == ["gen_ai.choice"]
    assert dict(logs[0].log_record.body) == {
        "index": 0,
        "message": {},
        "finish_reason": "unknown",
    }
