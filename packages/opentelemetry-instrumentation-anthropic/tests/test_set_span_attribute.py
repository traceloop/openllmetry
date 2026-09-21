import logging

import pytest
from opentelemetry.sdk.trace import TracerProvider

from opentelemetry.instrumentation.anthropic.utils import (
    _ANTHROPIC_SENTINEL_TYPES,
    set_span_attribute,
)


@pytest.fixture(scope="module")
def tracer():
    provider = TracerProvider()
    return provider.get_tracer(__name__)


def _new_span(tracer):
    return tracer.start_span("test")


def test_not_given_sentinel_is_not_set(tracer):
    # Regression for #4431: the Anthropic SDK's NOT_GIVEN sentinel is neither
    # None nor "", so it used to slip through and reach span.set_attribute,
    # which rejects it with an "Invalid type" warning.
    anthropic = pytest.importorskip("anthropic")
    span = _new_span(tracer)
    set_span_attribute(span, "gen_ai.request.temperature", anthropic.NOT_GIVEN)
    assert "gen_ai.request.temperature" not in dict(span.attributes)


def test_not_given_does_not_emit_otel_warning(tracer, caplog):
    anthropic = pytest.importorskip("anthropic")
    span = _new_span(tracer)
    with caplog.at_level(logging.WARNING, logger="opentelemetry"):
        set_span_attribute(span, "gen_ai.request.top_p", anthropic.NOT_GIVEN)
    assert not [r for r in caplog.records if "Invalid type" in r.getMessage()]


def test_real_values_still_set(tracer):
    span = _new_span(tracer)
    set_span_attribute(span, "gen_ai.request.temperature", 0.7)
    set_span_attribute(span, "gen_ai.request.max_tokens", 1024)
    set_span_attribute(span, "gen_ai.request.model", "claude-3")
    attrs = dict(span.attributes)
    assert attrs["gen_ai.request.temperature"] == 0.7
    assert attrs["gen_ai.request.max_tokens"] == 1024
    assert attrs["gen_ai.request.model"] == "claude-3"


def test_falsy_but_valid_values_still_set(tracer):
    # 0 and False are valid attribute values and must not be dropped.
    span = _new_span(tracer)
    set_span_attribute(span, "gen_ai.request.temperature", 0)
    set_span_attribute(span, "gen_ai.is_streaming", False)
    attrs = dict(span.attributes)
    assert attrs["gen_ai.request.temperature"] == 0
    assert attrs["gen_ai.is_streaming"] is False


def test_none_and_empty_string_still_skipped(tracer):
    span = _new_span(tracer)
    set_span_attribute(span, "gen_ai.request.none", None)
    set_span_attribute(span, "gen_ai.request.empty", "")
    attrs = dict(span.attributes)
    assert "gen_ai.request.none" not in attrs
    assert "gen_ai.request.empty" not in attrs


def test_omit_sentinel_is_not_set(tracer):
    # The SDK also exposes Omit; if present it must be filtered too.
    anthropic = pytest.importorskip("anthropic")
    if not hasattr(anthropic, "Omit"):
        pytest.skip("this anthropic version has no Omit sentinel")
    span = _new_span(tracer)
    set_span_attribute(span, "gen_ai.request.top_k", anthropic.Omit())
    assert "gen_ai.request.top_k" not in dict(span.attributes)


def test_sentinel_types_discovered():
    # On any anthropic version new enough to matter, at least NotGiven exists.
    pytest.importorskip("anthropic")
    assert len(_ANTHROPIC_SENTINEL_TYPES) >= 1
