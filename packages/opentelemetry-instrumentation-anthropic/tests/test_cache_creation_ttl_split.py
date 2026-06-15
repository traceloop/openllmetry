"""Unit tests for get_cache_creation_ttl_split and the TTL span attributes it drives."""

import sys
import importlib
import pytest
from opentelemetry.instrumentation.anthropic.utils import get_cache_creation_ttl_split

# Load SpanAttributes from the local source tree (not the installed wheel) so the
# new ephemeral-TTL constants are available regardless of which semconv_ai wheel is
# currently installed in this venv.
import importlib.util
import pathlib

_semconv_src = (
    pathlib.Path(__file__).parents[2]
    / "opentelemetry-semantic-conventions-ai"
    / "opentelemetry"
    / "semconv_ai"
    / "__init__.py"
)
_spec = importlib.util.spec_from_file_location("semconv_ai_local", _semconv_src)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
SpanAttributes = _mod.SpanAttributes

_ATTR_5M = SpanAttributes.GEN_AI_USAGE_CACHE_CREATION_EPHEMERAL_5M_INPUT_TOKENS
_ATTR_1H = SpanAttributes.GEN_AI_USAGE_CACHE_CREATION_EPHEMERAL_1H_INPUT_TOKENS


# ---------------------------------------------------------------------------
# Unit tests for get_cache_creation_ttl_split
# ---------------------------------------------------------------------------


def test_returns_none_none_when_cache_creation_absent_from_dict():
    assert get_cache_creation_ttl_split({}) == (None, None)


def test_returns_none_none_when_cache_creation_key_is_none_in_dict():
    assert get_cache_creation_ttl_split({"cache_creation": None}) == (None, None)


def test_reads_ttl_split_from_dict():
    usage = {
        "cache_creation": {
            "ephemeral_5m_input_tokens": 100,
            "ephemeral_1h_input_tokens": 200,
        }
    }
    assert get_cache_creation_ttl_split(usage) == (100, 200)


def test_reads_zero_values_from_dict():
    usage = {
        "cache_creation": {
            "ephemeral_5m_input_tokens": 0,
            "ephemeral_1h_input_tokens": 0,
        }
    }
    assert get_cache_creation_ttl_split(usage) == (0, 0)


def test_returns_none_for_missing_keys_in_dict():
    usage = {"cache_creation": {}}
    assert get_cache_creation_ttl_split(usage) == (None, None)


def test_returns_none_none_when_no_cache_creation_attr_on_object():
    class Usage:
        pass

    assert get_cache_creation_ttl_split(Usage()) == (None, None)


def test_returns_none_none_when_cache_creation_attr_is_none():
    class Usage:
        cache_creation = None

    assert get_cache_creation_ttl_split(Usage()) == (None, None)


def test_reads_ttl_split_from_object():
    class CacheCreation:
        ephemeral_5m_input_tokens = 150
        ephemeral_1h_input_tokens = 300

    class Usage:
        cache_creation = CacheCreation()

    assert get_cache_creation_ttl_split(Usage()) == (150, 300)


def test_reads_zero_values_from_object():
    class CacheCreation:
        ephemeral_5m_input_tokens = 0
        ephemeral_1h_input_tokens = 0

    class Usage:
        cache_creation = CacheCreation()

    assert get_cache_creation_ttl_split(Usage()) == (0, 0)


def test_returns_none_for_missing_attrs_on_object():
    class CacheCreation:
        pass

    class Usage:
        cache_creation = CacheCreation()

    assert get_cache_creation_ttl_split(Usage()) == (None, None)


def test_mixed_present_and_absent_keys_in_dict():
    usage = {"cache_creation": {"ephemeral_5m_input_tokens": 50}}
    five_m, one_h = get_cache_creation_ttl_split(usage)
    assert five_m == 50
    assert one_h is None


# ---------------------------------------------------------------------------
# SpanAttributes constants sanity check
# ---------------------------------------------------------------------------


def test_span_attribute_names():
    assert _ATTR_5M == "gen_ai.usage.cache_creation.ephemeral_5m_input_tokens"
    assert _ATTR_1H == "gen_ai.usage.cache_creation.ephemeral_1h_input_tokens"


# ---------------------------------------------------------------------------
# Span emission: verify attributes are set / skipped correctly
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_span():
    class _Span:
        def __init__(self):
            self.attrs = {}

        def set_attribute(self, name, value):
            self.attrs[name] = value

        def is_recording(self):
            return True

    return _Span()


def _emit_ttl_attrs(span, five_m, one_h):
    """Replicates the conditional attribute-setting logic from __init__.py / streaming.py."""
    from opentelemetry.instrumentation.anthropic.utils import set_span_attribute

    if five_m is not None:
        set_span_attribute(span, _ATTR_5M, five_m)
    if one_h is not None:
        set_span_attribute(span, _ATTR_1H, one_h)


def test_span_attrs_set_when_both_ttl_values_nonzero(mock_span):
    _emit_ttl_attrs(mock_span, 100, 200)
    assert mock_span.attrs[_ATTR_5M] == 100
    assert mock_span.attrs[_ATTR_1H] == 200


def test_span_attrs_set_when_both_ttl_values_are_zero(mock_span):
    _emit_ttl_attrs(mock_span, 0, 0)
    assert mock_span.attrs[_ATTR_5M] == 0
    assert mock_span.attrs[_ATTR_1H] == 0


def test_span_attrs_not_set_when_both_ttl_values_are_none(mock_span):
    _emit_ttl_attrs(mock_span, None, None)
    assert _ATTR_5M not in mock_span.attrs
    assert _ATTR_1H not in mock_span.attrs


def test_span_attrs_partial_set_when_only_5m_present(mock_span):
    _emit_ttl_attrs(mock_span, 75, None)
    assert mock_span.attrs[_ATTR_5M] == 75
    assert _ATTR_1H not in mock_span.attrs
