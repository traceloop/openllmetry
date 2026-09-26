"""Unit tests for the pinecone upsert/delete span-attribute helpers (#2688)."""

from unittest.mock import MagicMock

from opentelemetry.instrumentation.pinecone.query_handlers import (
    set_delete_input_attributes,
    set_upsert_input_attributes,
)
from opentelemetry.semconv_ai import SpanAttributes


def _attrs(span):
    return {call.args[0]: call.args[1] for call in span.set_attribute.call_args_list}


def test_upsert_attributes_from_kwargs():
    span = MagicMock()
    set_upsert_input_attributes(
        span, {"vectors": [1, 2, 3], "namespace": "ns1", "batch_size": 100}, ()
    )
    attrs = _attrs(span)
    assert attrs[SpanAttributes.PINECONE_UPSERT_VECTORS_COUNT] == 3
    assert attrs[SpanAttributes.PINECONE_UPSERT_NAMESPACE] == "ns1"
    assert attrs[SpanAttributes.PINECONE_UPSERT_BATCH_SIZE] == 100


def test_upsert_vectors_from_positional_arg():
    span = MagicMock()
    set_upsert_input_attributes(span, {}, ([1, 2],))
    assert _attrs(span)[SpanAttributes.PINECONE_UPSERT_VECTORS_COUNT] == 2


def test_upsert_generator_vectors_not_counted():
    span = MagicMock()
    set_upsert_input_attributes(span, {"vectors": (x for x in range(3))}, ())
    assert SpanAttributes.PINECONE_UPSERT_VECTORS_COUNT not in _attrs(span)


def test_delete_attributes_from_kwargs_with_dict_filter():
    span = MagicMock()
    set_delete_input_attributes(
        span,
        {"ids": ["a", "b"], "delete_all": True, "namespace": "ns", "filter": {"k": "v"}},
        (),
    )
    attrs = _attrs(span)
    assert attrs[SpanAttributes.PINECONE_DELETE_IDS_COUNT] == 2
    assert attrs[SpanAttributes.PINECONE_DELETE_NAMESPACE] == "ns"
    assert attrs[SpanAttributes.PINECONE_DELETE_DELETE_ALL] is True
    assert attrs[SpanAttributes.PINECONE_DELETE_FILTER] == '{"k": "v"}'


def test_delete_ids_from_positional_arg():
    span = MagicMock()
    set_delete_input_attributes(span, {}, (["x", "y", "z"],))
    assert _attrs(span)[SpanAttributes.PINECONE_DELETE_IDS_COUNT] == 3
