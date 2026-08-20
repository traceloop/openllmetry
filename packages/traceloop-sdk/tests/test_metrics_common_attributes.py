import json

from opentelemetry.context import attach, set_value
from opentelemetry.semconv_ai import SpanAttributes

from traceloop.sdk.tracing.tracing import metrics_common_attributes


def test_scalar_association_properties_pass_through():
    attach(set_value("association_properties", {"user_id": 1, "user_name": "John"}))

    attributes = metrics_common_attributes()

    assert attributes[f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.user_id"] == 1
    assert (
        attributes[f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.user_name"]
        == "John"
    )


def test_list_valued_association_property_is_json_encoded():
    """LangGraph tags spans with list-valued baggage (e.g. langgraph_triggers).
    Metric attributes must be hashable/scalar, so a raw list here previously
    crashed the OTel metrics SDK's aggregation and silently dropped every
    response-side span attribute set after it (gen_ai.output.messages, etc.)."""
    attach(
        set_value(
            "association_properties",
            {"langgraph_triggers": ["branch:to:resolve_naics"]},
        )
    )

    attributes = metrics_common_attributes()

    key = f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.langgraph_triggers"
    assert attributes[key] == json.dumps(["branch:to:resolve_naics"])
    # must be hashable, matching what the OTel metrics SDK requires for its
    # aggregation key: frozenset(attributes.items())
    hash(attributes[key])


def test_dict_valued_association_property_is_json_encoded():
    attach(
        set_value(
            "association_properties",
            {"metadata": {"nested": "value"}},
        )
    )

    attributes = metrics_common_attributes()

    key = f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.metadata"
    assert attributes[key] == json.dumps({"nested": "value"})
    hash(attributes[key])
