from dataclasses import FrozenInstanceError

import pytest

from opentelemetry.semconv_ai._contract import (
    AttributeSpec,
    Level,
    SpanSpec,
    parse_requirement_level,
)


class TestParseRequirementLevel:
    def test_plain_string_required(self):
        assert parse_requirement_level("required") == (Level.REQUIRED, None)

    def test_plain_string_opt_in(self):
        assert parse_requirement_level("opt_in") == (Level.OPT_IN, None)

    def test_dict_conditionally_required_carries_condition(self):
        raw = {"conditionally_required": "If the operation ended in an error."}
        assert parse_requirement_level(raw) == (
            Level.CONDITIONALLY_REQUIRED,
            "If the operation ended in an error.",
        )

    def test_dict_recommended_carries_condition(self):
        raw = {"recommended": "when available"}
        assert parse_requirement_level(raw) == (Level.RECOMMENDED, "when available")

    def test_unknown_level_raises(self):
        with pytest.raises(ValueError, match="unknown requirement_level"):
            parse_requirement_level("sometimes_maybe")


class TestSpanSpec:
    def test_required_filters_to_required_only(self):
        spec = SpanSpec(
            id="span.gen_ai.inference.client",
            span_kind="client",
            attributes=(
                AttributeSpec("gen_ai.operation.name", Level.REQUIRED, None, None),
                AttributeSpec("gen_ai.input.messages", Level.OPT_IN, None, None),
                AttributeSpec("error.type", Level.CONDITIONALLY_REQUIRED, "on error", None),
            ),
        )
        assert [a.name for a in spec.required()] == ["gen_ai.operation.name"]

    def test_frozen_instances_reject_mutation(self):
        spec = AttributeSpec("gen_ai.operation.name", Level.REQUIRED, None, None)
        with pytest.raises(FrozenInstanceError):
            spec.name = "other"

    def test_attribute_spec_with_populated_enum_members_is_hashable(self):
        spec = AttributeSpec(
            "gen_ai.operation.name", Level.REQUIRED, None, ("chat", "embeddings")
        )
        assert hash(spec) == hash(
            AttributeSpec(
                "gen_ai.operation.name", Level.REQUIRED, None, ("chat", "embeddings")
            )
        )

    def test_span_spec_holding_attributes_is_hashable(self):
        """SpanSpec holds Tuple[AttributeSpec, ...]; hashability is only
        guaranteed if every contained element is itself hashable."""
        spec = SpanSpec(
            id="span.gen_ai.inference.client",
            span_kind="client",
            attributes=(
                AttributeSpec("gen_ai.operation.name", Level.REQUIRED, None, ("chat",)),
                AttributeSpec("gen_ai.request.model", Level.RECOMMENDED, None, None),
            ),
        )
        assert len({spec, spec}) == 1
