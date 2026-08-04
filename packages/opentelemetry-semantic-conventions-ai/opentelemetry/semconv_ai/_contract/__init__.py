"""Data model for the OpenTelemetry GenAI semantic-convention contract.

The concrete contract lives in the generated sibling module `generated.py`,
which is produced by `make -C .semconv generate` and committed. Nothing here
reads it; these are plain value types so the conformance harness can be tested
without the generated artifact.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Tuple, Union


class Level(str, Enum):
    """Requirement level of an attribute on a span, per the semconv spec."""

    REQUIRED = "required"
    CONDITIONALLY_REQUIRED = "conditionally_required"
    RECOMMENDED = "recommended"
    OPT_IN = "opt_in"


def parse_requirement_level(
    raw: Union[str, Mapping[str, str]],
) -> Tuple[Level, Optional[str]]:
    """Normalise weaver's two encodings of requirement_level.

    Weaver emits either a bare string ("required", "opt_in") or a single-key
    mapping carrying the condition ({"conditionally_required": "If ..."}).
    """
    if isinstance(raw, str):
        try:
            return Level(raw), None
        except ValueError:
            raise ValueError(f"unknown requirement_level: {raw!r}") from None

    if isinstance(raw, Mapping) and len(raw) == 1:
        key, condition = next(iter(raw.items()))
        try:
            return Level(key), condition
        except ValueError:
            raise ValueError(f"unknown requirement_level: {key!r}") from None

    raise ValueError(f"unknown requirement_level: {raw!r}")


@dataclass(frozen=True)
class AttributeSpec:
    """One attribute as the contract defines it for a given span."""

    name: str
    level: Level
    condition: Optional[str] = None
    enum_members: Optional[Tuple[str, ...]] = None


@dataclass(frozen=True)
class SpanSpec:
    """One span group from the contract."""

    id: str
    span_kind: Optional[str]
    attributes: Tuple[AttributeSpec, ...]

    def required(self) -> Tuple[AttributeSpec, ...]:
        return tuple(a for a in self.attributes if a.level is Level.REQUIRED)

    def by_name(self, name: str) -> Optional[AttributeSpec]:
        for a in self.attributes:
            if a.name == name:
                return a
        return None


def enum_members_of(attr_type: Any) -> Optional[Tuple[str, ...]]:
    """Extract enum values from weaver's attribute `type` field, if it is an enum."""
    if isinstance(attr_type, Mapping) and "members" in attr_type:
        return tuple(str(m["value"]) for m in attr_type["members"])
    return None


__all__ = [
    "AttributeSpec",
    "Level",
    "SpanSpec",
    "enum_members_of",
    "parse_requirement_level",
]
