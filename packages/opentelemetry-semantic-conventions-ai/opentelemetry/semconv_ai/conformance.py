"""Check emitted spans against the OpenTelemetry GenAI semantic-convention contract.

`check_attributes` is pure — it takes a mapping, not an OTel object — so it can be
tested without a tracer. `assert_conforms` is the pytest-facing adapter.

The contract is two-sided. The registry supplies the vocabulary and the constraints;
each package supplies `expected`, the attributes it promises to emit. Only ~9% of
registry attributes are `required`, so registry levels alone cannot detect a dropped
attribute like `gen_ai.response.finish_reasons`, which is `recommended` everywhere it
appears. The package's own promise is what makes that detectable.

Namespace scope is deliberately asymmetric. Every attribute the contract marks
`required` is checked regardless of namespace, so `mcp.method.name` is enforced on MCP
spans. Undeclared-attribute policing applies only to `gen_ai.*`, because namespaces
such as `traceloop.*` are ours to use freely.

Enum values follow OTel's open-enum semantics: a value outside the declared members is
reported but never blocks. The registry carries no extensibility metadata, and some
providers (ollama, together, replicate) have no member at all.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Mapping, Optional

from opentelemetry.semconv_ai._contract import SpanSpec
from opentelemetry.semconv_ai._contract.generated import SPANS

GEN_AI_PREFIX = "gen_ai."

MISSING_REQUIRED = "missing_required"
MISSING_EXPECTED = "missing_expected"
UNDECLARED_GEN_AI = "undeclared_gen_ai"
UNKNOWN_ENUM_VALUE = "unknown_enum_value"

#: Kinds that fail a test in enforcing mode. `UNKNOWN_ENUM_VALUE` is absent
#: deliberately — OTel enums are open, so an unlisted value is informational.
BLOCKING_KINDS = frozenset({MISSING_REQUIRED, MISSING_EXPECTED, UNDECLARED_GEN_AI})


class ConformanceWarning(UserWarning):
    """Emitted for non-blocking findings, and for all findings in warn-only mode."""


@dataclass(frozen=True)
class Violation:
    kind: str
    attribute: str
    detail: str

    @property
    def blocking(self) -> bool:
        return self.kind in BLOCKING_KINDS

    def __str__(self) -> str:
        return f"[{self.kind}] {self.attribute}: {self.detail}"


def check_attributes(
    attributes: Mapping[str, Any],
    spec: SpanSpec,
    extensions: FrozenSet[str] = frozenset(),
    expected: FrozenSet[str] = frozenset(),
) -> List[Violation]:
    """Return every way `attributes` violates `spec`. Empty list means conforming.

    `expected` names attributes the calling package promises to emit; each absent one
    is a blocking violation regardless of its registry requirement level.

    Raises ValueError if `expected` names something neither declared by `spec` nor
    listed in `extensions` — that is a typo in the package's declaration, not a
    telemetry defect, and must fail loudly rather than silently pass.
    """
    for name in sorted(expected):
        if spec.by_name(name) is None and name not in extensions:
            raise ValueError(
                f"{name!r} appears in `expected` but is neither declared by "
                f"{spec.id} nor listed in `extensions` — likely a typo"
            )

    violations: List[Violation] = []

    for attr in spec.required():
        if attr.name not in attributes:
            violations.append(
                Violation(
                    MISSING_REQUIRED,
                    attr.name,
                    f"required by {spec.id} but not present on the span",
                )
            )

    for name in sorted(expected):
        if name not in attributes:
            violations.append(
                Violation(
                    MISSING_EXPECTED,
                    name,
                    "this package promises to emit it, but the span does not carry it",
                )
            )

    for name, value in attributes.items():
        if not name.startswith(GEN_AI_PREFIX):
            continue

        declared = spec.by_name(name)
        if declared is None:
            if name in extensions:
                continue
            violations.append(
                Violation(
                    UNDECLARED_GEN_AI,
                    name,
                    "not defined by the contract for this span. Either use a contract "
                    "attribute or declare it in _contract/extensions.py",
                )
            )
            continue

        if declared.enum_members and value not in declared.enum_members:
            violations.append(
                Violation(
                    UNKNOWN_ENUM_VALUE,
                    name,
                    f"value {value!r} is not among {list(declared.enum_members)}; "
                    "OTel enums are open, so this is informational only",
                )
            )

    violations.sort(key=lambda v: (v.attribute, v.kind))
    return violations


def assert_conforms(
    span: Any,
    group_id: str,
    *,
    enforcing: bool,
    extensions: FrozenSet[str] = frozenset(),
    expected: FrozenSet[str] = frozenset(),
    _spans: Optional[Dict[str, SpanSpec]] = None,
) -> List[Violation]:
    """Check one span against a contract group.

    Enforcing mode raises on blocking violations. Non-blocking findings always warn,
    never fail, in either mode. Returns every violation found.
    """
    table = SPANS if _spans is None else _spans
    if group_id not in table:
        raise KeyError(f"no such span group in the contract: {group_id!r}")

    violations = check_attributes(
        dict(span.attributes or {}), table[group_id], extensions, expected
    )
    if not violations:
        return []

    span_name = getattr(span, "name", "<unnamed>")
    blocking = [v for v in violations if v.blocking]

    if enforcing and blocking:
        report = "\n".join(f"  {v}" for v in blocking)
        raise AssertionError(f"span {span_name!r} violates {group_id}:\n{report}")

    reportable = violations if not enforcing else [v for v in violations if not v.blocking]
    if reportable:
        report = "\n".join(f"  {v}" for v in reportable)
        warnings.warn(
            f"span {span_name!r} violates {group_id}:\n{report}",
            ConformanceWarning,
            stacklevel=2,
        )
    return violations


__all__ = [
    "BLOCKING_KINDS",
    "ConformanceWarning",
    "Violation",
    "assert_conforms",
    "check_attributes",
]
