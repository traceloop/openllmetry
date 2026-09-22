import re

import pytest

from opentelemetry.semconv_ai import (
    EventAttributes,
    Events,
    GenAISecurityThreatActionValues,
    GenAISecurityThreatSeverityValues,
)


def test_security_threat_event_value():
    """Verify the standardized security threat event name."""

    assert Events.GEN_AI_SECURITY_THREAT_DETECTED.value == (
        "gen_ai.security.threat.detected"
    )


@pytest.mark.parametrize(
    "attribute, expected",
    [
        (
            EventAttributes.GEN_AI_SECURITY_THREAT_RULE_ID,
            "gen_ai.security.threat.rule_id",
        ),
        (
            EventAttributes.GEN_AI_SECURITY_THREAT_CATEGORY,
            "gen_ai.security.threat.category",
        ),
        (
            EventAttributes.GEN_AI_SECURITY_THREAT_SEVERITY,
            "gen_ai.security.threat.severity",
        ),
        (
            EventAttributes.GEN_AI_SECURITY_THREAT_SCANNER_NAME,
            "gen_ai.security.threat.scanner_name",
        ),
        (
            EventAttributes.GEN_AI_SECURITY_THREAT_SCANNER_VERSION,
            "gen_ai.security.threat.scanner_version",
        ),
        (
            EventAttributes.GEN_AI_SECURITY_THREAT_ACTION,
            "gen_ai.security.threat.action",
        ),
    ],
)
def test_security_threat_attribute_values(attribute, expected):
    """Verify each security threat attribute uses its exact semantic key."""

    assert attribute.value == expected


def test_security_threat_names_use_dot_notation():
    """Verify threat event and attribute keys follow OTel dot notation."""

    values = [
        Events.GEN_AI_SECURITY_THREAT_DETECTED.value,
        *(attribute.value for attribute in EventAttributes if attribute.name.startswith("GEN_AI_SECURITY_THREAT_")),
    ]
    assert all(re.fullmatch(r"[a-z0-9_]+(?:\.[a-z0-9_]+)+", value) for value in values)
    assert all(value.startswith("gen_ai.security.threat.") for value in values)


def test_security_threat_severity_values():
    """Verify the supported security threat severity values."""

    assert {value.value for value in GenAISecurityThreatSeverityValues} == {
        "low",
        "medium",
        "high",
        "critical",
    }


def test_security_threat_action_values():
    """Verify the supported security threat response action values."""

    assert {value.value for value in GenAISecurityThreatActionValues} == {
        "blocked",
        "warned",
        "logged",
    }
