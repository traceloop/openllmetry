from __future__ import annotations

import copy
import logging
import threading
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from opentelemetry.trace import Span

SAFETY_EVENT_NAME = "fortifyroot.safety.violation"

logger = logging.getLogger(__name__)


class SafetyDecision(str, Enum):
    ALLOW = "ALLOW"
    MASK = "MASK"


class SafetyLocation(str, Enum):
    PROMPT = "PROMPT"
    COMPLETION = "COMPLETION"


@dataclass(frozen=True, slots=True)
class SafetyFinding:
    category: str
    severity: str
    action: str
    rule_name: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class SafetyContext:
    provider: str
    text: str
    location: SafetyLocation
    span_name: str
    request_type: str | None = None
    segment_index: int | None = None
    segment_role: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SafetyResult:
    text: str
    findings: Sequence[SafetyFinding] = ()
    overall_action: str = SafetyDecision.ALLOW.value


PromptSafetyHandler = Callable[[SafetyContext], SafetyResult | None]
CompletionSafetyHandler = Callable[[SafetyContext], SafetyResult | None]


_handler_lock = threading.RLock()
_prompt_handler: PromptSafetyHandler | None = None
_completion_handler: CompletionSafetyHandler | None = None


def register_prompt_safety_handler(handler: PromptSafetyHandler | None) -> None:
    global _prompt_handler
    with _handler_lock:
        _prompt_handler = handler


def register_completion_safety_handler(handler: CompletionSafetyHandler | None) -> None:
    global _completion_handler
    with _handler_lock:
        _completion_handler = handler


def clear_safety_handlers() -> None:
    from opentelemetry.instrumentation.fortifyroot.streaming import (
        clear_completion_safety_stream_factory,
    )

    register_prompt_safety_handler(None)
    register_completion_safety_handler(None)
    clear_completion_safety_stream_factory()


def get_prompt_safety_handler() -> PromptSafetyHandler | None:
    with _handler_lock:
        return _prompt_handler


def get_completion_safety_handler() -> CompletionSafetyHandler | None:
    with _handler_lock:
        return _completion_handler


def run_prompt_safety(
    *,
    span: Span | None,
    provider: str,
    span_name: str,
    text: str | None,
    location: SafetyLocation,
    request_type: str | None = None,
    segment_index: int | None = None,
    segment_role: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SafetyResult | None:
    return _run_safety(
        handler=get_prompt_safety_handler(),
        span=span,
        provider=provider,
        span_name=span_name,
        text=text,
        location=location,
        request_type=request_type,
        segment_index=segment_index,
        segment_role=segment_role,
        metadata=metadata,
    )


def run_completion_safety(
    *,
    span: Span | None,
    provider: str,
    span_name: str,
    text: str | None,
    location: SafetyLocation,
    request_type: str | None = None,
    segment_index: int | None = None,
    segment_role: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SafetyResult | None:
    return _run_safety(
        handler=get_completion_safety_handler(),
        span=span,
        provider=provider,
        span_name=span_name,
        text=text,
        location=location,
        request_type=request_type,
        segment_index=segment_index,
        segment_role=segment_role,
        metadata=metadata,
    )


def clone_value(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def get_object_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def set_object_value(obj: Any, key: str, value: Any) -> bool:
    if isinstance(obj, MutableMapping):
        obj[key] = value
        return True
    try:
        setattr(obj, key, value)
        return True
    except Exception:
        return False


def _run_safety(
    *,
    handler: Callable[[SafetyContext], SafetyResult | None] | None,
    span: Span | None,
    provider: str,
    span_name: str,
    text: str | None,
    location: SafetyLocation,
    request_type: str | None,
    segment_index: int | None,
    segment_role: str | None,
    metadata: Mapping[str, Any] | None,
) -> SafetyResult | None:
    if handler is None or text is None or text == "":
        return None

    context = SafetyContext(
        provider=provider,
        text=text,
        location=location,
        span_name=span_name,
        request_type=request_type,
        segment_index=segment_index,
        segment_role=segment_role,
        metadata=metadata or {},
    )
    try:
        result = handler(context)
    except Exception:
        logger.warning("Safety handler execution failed", exc_info=True)
        return None
    if result is None:
        return None

    normalized = _normalize_result(text, result)
    _emit_findings(span, context, normalized)
    return normalized


def _normalize_result(original_text: str, result: SafetyResult) -> SafetyResult:
    text = result.text if result.text is not None else original_text
    findings = tuple(_normalize_finding(finding) for finding in result.findings)
    overall_action = _normalize_decision(result.overall_action)
    return SafetyResult(text=text, findings=findings, overall_action=overall_action)


def _normalize_finding(finding: SafetyFinding) -> SafetyFinding:
    return SafetyFinding(
        category=str(finding.category).upper(),
        severity=str(finding.severity).upper(),
        action=_normalize_decision(finding.action),
        rule_name=finding.rule_name,
        start=int(finding.start),
        end=int(finding.end),
    )


def _normalize_decision(value: str) -> str:
    raw = str(value).strip().upper()
    if raw == SafetyDecision.MASK.value:
        return SafetyDecision.MASK.value
    return SafetyDecision.ALLOW.value


def _emit_findings(
    span: Span | None,
    context: SafetyContext,
    result: SafetyResult,
) -> None:
    if span is None or not span.is_recording():
        return

    for finding in result.findings:
        attributes: dict[str, Any] = {
            "fortifyroot.safety.category": finding.category,
            "fortifyroot.safety.severity": finding.severity,
            "fortifyroot.safety.action": finding.action,
            "fortifyroot.safety.location": context.location.value,
            "fortifyroot.safety.rule_name": finding.rule_name,
            "fortifyroot.safety.start": finding.start,
            "fortifyroot.safety.end": finding.end,
        }
        if context.segment_index is not None:
            attributes["fortifyroot.safety.segment_index"] = context.segment_index
        if context.segment_role:
            attributes["fortifyroot.safety.segment_role"] = context.segment_role
        span.add_event(SAFETY_EVENT_NAME, attributes=attributes)
