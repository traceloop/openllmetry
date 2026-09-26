"""Opt-in W3C trace-context helpers for agent-to-agent calls."""

from typing import Any, MutableMapping

from opentelemetry import propagate
from opentelemetry.context import Context


def inject_trace_context(carrier: MutableMapping[str, Any] | None = None) -> MutableMapping[str, Any]:
    """Inject the current W3C trace context into an HTTP-like carrier."""
    target: MutableMapping[str, Any] = carrier if carrier is not None else {}
    propagate.inject(target)
    return target


def extract_trace_context(carrier: Any) -> Context:
    """Extract a remote W3C trace context without changing the current context."""
    return propagate.extract(carrier)

