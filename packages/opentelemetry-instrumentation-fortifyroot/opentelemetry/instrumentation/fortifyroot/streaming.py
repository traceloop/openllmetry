from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from opentelemetry.trace import Span

from opentelemetry.instrumentation.fortifyroot.safety import (
    HANDLER_LOCK,
    SafetyContext,
    SafetyLocation,
    SafetyResult,
    _emit_findings,
    _normalize_result,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SafetyStreamContext:
    """Streaming metadata used to build per-response safety sessions."""

    provider: str
    location: SafetyLocation
    span_name: str
    request_type: str | None = None
    segment_index: int | None = None
    segment_role: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class CompletionSafetyStreamSession(Protocol):
    """Protocol implemented by SDK streaming safety sessions."""

    def process_chunk(self, text: str) -> SafetyResult | None:
        ...

    def flush(self) -> SafetyResult | None:
        ...

    def get_pending_text(self) -> str:
        ...


CompletionSafetyStreamFactory = Callable[
    [SafetyStreamContext],
    CompletionSafetyStreamSession | None,
]

_handler_lock = HANDLER_LOCK
_completion_stream_factory: CompletionSafetyStreamFactory | None = None


def register_completion_safety_stream_factory(
    factory: CompletionSafetyStreamFactory | None,
) -> None:
    """Set the global factory used to create streaming safety sessions."""
    global _completion_stream_factory
    with _handler_lock:
        _completion_stream_factory = factory


def get_completion_safety_stream_factory() -> CompletionSafetyStreamFactory | None:
    """Return the currently registered streaming safety session factory, or None."""
    with _handler_lock:
        return _completion_stream_factory


def clear_completion_safety_stream_factory() -> None:
    """Unregister the current streaming safety session factory."""
    register_completion_safety_stream_factory(None)


class BoundCompletionSafetyStream:
    """Bind a provider span/context to a streaming safety session."""

    def __init__(
        self,
        *,
        span: Span | None,
        context: SafetyStreamContext,
        session: CompletionSafetyStreamSession,
    ) -> None:
        self._span = span
        self._context = context
        self._session = session

    def process_chunk(self, text: str | None) -> str:
        """Pass a chunk of streamed text through the safety session and return the (possibly modified) text."""
        if text is None or text == "":
            return ""

        try:
            result = self._session.process_chunk(text)
        except Exception:
            logger.warning("Streaming safety session execution failed", exc_info=True)
            return text
        return self._normalize_and_emit(text, result)

    def flush(self) -> str:
        """Flush the safety session and return any remaining buffered text."""
        original_text = self._pending_text()
        try:
            result = self._session.flush()
        except Exception:
            logger.warning("Streaming safety session flush failed", exc_info=True)
            return original_text
        return self._normalize_and_emit(original_text, result)

    def _normalize_and_emit(
        self,
        original_text: str,
        result: SafetyResult | None,
    ) -> str:
        # When the session returns None, no pending text remains to release; returning empty string is intentional.
        if result is None:
            return ""

        normalized = _normalize_result(original_text, result)
        _emit_findings(
            self._span,
            SafetyContext(
                provider=self._context.provider,
                text=original_text,
                location=self._context.location,
                span_name=self._context.span_name,
                request_type=self._context.request_type,
                segment_index=self._context.segment_index,
                segment_role=self._context.segment_role,
                metadata=self._context.metadata,
            ),
            normalized,
        )
        return normalized.text

    def _pending_text(self) -> str:
        getter = getattr(self._session, "get_pending_text", None)
        if not callable(getter):
            return ""
        try:
            pending_text = getter()
        except Exception:
            logger.warning(
                "Streaming safety session pending-text lookup failed",
                exc_info=True,
            )
            return ""
        return pending_text if isinstance(pending_text, str) else ""


def create_completion_safety_stream(
    *,
    span: Span | None,
    provider: str,
    span_name: str,
    location: SafetyLocation,
    request_type: str | None = None,
    segment_index: int | None = None,
    segment_role: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> BoundCompletionSafetyStream | None:
    """Create a bound completion streaming safety helper for a provider span."""

    factory = get_completion_safety_stream_factory()
    if factory is None:
        return None

    context = SafetyStreamContext(
        provider=provider,
        location=location,
        span_name=span_name,
        request_type=request_type,
        segment_index=segment_index,
        segment_role=segment_role,
        metadata=metadata or {},
    )
    try:
        session = factory(context)
    except Exception:
        logger.warning("Streaming safety session creation failed", exc_info=True)
        return None
    if session is None:
        return None
    return BoundCompletionSafetyStream(span=span, context=context, session=session)
