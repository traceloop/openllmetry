"""Emits the GenAI semantic convention events used when legacy attributes are disabled."""

from dataclasses import asdict
from typing import Optional

from opentelemetry._logs import Logger, LogRecord
from opentelemetry.instrumentation.runpod.event_models import (
    ChoiceEvent,
    CompletionMessage,
    MessageEvent,
)
from opentelemetry.instrumentation.runpod.utils import (
    dont_throw,
    should_emit_events,
    should_send_prompts,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

EVENT_ATTRIBUTES = {GenAIAttributes.GEN_AI_SYSTEM: "runpod"}
"""The attributes carried by every event emitted by this instrumentation."""


@dont_throw
def emit_request_event(event_logger: Optional[Logger], content) -> None:
    """Emits ``gen_ai.user.message`` carrying the payload submitted to the endpoint."""
    if content is None:
        return

    body = asdict(MessageEvent(content=content))
    # The event name already carries the role, so drop the duplicate - the
    # semantic conventions make it conditionally required only when it differs.
    body.pop("role", None)
    if not should_send_prompts():
        body.pop("content", None)

    _emit(event_logger, "gen_ai.user.message", body)


@dont_throw
def emit_response_event(event_logger: Optional[Logger], content) -> None:
    """Emits ``gen_ai.choice`` carrying the response recorded on the span."""
    if content is None:
        return

    body = asdict(
        ChoiceEvent(
            index=0,
            message=CompletionMessage(content=content, role="assistant"),
        )
    )
    body["message"].pop("role", None)
    if not should_send_prompts():
        body["message"].pop("content", None)

    _emit(event_logger, "gen_ai.choice", body)


def _emit(event_logger: Optional[Logger], event_name: str, body: dict) -> None:
    """
    Emits one event to the OpenTelemetry SDK.

    Emitting is a no-op unless the instrumentor runs with
    ``use_legacy_attributes=False``, which is the mode in which these events - rather
    than span attributes - carry the request and response content.
    """
    if not should_emit_events() or event_logger is None:
        return

    event_logger.emit(
        LogRecord(
            body=body,
            attributes=EVENT_ATTRIBUTES,
            event_name=event_name,
        )
    )
