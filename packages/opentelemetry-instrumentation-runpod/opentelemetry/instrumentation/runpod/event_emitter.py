"""Emits the GenAI semantic convention events used when legacy attributes are disabled."""

from dataclasses import asdict
from typing import Optional, Union

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

VALID_MESSAGE_ROLES = ("user", "system", "assistant", "tool")
"""The roles that may appear in the name of a ``gen_ai.<role>.message`` event."""

EVENT_ATTRIBUTES = {GenAIAttributes.GEN_AI_SYSTEM: "runpod"}
"""The attributes carried by every event emitted by this instrumentation."""


@dont_throw
def emit_request_event(event_logger: Optional[Logger], content) -> None:
    """Emits ``gen_ai.user.message`` carrying the payload submitted to the endpoint."""
    if content is None:
        return

    emit_event(MessageEvent(content=content), event_logger)


@dont_throw
def emit_response_event(event_logger: Optional[Logger], content) -> None:
    """Emits ``gen_ai.choice`` carrying the response recorded on the span."""
    if content is None:
        return

    emit_event(
        ChoiceEvent(
            index=0,
            message=CompletionMessage(content=content, role="assistant"),
        ),
        event_logger,
    )


def emit_event(
    event: Union[MessageEvent, ChoiceEvent], event_logger: Optional[Logger]
) -> None:
    """
    Emits an event to the OpenTelemetry SDK.

    Emitting is a no-op unless the instrumentor runs with
    ``use_legacy_attributes=False``, which is the mode in which these events - rather
    than span attributes - carry the request and response content.
    """
    if not should_emit_events() or event_logger is None:
        return

    if isinstance(event, MessageEvent):
        _emit_message_event(event, event_logger)
    elif isinstance(event, ChoiceEvent):
        _emit_choice_event(event, event_logger)
    else:
        raise TypeError("Unsupported event type")


def _emit_message_event(event: MessageEvent, event_logger: Logger) -> None:
    body = asdict(event)

    if event.role in VALID_MESSAGE_ROLES:
        name = f"gen_ai.{event.role}.message"
        # The semantic conventions make the role conditionally required when it
        # differs from the role in the event name, so drop the duplicate.
        body.pop("role", None)
    else:
        name = "gen_ai.user.message"

    if not should_send_prompts():
        body.pop("content", None)

    event_logger.emit(
        LogRecord(
            body=body,
            attributes=EVENT_ATTRIBUTES,
            event_name=name,
        )
    )


def _emit_choice_event(event: ChoiceEvent, event_logger: Logger) -> None:
    body = asdict(event)

    if event.message.get("role") == "assistant":
        body["message"].pop("role", None)

    if not should_send_prompts():
        body["message"].pop("content", None)

    event_logger.emit(
        LogRecord(
            body=body,
            attributes=EVENT_ATTRIBUTES,
            event_name="gen_ai.choice",
        )
    )
