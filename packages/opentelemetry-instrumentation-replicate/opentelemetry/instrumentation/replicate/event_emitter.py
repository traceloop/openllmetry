from dataclasses import asdict
from enum import Enum
from typing import Union

from opentelemetry._events import Event, EventLogger
from opentelemetry.instrumentation.replicate.event_models import (
    ChoiceEvent,
    MessageEvent,
)
from opentelemetry.instrumentation.replicate.utils import (
    dont_throw,
    should_emit_events,
    should_send_prompts,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from replicate.prediction import Prediction


class Roles(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


VALID_MESSAGE_ROLES = {role.value for role in Roles}
"""The valid roles for naming the message event."""

EVENT_ATTRIBUTES = {GenAIAttributes.GEN_AI_SYSTEM: "replicate"}
"""The attributes to be used for the event."""


@dont_throw
def emit_choice_events(
    response: Union[str, list, Prediction], event_logger: Union[EventLogger, None]
):
    # Handle replicate.run responses
    if isinstance(response, list):
        for i, generation in enumerate(response):
            emit_event(
                ChoiceEvent(
                    index=i, message={"content": generation, "role": "assistant"}
                ),
                event_logger,
            )
    # Handle replicate.predictions.create responses
    elif isinstance(response, Prediction):
        emit_event(
            ChoiceEvent(
                index=0, message={"content": response.output, "role": "assistant"}
            ),
            event_logger,
        )
    # Handle replicate.stream responses built from _build_from_streaming_response
    elif isinstance(response, str):
        emit_event(
            ChoiceEvent(index=0, message={"content": response, "role": "assistant"}),
            event_logger,
        )
    else:
        raise ValueError(
            "It wasn't possible to emit the choice events due to an unsupported response type"
        )


def emit_event(
    event: Union[MessageEvent, ChoiceEvent], event_logger: Union[EventLogger, None]
) -> None:
    """
    Emit an event to the OpenTelemetry SDK.

    Args:
        event: The event to emit.
    """
    if not should_emit_events() or event_logger is None:
        return

    if isinstance(event, MessageEvent):
        _emit_message_event(event, event_logger)
    elif isinstance(event, ChoiceEvent):
        _emit_choice_event(event, event_logger)
    else:
        raise TypeError("Unsupported event type")


def _emit_message_event(event: MessageEvent, event_logger: EventLogger) -> None:
    body = asdict(event)

    if event.role in VALID_MESSAGE_ROLES:
        name = "gen_ai.{}.message".format(event.role)
        # According to the semantic conventions, the role is conditionally required if available
        # and not equal to the "role" in the message name. So, remove the role from the body if
        # it is the same as the in the event name.
        body.pop("role", None)
    else:
        name = "gen_ai.user.message"

    # According to the semantic conventions, only the assistant role has tool call
    if event.role != Roles.ASSISTANT.value and event.tool_calls is not None:
        del body["tool_calls"]
    elif event.tool_calls is None:
        del body["tool_calls"]

    if not should_send_prompts():
        del body["content"]
        if body.get("tool_calls") is not None:
            for tool_call in body["tool_calls"]:
                tool_call["function"].pop("arguments", None)

    event_logger.emit(Event(name=name, body=body, attributes=EVENT_ATTRIBUTES))


def _emit_choice_event(event: ChoiceEvent, event_logger: EventLogger) -> None:
    body = asdict(event)
    if event.message["role"] == Roles.ASSISTANT.value:
        # According to the semantic conventions, the role is conditionally required if available
        # and not equal to "assistant", so remove the role from the body if it is "assistant".
        body["message"].pop("role", None)

    if event.tool_calls is None:
        del body["tool_calls"]

    if not should_send_prompts():
        body["message"].pop("content", None)
        if body.get("tool_calls") is not None:
            for tool_call in body["tool_calls"]:
                tool_call["function"].pop("arguments", None)

    event_logger.emit(
        Event(name="gen_ai.choice", body=body, attributes=EVENT_ATTRIBUTES)
    )
