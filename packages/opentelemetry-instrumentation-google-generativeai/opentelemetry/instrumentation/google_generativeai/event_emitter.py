from dataclasses import asdict
from enum import Enum
from typing import Union

from google.genai.types import GenerateContentResponse
from opentelemetry._events import Event, EventLogger
from opentelemetry.instrumentation.google_generativeai.event_models import (
    ChoiceEvent,
    MessageEvent,
)
from opentelemetry.instrumentation.google_generativeai.utils import (
    part_to_dict,
    should_emit_events,
    should_send_prompts,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


class Roles(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


VALID_MESSAGE_ROLES = {role.value for role in Roles}
"""The valid roles for naming the message event."""

EVENT_ATTRIBUTES = {GenAIAttributes.GEN_AI_SYSTEM: "gemini"}
"""The attributes to be used for the event."""


def emit_message_events(args, kwargs, event_logger: Union[EventLogger]):
    contents = []

    # Get all prompts (Gemini accepts multiple prompts at once)
    for arg in args:
        if isinstance(arg, str):
            contents.append(arg)
        elif isinstance(arg, list):
            contents.extend(arg)

    # Process kwargs["contents"] if it exists
    if "contents" in kwargs:
        kwarg_contents = kwargs["contents"]
        if isinstance(kwarg_contents, str):
            contents.append(kwarg_contents)
        elif isinstance(kwarg_contents, list):
            contents.extend(kwarg_contents)

    for prompt in contents:
        emit_event(MessageEvent(content=prompt, role="user"), event_logger)


def emit_choice_events(
    response: GenerateContentResponse, event_logger: Union[EventLogger]
):
    for index, candidate in enumerate(response.candidates):
        emit_event(
            ChoiceEvent(
                index=index,
                message={
                    "content": [part_to_dict(i) for i in candidate.content.parts],
                    "role": candidate.content.role,
                },
                finish_reason=candidate.finish_reason.name,
            ),
            event_logger,
        )


def emit_event(
    event: Union[MessageEvent, ChoiceEvent], event_logger: Union[EventLogger]
) -> None:
    """
    Emit an event to the OpenTelemetry SDK.

    Args:
        event: The event to emit.
    """
    if not should_emit_events():
        return

    if isinstance(event, MessageEvent):
        _emit_message_event(event, event_logger)
    elif isinstance(event, ChoiceEvent):
        _emit_choice_event(event, event_logger)
    else:
        raise TypeError("Unsupported event type")


def _emit_message_event(event: MessageEvent, event_logger: Union[EventLogger]) -> None:
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


def _emit_choice_event(event: ChoiceEvent, event_logger: Union[EventLogger]) -> None:
    body = asdict(event)
    if event.message["role"] == Roles.ASSISTANT.value:
        # According to the semantic conventions, the role is conditionally required if available
        # and not equal to "assistant", so remove the role from the body if it is "assistant".
        body["message"].pop("role", None)

    if event.tool_calls is None:
        del body["tool_calls"]

    if not should_send_prompts():
        body["message"].pop("content", None)
        body["message"].pop("role", None)
        if body.get("tool_calls") is not None:
            for tool_call in body["tool_calls"]:
                tool_call["function"].pop("arguments", None)

    event_logger.emit(
        Event(name="gen_ai.choice", body=body, attributes=EVENT_ATTRIBUTES)
    )
