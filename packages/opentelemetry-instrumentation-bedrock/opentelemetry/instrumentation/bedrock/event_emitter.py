import json
from dataclasses import asdict
from enum import Enum
from typing import List, Optional, Union

from opentelemetry._events import Event, EventLogger
from opentelemetry.instrumentation.bedrock.event_models import ChoiceEvent, MessageEvent
from opentelemetry.instrumentation.bedrock.utils import (
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

EVENT_ATTRIBUTES = {
    GenAIAttributes.GEN_AI_SYSTEM: GenAIAttributes.GenAiSystemValues.AWS_BEDROCK.value
}
"""The attributes to be used for the event."""


def emit_message_events(event_logger: Optional[EventLogger], kwargs):
    input_body = json.loads(kwargs.get("body"))
    prompt = input_body.get("prompt")
    messages = input_body.get("messages")
    input_text = input_body.get("inputText")
    system_messages = input_body.get("system")

    if system_messages:
        for message in system_messages:
            emit_event(
                MessageEvent(content=message.get("text"), role="system"), event_logger
            )

    if messages:
        for message in messages:
            emit_event(
                MessageEvent(
                    content=message.get("content"), role=message.get("role", "user")
                ),
                event_logger,
            )
    elif prompt is not None:
        emit_event(MessageEvent(content=prompt, role="user"), event_logger)
    elif input_text is not None:
        emit_event(MessageEvent(content=input_text, role="user"), event_logger)
    else:
        raise ValueError(
            "It wasn't possible to emit the input events due to unknown kwargs."
        )


def emit_choice_events(event_logger: Optional[EventLogger], response):
    response_body: dict = json.loads(response.get("body").read())

    if response_body.get("completions") is not None:
        for i, message in enumerate(response_body.get("completions")):
            emit_event(
                ChoiceEvent(
                    index=i,
                    message={
                        "content": message.get("data", {}).get("text"),
                        "role": "assistant",
                    },
                    finish_reason=message.get("finishReason", {}).get(
                        "reason", "unknown"
                    ),
                ),
                event_logger,
            )
    elif (
        response_body.get("completion") is not None
        or response_body.get("generation") is not None
    ):
        emit_event(
            ChoiceEvent(
                index=0,
                message={
                    "content": response_body.get("completion")
                    or response_body.get("generation"),
                    "role": "assistant",
                },
                finish_reason=response_body.get("stop_reason", "unknown"),
            ),
            event_logger,
        )
    elif response_body.get("generations") is not None:
        for i, message in enumerate(response_body.get("generations")):
            emit_event(
                ChoiceEvent(
                    index=i,
                    message={"content": message.get("text"), "role": "assistant"},
                    finish_reason=message.get("finish_reason", "unknown"),
                ),
                event_logger,
            )
    elif response_body.get("choices") is not None:
        for i, message in enumerate(response_body.get("choices")):
            emit_event(
                ChoiceEvent(
                    index=i,
                    message={"content": message.get("text"), "role": "assistant"},
                    finish_reason=message.get("finish_reason", "unknown"),
                ),
                event_logger,
            )
    elif response_body.get("output") is not None:
        emit_event(
            ChoiceEvent(
                index=0,
                message={
                    "content": response_body.get("output", {})
                    .get("message", {})
                    .get("content"),
                    "role": "assistant",
                },
                finish_reason=response_body.get("stopReason", "unknown"),
            ),
            event_logger,
        )
    elif response_body.get("results") is not None:
        for i, message in enumerate(response_body.get("results")):
            emit_event(
                ChoiceEvent(
                    index=i,
                    message={"content": message.get("outputText"), "role": "assistant"},
                    finish_reason=message.get("completionReason", "unknown"),
                ),
                event_logger,
            )
    elif response_body.get("content") is not None:
        emit_event(
            ChoiceEvent(
                index=0,
                message={"content": response_body.get("content"), "role": "assistant"},
                finish_reason=response_body.get("stop_reason", "unknown"),
            ),
            event_logger,
        )
    else:
        raise ValueError(
            "It wasn't possible to emit the choice events due to an unknow response body."
        )


def emit_input_events_converse(kwargs, event_logger):
    system_messages = kwargs.get("system")
    messages = kwargs.get("messages")

    if system_messages:
        for message in system_messages:
            emit_event(
                MessageEvent(content=message.get("text"), role="system"), event_logger
            )

    for message in messages:
        emit_event(
            MessageEvent(
                content=message.get("content"),
                # Sometimes "role" is None in the response object,
                # so its setted it to "user" by default
                role=message.get("role") or "user",
            ),
            event_logger,
        )


def emit_response_event_converse(response, event_logger):
    emit_event(
        ChoiceEvent(
            index=0,
            message={
                "content": response.get("output", {}).get("message", {}).get("content"),
                "role": response.get("output", {}).get("message", {}).get("role"),
            },
            finish_reason=response.get("stopReason", "unknown"),
        ),
        event_logger,
    )


def emit_streaming_response_event(response_body, event_logger):
    emit_event(
        ChoiceEvent(
            index=0,
            message={
                "content": response_body.get("content")
                or response_body.get("outputText"),
                "role": "assistant",
            },
            # Sometimes, the value is None, what goes agains the semantic conventions
            finish_reason=response_body.get("stop_reason") or "unknown",
        ),
        event_logger,
    )


def emit_streaming_converse_response_event(
    event_logger: Optional[EventLogger],
    response_msg: List[str],
    role: str,
    finish_reason: str,
):
    accumulated_text = "".join(response_msg)
    emit_event(
        ChoiceEvent(
            index=0,
            message={"content": accumulated_text, "role": role},
            finish_reason=finish_reason,
        ),
        event_logger,
    )


def emit_event(
    event: Union[MessageEvent, ChoiceEvent], event_logger: Optional[EventLogger]
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


def _emit_message_event(
    event: MessageEvent, event_logger: Optional[EventLogger]
) -> None:
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


def _emit_choice_event(event: ChoiceEvent, event_logger: Optional[EventLogger]) -> None:
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
