from dataclasses import asdict
from enum import Enum
from typing import Union

from opentelemetry._events import Event
from opentelemetry.instrumentation.vertexai.event_models import (
    ChoiceEvent,
    MessageEvent,
)
from opentelemetry.instrumentation.vertexai.utils import (
    dont_throw,
    should_emit_events,
    should_send_prompts,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from vertexai.generative_models import GenerationResponse


class Roles(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


VALID_MESSAGE_ROLES = {role.value for role in Roles}
"""The valid roles for naming the message event."""

EVENT_ATTRIBUTES = {
    GenAIAttributes.GEN_AI_SYSTEM: GenAIAttributes.GenAiSystemValues.VERTEX_AI.value
}
"""The attributes to be used for the event."""


def _parse_vertex_finish_reason(reason):
    if reason is None:
        return "unknown"

    finish_reason_map = {
        0: "unspecified",
        1: "stop",
        2: "max_tokens",
        3: "safety",
        4: "recitation",
        5: "other",
        6: "blocklist",
        7: "prohibited_content",
        8: "spii",
        9: "malformed_function_call",
    }

    if hasattr(reason, "value"):
        reason_value = reason.value
    else:
        reason_value = reason

    return finish_reason_map.get(reason_value, "unknown")


@dont_throw
def emit_prompt_events(args, event_logger):
    prompt = ""
    if args is not None and len(args) > 0:
        for arg in args:
            if isinstance(arg, str):
                prompt = f"{prompt}{arg}\n"
            elif isinstance(arg, list):
                for subarg in arg:
                    prompt = f"{prompt}{subarg}\n"
    emit_event(MessageEvent(content=prompt, role=Roles.USER.value), event_logger)


def emit_response_events(response, event_logger):
    if isinstance(response, str):
        emit_event(
            ChoiceEvent(
                index=0,
                message={"content": response, "role": Roles.ASSISTANT.value},
                finish_reason="unknown",
            ),
            event_logger,
        )
    elif isinstance(response, GenerationResponse):
        for candidate in response.candidates:
            emit_event(
                ChoiceEvent(
                    index=candidate.index,
                    message={
                        "content": candidate.text,
                        "role": Roles.ASSISTANT.value,
                    },
                    finish_reason=_parse_vertex_finish_reason(candidate.finish_reason),
                ),
                event_logger,
            )


def emit_event(event: Union[MessageEvent, ChoiceEvent], event_logger) -> None:
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


def _emit_message_event(event: MessageEvent, event_logger) -> None:
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


def _emit_choice_event(event: ChoiceEvent, event_logger) -> None:
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
