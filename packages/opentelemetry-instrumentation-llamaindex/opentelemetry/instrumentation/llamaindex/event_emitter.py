from dataclasses import asdict
from enum import Enum
from typing import Union

from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMChatStartEvent,
)
from llama_index.core.instrumentation.events.rerank import ReRankStartEvent
from opentelemetry._events import Event
from opentelemetry.instrumentation.llamaindex.event_models import (
    ChoiceEvent,
    MessageEvent,
)
from opentelemetry.instrumentation.llamaindex.utils import (
    should_emit_events,
    should_send_prompts,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from .config import Config


class Roles(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


VALID_MESSAGE_ROLES = {role.value for role in Roles}
"""The valid roles for naming the message event."""

EVENT_ATTRIBUTES = {GenAIAttributes.GEN_AI_SYSTEM: "llamaindex"}
"""The attributes to be used for the event."""


def emit_chat_message_events(event: LLMChatStartEvent):
    for message in event.messages:
        emit_event(MessageEvent(content=message.content, role=message.role.value))


def emit_chat_response_events(event: LLMChatEndEvent):
    if event.response:
        try:
            finish_reason = event.response.raw.get("choices", [{}])[0].get(
                "finish_reason", "unknown"
            )
        except (AttributeError, ValueError):
            finish_reason = "unknown"
        emit_choice_event(
            index=0,
            content=event.response.message.content,
            role=event.response.message.role.value,
            finish_reason=finish_reason,
        )


def emit_rerank_message_event(event: ReRankStartEvent):
    if event.query:
        if isinstance(event.query, str):
            emit_message_event(content=event.query, role="user")
        else:
            emit_message_event(content=event.query.query_str, role="user")


def emit_message_event(*, content, role: str):
    emit_event(MessageEvent(content=content, role=role))


def emit_choice_event(
    *,
    index: int = 0,
    content,
    role: str,
    finish_reason: str,
):
    emit_event(
        ChoiceEvent(
            index=index,
            message={"content": content, "role": role},
            finish_reason=finish_reason,
        )
    )


def emit_event(event: Union[MessageEvent, ChoiceEvent]) -> None:
    """
    Emit an event to the OpenTelemetry SDK.

    Args:
        event: The event to emit.
    """
    if not should_emit_events():
        return

    if isinstance(event, MessageEvent):
        _emit_message_event(event)
    elif isinstance(event, ChoiceEvent):
        _emit_choice_event(event)
    else:
        raise TypeError("Unsupported event type")


def _emit_message_event(event: MessageEvent) -> None:
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

    Config.event_logger.emit(Event(name=name, body=body, attributes=EVENT_ATTRIBUTES))


def _emit_choice_event(event: ChoiceEvent) -> None:
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

    Config.event_logger.emit(
        Event(name="gen_ai.choice", body=body, attributes=EVENT_ATTRIBUTES)
    )
