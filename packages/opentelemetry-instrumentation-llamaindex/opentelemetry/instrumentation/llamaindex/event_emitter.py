from dataclasses import asdict
from enum import Enum
from typing import Optional, Union

from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMChatStartEvent,
)
from llama_index.core.instrumentation.events.rerank import ReRankStartEvent

from opentelemetry._logs import LogRecord
from opentelemetry.instrumentation.llamaindex.event_models import (
    ChoiceEvent,
    MessageEvent,
)
from opentelemetry.instrumentation.llamaindex.utils import (
    should_emit_events,
    should_send_prompts,
)
from opentelemetry.instrumentation.llamaindex._response_utils import extract_finish_reasons
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


def _event_attributes(provider_name: Optional[str] = None) -> dict:
    """Build event attributes with the actual LLM provider name."""
    return {GenAIAttributes.GEN_AI_PROVIDER_NAME: provider_name or "llamaindex"}


def emit_chat_message_events(event: LLMChatStartEvent, provider_name: Optional[str] = None):
    for message in event.messages:
        emit_event(MessageEvent(content=message.content, role=message.role.value), provider_name=provider_name)


def emit_chat_response_events(event: LLMChatEndEvent, provider_name: Optional[str] = None):
    if event.response:
        reasons = extract_finish_reasons(event.response.raw) if event.response.raw else []
        finish_reason = reasons[0] if reasons else ""
        emit_choice_event(
            index=0,
            content=event.response.message.content,
            role=event.response.message.role.value,
            finish_reason=finish_reason,
            provider_name=provider_name,
        )


def emit_rerank_message_event(event: ReRankStartEvent, provider_name: Optional[str] = None):
    if event.query:
        if isinstance(event.query, str):
            emit_message_event(content=event.query, role="user", provider_name=provider_name)
        else:
            emit_message_event(content=event.query.query_str, role="user", provider_name=provider_name)


def emit_message_event(*, content, role: str, provider_name: Optional[str] = None):
    emit_event(MessageEvent(content=content, role=role), provider_name=provider_name)


def emit_choice_event(
    *,
    index: int = 0,
    content,
    role: str,
    finish_reason: str,
    provider_name: Optional[str] = None,
):
    emit_event(
        ChoiceEvent(
            index=index,
            message={"content": content, "role": role},
            finish_reason=finish_reason,
        ),
        provider_name=provider_name,
    )


def emit_event(event: Union[MessageEvent, ChoiceEvent], provider_name: Optional[str] = None) -> None:
    """
    Emit an event to the OpenTelemetry SDK.

    Args:
        event: The event to emit.
        provider_name: The actual LLM provider name (e.g. "openai", "anthropic").
    """
    if not should_emit_events():
        return

    if isinstance(event, MessageEvent):
        _emit_message_event(event, provider_name=provider_name)
    elif isinstance(event, ChoiceEvent):
        _emit_choice_event(event, provider_name=provider_name)
    else:
        raise TypeError("Unsupported event type")


def _emit_message_event(event: MessageEvent, provider_name: Optional[str] = None) -> None:
    body = asdict(event)
    attrs = _event_attributes(provider_name)

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

    log_record = LogRecord(
        body=body,
        attributes=attrs,
        event_name=name
    )
    Config.event_logger.emit(log_record)


def _emit_choice_event(event: ChoiceEvent, provider_name: Optional[str] = None) -> None:
    body = asdict(event)
    attrs = _event_attributes(provider_name)

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

    log_record = LogRecord(
        body=body,
        attributes=attrs,
        event_name="gen_ai.choice"
    )
    Config.event_logger.emit(log_record)
