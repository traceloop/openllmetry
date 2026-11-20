from dataclasses import asdict
from enum import Enum
from typing import Union

from opentelemetry._logs import Logger, LogRecord
from opentelemetry.instrumentation.oci_genai.event_models import ChoiceEvent, MessageEvent
from opentelemetry.instrumentation.oci_genai.span_utils import (
    CHAT,
    EMBEDDINGS,
    OCI_GENAI_PROVIDER,
    RERANK,
    TEXT_COMPLETION,
    chat_request_to_messages,
    embedding_input_parts,
    response_to_output_messages,
    stream_choices_to_messages,
)
from opentelemetry.instrumentation.oci_genai.utils import (
    dont_throw,
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

EVENT_ATTRIBUTES = {GenAIAttributes.GEN_AI_PROVIDER_NAME: OCI_GENAI_PROVIDER}
"""The attributes to be used for the event."""


def _parts_to_content(parts):
    """Collapse OTel parts into event body content: a string when text-only, otherwise the parts list."""
    parts = [part for part in parts if part.get("type") != "tool_call"]
    if not parts:
        return None
    if all(part.get("type") == "text" for part in parts):
        return "\n".join(part.get("content", "") for part in parts)
    if len(parts) == 1 and parts[0].get("type") == "tool_call_response":
        return parts[0].get("response")
    return parts


def _parts_to_tool_calls(parts):
    tool_calls = [
        {
            "id": part.get("id") or "",
            "type": "function",
            "function": {"function_name": part.get("name"), "arguments": part.get("arguments")},
        }
        for part in parts
        if part.get("type") == "tool_call"
    ]
    return tool_calls or None


def _emit_message(message, event_logger):
    emit_event(
        MessageEvent(
            content=_parts_to_content(message["parts"]),
            role=message.get("role") or "user",
            tool_calls=_parts_to_tool_calls(message["parts"]),
        ),
        event_logger,
    )


def _emit_choices(messages, event_logger):
    for index, message in enumerate(messages):
        emit_event(
            ChoiceEvent(
                index=index,
                message={"content": _parts_to_content(message["parts"]), "role": message.get("role") or "assistant"},
                finish_reason=message.get("finish_reason") or None,
                tool_calls=_parts_to_tool_calls(message["parts"]),
            ),
            event_logger,
        )


@dont_throw
def emit_input_events(operation, details, event_logger):
    if operation == CHAT:
        messages, system_parts = chat_request_to_messages(getattr(details, "chat_request", None))
        if system_parts:
            _emit_message({"role": Roles.SYSTEM.value, "parts": system_parts}, event_logger)
        for message in messages:
            _emit_message(message, event_logger)
    elif operation == TEXT_COMPLETION:
        prompt = getattr(getattr(details, "inference_request", None), "prompt", None)
        if prompt:
            emit_event(MessageEvent(content=prompt, role=Roles.USER.value), event_logger)
    elif operation == EMBEDDINGS:
        for parts in embedding_input_parts(details):
            _emit_message({"role": Roles.USER.value, "parts": parts}, event_logger)
    elif operation == RERANK:
        for document in getattr(details, "documents", None) or []:
            emit_event(MessageEvent(content=document, role=Roles.SYSTEM.value), event_logger)
        query = getattr(details, "input", None)
        if query:
            emit_event(MessageEvent(content=query, role=Roles.USER.value), event_logger)


@dont_throw
def emit_response_events(operation, response, event_logger):
    _emit_choices(response_to_output_messages(operation, getattr(response, "data", None)), event_logger)


@dont_throw
def emit_streaming_response_events(accumulator, event_logger):
    _emit_choices(stream_choices_to_messages(accumulator.choices_list()), event_logger)


def emit_event(event: Union[MessageEvent, ChoiceEvent], event_logger: Union[Logger, None]) -> None:
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


def _emit_message_event(event: MessageEvent, event_logger: Logger) -> None:
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

    log_record = LogRecord(body=body, attributes=EVENT_ATTRIBUTES, event_name=name)
    event_logger.emit(log_record)


def _emit_choice_event(event: ChoiceEvent, event_logger: Logger) -> None:
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

    log_record = LogRecord(body=body, attributes=EVENT_ATTRIBUTES, event_name="gen_ai.choice")
    event_logger.emit(log_record)
