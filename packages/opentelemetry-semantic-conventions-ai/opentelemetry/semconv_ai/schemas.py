"""
This module provides TypedDict definitions for the JSON values used
in Generative AI span attributes, including `gen_ai.input_messages`,
`gen_ai.output_messages`, and `gen_ai.system_instructions`.

Schemas can be found at
https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-input-messages.json
https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-output-messages.json
https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-system-instructions.json

This file is the Python implementation of the JSON schemas.
"""

from typing import Any, Literal, TypedDict
from typing_extensions import NotRequired


Modality = Literal["image", "video", "audio"]
Role = Literal["system", "user", "assistant", "tool"]
FinishReason = Literal["stop", "length", "content_filter", "tool_call", "error"]


class BlobPart(TypedDict):
    """Represents blob binary data sent inline to the model"""
    # The type of the content captured in this part.
    type: Literal["blob"]
    # The general modality of the data if it is known.
    # Instrumentations SHOULD also set the mimeType field if the specific type is known.
    modality: Modality | str
    # Raw bytes of the attached data. This field SHOULD be encoded as a base64 string when transmitted as JSON.
    content: str | bytes
    # The IANA MIME type of the attached data.
    mime_type: NotRequired[str | None] = None


class FilePartBase(TypedDict):
    """Represents an external referenced file sent to the model by file id"""
    # The type of the content captured in this part.
    type: Literal["file"]
    # The general modality of the data if it is known.
    # Instrumentations SHOULD also set the mimeType field if the specific type is known.
    modality: Modality | str
    # An identifier referencing a file that was pre-uploaded to the provider.
    file_id: str
    # The IANA MIME type of the attached data.
    mime_type: NotRequired[str | None] = None


class FilePart(FilePartBase, dict):
    # Allows for additional properties.
    pass


class ReasoningPartBase(TypedDict):
    """Represents reasoning/thinking content received from the model."""
    # The type of the content captured in this part.
    type: Literal["reasoning"]
    # Reasoning/thinking content received from the model.
    content: str


class ReasoningPart(ReasoningPartBase, dict):
    # Allows for additional properties.
    pass


class TextPartBase(TypedDict):
    """Represents text content sent to or received from the model."""
    # The type of the content captured in this part.
    type: Literal["text"]
    # Text content sent to or received from the model.
    content: str


class TextPart(TextPartBase, dict):
    # Allows for additional properties.
    pass


class ToolCallRequestPartBase(TypedDict):
    """Represents a tool call requested by the model."""
    # The type of the content captured in this part.
    type: Literal["tool_call"]
    # Name of the tool.
    name: str
    # Unique identifier for the tool call.
    id: NotRequired[str | None] = None
    # Arguments for the tool call.
    arguments: NotRequired[Any] = None


class ToolCallRequestPart(ToolCallRequestPartBase, dict):
    # Allows for additional properties.
    pass


class ToolCallResponsePartBase(TypedDict):
    """Represents a tool call result sent to the model or a built-in tool call outcome and details."""
    # The type of the content captured in this part.
    type: Literal["tool_call_response"]
    # Response from the tool call.
    response: Any
    # Unique tool call identifier.
    id: NotRequired[str | None] = None


class ToolCallResponsePart(ToolCallResponsePartBase, dict):
    # Allows for additional properties.
    pass


class UriPartBase(TypedDict):
    """Represents an external referenced file sent to the model by URI"""
    # The type of the content captured in this part.
    type: Literal["uri"]
    # The general modality of the data if it is known.
    # Instrumentations SHOULD also set the mimeType field if the specific type is known.
    modality: Modality | str
    # A URI referencing attached data. It should not be a base64 data URL, which should use the `blob` part instead.
    # The URI may use a scheme known to the provider api (e.g. `gs://bucket/object.png`),
    # or be a publicly accessible location.
    uri: str
    # The IANA MIME type of the attached data.
    mime_type: NotRequired[str | None] = None


class UriPart(UriPartBase, dict):
    # Allows for additional properties.
    pass


class GenericPartBase(TypedDict):
    """Represents an arbitrary message part with any type and properties.
    This allows for extensibility with custom message part types.
    """
    # The type of the content captured in this part.
    type: str


class GenericPart(GenericPartBase, dict):
    # Allows for additional properties.
    pass


class InputChatMessageBase(TypedDict):
    # Role of the entity that created the message.
    role: Role | str
    # List of message parts that make up the message content.
    parts: list[
        TextPart
        | ToolCallRequestPart
        | ToolCallResponsePart
        | BlobPart
        | FilePart
        | UriPart
        | ReasoningPart
        | GenericPart
    ]
    # The name of the participant.
    name: NotRequired[str | None] = None


class OutputMessageBase(TypedDict):
    """Represents an output message generated by the model or agent.
    The output message captures specific response (choice, candidate).
    """
    # Role of the entity that created the message.
    role: Role | str
    # List of message parts that make up the message content.
    parts: list[
        TextPart
        | ToolCallRequestPart
        | ToolCallResponsePart
        | BlobPart
        | FilePart
        | UriPart
        | ReasoningPart
        | GenericPart
    ]
    finish_reason: FinishReason | str
    # The name of the participant.
    name: NotRequired[str | None] = None

class OutputMessage(OutputMessageBase, dict):
    # Allows for additional properties.
    pass


class ChatMessage(InputChatMessageBase, dict):
    # Allows for additional properties.
    pass

# Represents the list of input messages sent to the model.
InputMessages = list[ChatMessage]
# Represents the list of output messages generated by the model or agent.
OutputMessages = list[OutputMessage]
# Represents the list of system instructions sent to the model.
SystemInstructions = list[
    TextPart
    | ToolCallRequestPart
    | ToolCallResponsePart
    | BlobPart
    | FilePart
    | UriPart
    | ReasoningPart
    | GenericPart
]
