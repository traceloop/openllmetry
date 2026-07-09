from dataclasses import dataclass
from typing import Any, List, Literal, Optional, TypedDict


class _FunctionToolCall(TypedDict):
    function_name: str
    arguments: Optional[dict[str, Any]]


class ToolCall(TypedDict):
    """Represents a tool call in the AI model."""

    id: str
    function: _FunctionToolCall
    type: Literal["function"]


class CompletionMessage(TypedDict):
    """Represents a message in the AI model."""

    content: Any
    role: str = "assistant"


@dataclass
class MessageEvent:
    """Represents an input event for the AI model."""

    content: Any
    role: str = "user"
    tool_calls: Optional[List[ToolCall]] = None


@dataclass
class ChoiceEvent:
    """Represents a completion event for the AI model."""

    index: int
    message: CompletionMessage
    finish_reason: str = "unknown"
    tool_calls: Optional[List[ToolCall]] = None
