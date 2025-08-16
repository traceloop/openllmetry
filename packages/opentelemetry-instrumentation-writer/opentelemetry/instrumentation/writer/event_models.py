from dataclasses import dataclass
from typing import Any, List, Literal, Optional, TypedDict


class _FunctionToolCall(TypedDict):
    name: str
    arguments: Optional[dict[str, Any]]


class ToolCall(TypedDict):
    id: str
    function: _FunctionToolCall
    type: Literal["function"]


class CompletionMessage(TypedDict):
    content: Any
    role: str = "assistant"


@dataclass
class MessageEvent:
    content: Any
    role: str = "user"
    tool_calls: Optional[List[ToolCall]] = None


@dataclass
class ChoiceEvent:
    index: int
    message: CompletionMessage
    finish_reason: str = "unknown"
    tool_calls: Optional[List[ToolCall]] = None
