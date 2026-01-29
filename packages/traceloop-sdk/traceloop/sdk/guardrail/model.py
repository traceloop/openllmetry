"""
Data models and type definitions for the guardrail system.
"""
from dataclasses import dataclass
from typing import TypeVar, Generic, Any, Union, Callable, Awaitable

T = TypeVar("T")
GuardInput = TypeVar("GuardInput")
FailureResult = TypeVar("FailureResult")
GuardedFunctionResult = TypeVar("GuardedFunctionResult")

# Type aliases for guards and handlers
Guard = Union[Callable[[GuardInput], bool], Callable[[GuardInput], Awaitable[bool]]]
OnFailureHandler = Union[
    Callable[
        ["GuardedOutput[T, GuardInput]"], FailureResult
    ],
    Callable[
        ["GuardedOutput[T, GuardInput]"],
        Awaitable[FailureResult],
    ],
]


@dataclass
class GuardedOutput(Generic[T, GuardInput]):
    """
    Container that separates what to return from what to evaluate.

    Attributes:
        result: The original value returned by the guarded function
        guard_inputs: List of inputs for each guard (must match number of guards)

    Example:
        async def generate_email() -> GuardedOutput[str, dict]:
            text = await llm.complete("Write a customer email...")
            return GuardedOutput(
                result=text,
                guard_inputs=[{"text": text, "word_count": len(text.split())}],
            )
    """

    result: T
    guard_inputs: list[GuardInput]


class GuardValidationError(Exception):
    """
    Raised when guard fails and on_failure handler raises.

    Attributes:
        message: Error description
        output: The full GuardedOutput that failed validation
    """

    def __init__(self, message: str, output: GuardedOutput[Any, Any]):
        self.output = output
        super().__init__(message)

    def __str__(self) -> str:
        return f"{self.args[0]} (guard_inputs: {self.output.guard_inputs})"


class GuardExecutionError(Exception):
    """
    Raised when the guard function itself throws an exception during execution.

    This is different from GuardValidationError which is raised when the guard
    returns False. GuardExecutionError indicates the guard could not complete.

    Attributes:
        message: Error description
        original_exception: The exception that was raised by the guard
        guard_input: The input that was passed to the guard
        guard_index: The index of the guard that failed (0-based)
    """

    def __init__(
        self,
        message: str,
        original_exception: Exception,
        guard_input: Any,
        guard_index: int | None = None,
    ):
        self.original_exception = original_exception
        self.guard_input = guard_input
        self.guard_index = guard_index
        super().__init__(message)

    def __str__(self) -> str:
        index_info = f" [guard {self.guard_index}]" if self.guard_index is not None else ""
        return (
            f"{self.args[0]}{index_info}: {self.original_exception} "
            f"(guard_input: {self.guard_input})"
        )


class GuardInputTypeError(Exception):
    """
    Raised when a guard_input doesn't match the expected type of its guard function.

    This error occurs during input validation before guards are executed.

    Attributes:
        message: Error description
        guard_index: The index of the guard with type mismatch (0-based)
        expected_type: The type annotation from the guard function
        actual_type: The actual type of the guard_input
        validation_error: The underlying Pydantic validation error
    """

    def __init__(
        self,
        message: str,
        guard_index: int,
        expected_type: Any,
        actual_type: type,
        validation_error: Exception | None = None,
    ):
        self.guard_index = guard_index
        self.expected_type = expected_type
        self.actual_type = actual_type
        self.validation_error = validation_error
        super().__init__(message)

    def __str__(self) -> str:
        return (
            f"{self.args[0]} [guard {self.guard_index}]: "
            f"expected {self.expected_type}, got {self.actual_type.__name__}"
        )
