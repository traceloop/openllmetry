"""
Built-in failure handlers for guardrails.

On failure handlers are called when a guard returns False.
They receive the full GuardedOutput and can raise, log, or perform custom actions.
"""
import logging
from typing import Any, Callable, Type

from .model import GuardedFunctionOutput, GuardValidationError, FailureResult

logger = logging.getLogger("traceloop.guardrail")


class OnFailure:
    """Built-in failure handlers."""

    @staticmethod
    def raise_exception(
        message: str = "Guard validation failed",
        exception_type: Type[Exception] = GuardValidationError
    ) -> Callable[[GuardedFunctionOutput[Any, Any]], None]:
        """
        Raise GuardValidationError on failure.

        Args:
            message: Error message to include in the exception

        Example:
            on_failure=OnFailure.raise_exception("PII detected in response")
        """

        def handler(output: GuardedFunctionOutput[Any, Any]) -> None:
            raise exception_type(message, output)

        return handler

    @staticmethod
    def log(
        level: int = logging.WARNING,
        message: str = "Guard validation failed",
    ) -> Callable[[GuardedFunctionOutput[Any, Any]], None]:
        """
        Log failure and continue (return the result anyway).

        Args:
            level: Logging level (default: WARNING)
            message: Log message prefix

        Example:
            on_failure=OnFailure.log(message="Toxic content detected")
        """

        def handler(output: GuardedFunctionOutput[Any, Any]) -> None:
            logger.log(level, f"{message}: guard_input={output.guard_input}")

        return handler

    @staticmethod
    def noop() -> Callable[[GuardedFunctionOutput[Any, Any]], None]:
        """
        Do nothing on failure (shadow mode).

        Useful for testing guards in production without affecting behavior.

        Example:
            on_failure=OnFailure.noop()  # Just observe, don't block
        """

        def handler(output: GuardedFunctionOutput[Any, Any]) -> None:
            pass

        return handler

    @staticmethod
    def return_value(
        value: FailureResult,
    ) -> Callable[[GuardedFunctionOutput[Any, Any]], FailureResult]:
        """
        Return a fallback value on failure.

        Args:
            value: The value to return when guard fails

        Example:
            on_failure=OnFailure.return_value("Sorry, I can't help with that.")
            on_failure=OnFailure.return_value({"error": "blocked", "reason": "PII"})
        """

        def handler(output: GuardedFunctionOutput[Any, Any]) -> FailureResult:
            return value

        return handler
