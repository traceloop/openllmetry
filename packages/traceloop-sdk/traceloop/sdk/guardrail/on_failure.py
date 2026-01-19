"""
Built-in failure handlers for guardrails.

On failure handlers are called when a guard returns False.
They receive the full GuardedOutput and can raise, log, or perform custom actions.
"""
import logging
from typing import Any, Callable, TypeVar

from .model import GuardedOutput, GuardValidationError

T = TypeVar("T")
Z = TypeVar("Z")

logger = logging.getLogger("traceloop.guardrail")


class OnFailure:
    """Built-in failure handlers."""

    @staticmethod
    def raise_exception(
        message: str = "Guard validation failed",
    ) -> Callable[[GuardedOutput[Any, Any]], None]:
        """
        Raise GuardValidationError on failure.

        Args:
            message: Error message to include in the exception

        Example:
            on_failure=OnFailure.raise_exception("PII detected in response")
        """

        def handler(output: GuardedOutput[Any, Any]) -> None:
            raise GuardValidationError(message, output)

        return handler

    @staticmethod
    def log(
        level: int = logging.WARNING,
        message: str = "Guard validation failed",
    ) -> Callable[[GuardedOutput[Any, Any]], None]:
        """
        Log failure and continue (return the result anyway).

        Args:
            level: Logging level (default: WARNING)
            message: Log message prefix

        Example:
            on_failure=OnFailure.log(message="Toxic content detected")
        """

        def handler(output: GuardedOutput[Any, Any]) -> None:
            logger.log(level, f"{message}: guard_input={output.guard_input}")

        return handler

    @staticmethod
    def noop() -> Callable[[GuardedOutput[Any, Any]], None]:
        """
        Do nothing on failure (shadow mode).

        Useful for testing guards in production without affecting behavior.

        Example:
            on_failure=OnFailure.noop()  # Just observe, don't block
        """

        def handler(output: GuardedOutput[Any, Any]) -> None:
            pass

        return handler
