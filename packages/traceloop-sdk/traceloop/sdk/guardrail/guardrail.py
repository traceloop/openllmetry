"""
Guardrails class for running guarded operations through the Traceloop client.
"""
import asyncio
import time
from typing import TypeVar, Callable, Awaitable, Union, Any, Dict
from uuid import uuid4

import httpx
from opentelemetry.trace.status import Status, StatusCode

from traceloop.sdk.tracing import get_tracer
from traceloop.sdk.evaluator.evaluator import Evaluator

from .model import GuardedOutput

T = TypeVar("T")  # Result type
Z = TypeVar("Z")  # Guard input type

# Type aliases for guards and handlers
Guard = Union[Callable[[Z], bool], Callable[[Z], Awaitable[bool]]]
OnFailureHandler = Union[
    Callable[[GuardedOutput[T, Z]], None],
    Callable[[GuardedOutput[T, Z]], Awaitable[None]],
]


class Guardrails:
    """
    Guardrails class for running guarded operations.

    Access via the Traceloop client:
        client = Traceloop.init(api_key="...")
        result = await client.guardrails.run(
            func=my_agent,
            guard=lambda z: z["score"] > 0.8,
            on_failure=OnFailure.raise_exception("Quality check failed"),
        )
    """

    _evaluator: Evaluator
    _async_http: httpx.AsyncClient

    def __init__(self, async_http_client: httpx.AsyncClient):
        self._async_http = async_http_client
        self._evaluator = Evaluator(async_http_client)

    async def run(
        self,
        func: Callable[[], Awaitable[GuardedOutput[T, Z]]],
        guard: Guard[Z],
        on_failure: OnFailureHandler[T, Z],
    ) -> T:
        """
        Execute a function with guardrail protection.

        Args:
            func: Async function that returns GuardedOutput[T, Z].
                  Executed immediately inside run().

            guard: Function that receives Z (guard_input) and returns bool.
                   True = pass, False = fail.
                   Can be a lambda, custom function, or EvaluatorDetails.as_guard().

            on_failure: Called when guard returns False.
                        Receives the full GuardedOutput[T, Z].
                        Can raise, log, or perform custom actions.

        Returns:
            T: The result from GuardedOutput.result

        Raises:
            GuardValidationError: If guard fails and on_failure raises

        Example:
            result = await client.guardrails.run(
                func=generate_email,
                guard=EvaluatorMadeByTraceloop.pii_detector().as_guard(
                    condition=Condition.is_false("has_pii")
                ),
                on_failure=OnFailure.raise_exception("PII detected"),
            )
        """
        with get_tracer() as tracer:
            with tracer.start_as_current_span("guardrail.run") as span:
                start_time = time.perf_counter()

                # 1. Execute func
                output: GuardedOutput[T, Z] = await func()

                # 2. Run guard
                try:
                    guard_result = guard(output.guard_input)
                    if asyncio.iscoroutine(guard_result):
                        guard_result = await guard_result
                    guard_passed = bool(guard_result)
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    guard_passed = False

                duration_ms = (time.perf_counter() - start_time) * 1000

                # Set span attributes
                span.set_attribute("guardrail.passed", guard_passed)
                span.set_attribute("guardrail.duration_ms", duration_ms)

                # 3. Handle failure
                if not guard_passed:
                    span.set_status(
                        Status(StatusCode.ERROR, "Guard validation failed")
                    )
                    failure_result = on_failure(output)
                    if asyncio.iscoroutine(failure_result):
                        await failure_result

                # 4. Return result
                return output.result


    async def run_evaluator(
        self,
        input_data: Dict[str, Any],
        evaluator_slug: str,
        evaluator_version: str | None = None,
        evaluator_config: Dict[str, Any] | None = None,
        timeout_in_sec: int = 60,
    ) -> Dict[str, Any]:
        """
        Run a Traceloop evaluator directly and return its result.

        This is useful when you need to evaluate content without the full
        guardrail flow, or when building custom guard logic.

        Args:
            input_data: Dictionary of input fields for the evaluator
            evaluator_slug: The evaluator identifier (e.g., "pii-detector")
            evaluator_version: Optional version of the evaluator
            evaluator_config: Optional configuration for the evaluator
            timeout_in_sec: Timeout in seconds (default: 60)

        Returns:
            Dict containing the evaluator result

        Example:
            result = await client.guardrails.run_evaluator(
                input_data={"text": "Hello, my name is John"},
                evaluator_slug="pii-detector",
            )
            print(result)  # {"has_pii": True, "pii_types": ["PERSON"]}
        """
        # Generate internal IDs for guardrail execution (no experiment context)
        internal_task_id = f"guard-{uuid4().hex[:8]}"
        internal_experiment_id = f"guard-exp-{uuid4().hex[:8]}"
        internal_run_id = f"guard-run-{uuid4().hex[:8]}"

        eval_response = await self._evaluator.run_experiment_evaluator(
            evaluator_slug=evaluator_slug,
            task_id=internal_task_id,
            experiment_id=internal_experiment_id,
            experiment_run_id=internal_run_id,
            input=input_data,
            evaluator_version=evaluator_version,
            evaluator_config=evaluator_config,
            timeout_in_sec=timeout_in_sec,
        )
        return eval_response.result
