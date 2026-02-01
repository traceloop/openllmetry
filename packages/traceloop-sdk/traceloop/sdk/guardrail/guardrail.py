"""
Guardrails class for running guarded operations through the Traceloop client.
"""
import asyncio
import inspect
import json
import time
from typing import Any, Callable, Awaitable, cast, Optional

import httpx
from pydantic import TypeAdapter
from opentelemetry.trace import Tracer, Span
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from traceloop.sdk.tracing import get_tracer
from .span_attributes import (
    GEN_AI_GUARDRAIL_NAME,
    GEN_AI_GUARDRAIL_STATUS,
    GEN_AI_GUARDRAIL_DURATION,
    GEN_AI_GUARDRAIL_GUARD_COUNT,
    GEN_AI_GUARDRAIL_FAILED_GUARD_COUNT,
    GEN_AI_GUARDRAIL_INPUT,
    GEN_AI_GUARDRAIL_ERROR_TYPE,
    GEN_AI_GUARDRAIL_ERROR_MESSAGE,
)
from traceloop.sdk.evaluator.evaluator import Evaluator
from .on_failure import OnFailure
from .default_mapper import default_input_mapper
from .model import (
    GuardedResult,
    Guard,
    OnFailureHandler,
    InputMapper,
    FailureResult,
    GuardExecutionError,
    GuardInputTypeError,
)


class Guardrails:
    """
    Guardrails class for running guarded operations.

    Access via the Traceloop client:
        client = Traceloop.init(api_key="...")

    Usage:
        g = client.guardrails.create(
            guards=[
                lambda z: z["score"] > 0.8,
                Guards.pii_detector(),
            ],
            on_failure=OnFailure.raise_exception("Guard failed"),
        )
        result = await g.run(my_function)
    """

    _evaluator: Evaluator
    _async_http: httpx.AsyncClient
    _guards: list[Guard]
    _on_failure: Optional[OnFailureHandler]
    _run_all: bool
    _parallel: bool
    _name: str

    def __init__(self, async_http_client: httpx.AsyncClient):
        self._async_http = async_http_client
        self._evaluator = Evaluator(async_http_client)
        self._guards = []
        self._on_failure = None
        self._run_all = False
        self._parallel = True
        self._name = ""

    def create(
        self,
        guards: list[Guard],
        on_failure: OnFailureHandler = OnFailure.raise_exception(),
        name: str = "",
        run_all: bool = False,
        parallel: bool = True,
    ) -> "Guardrails":
        """
        Create a new guardrail instance with the given guards and failure handler.

        Args:
            name: Identifier for this guardrail configuration. Used as the
                  gen_ai.guardrail span attribute.
            guards: List of guard functions. Each receives its corresponding
                    guard_input and returns bool. True = pass, False = fail.
            on_failure: Called when any guard returns False.
            run_all: If True, run all guards before handling failures.
                     If False (default), stop at first failure.
            parallel: If True (default), run guards in parallel.
                      If False, run guards sequentially.

        Returns:
            Guardrails: A new instance configured with the given guards.

        Example:
            g = client.guardrails.create(
                name="quality-check",
                guards=[
                    lambda z: z["score"] > 0.8,
                    Guards.pii_detector(),
                ],
                on_failure=OnFailure.raise_exception("Guard failed"),
                parallel=True,
            )
        """
        instance = Guardrails(self._async_http)
        instance._name = name
        instance._guards = guards
        instance._on_failure = on_failure
        instance._run_all = run_all
        instance._parallel = parallel
        return instance

    def _validate_inputs(self, guard_inputs: list[Any]) -> None:
        """
        Validate guard_inputs match guards in count and type.

        Uses Pydantic TypeAdapter for robust type validation including
        Pydantic models, TypedDicts, generic types, etc.

        Raises:
            ValueError: If the number of guard_inputs doesn't match the number of guards
            GuardInputTypeError: If a guard_input doesn't match the guard's expected type
        """
        # Length validation
        if len(guard_inputs) != len(self._guards):
            raise ValueError(
                f"Number of guard_inputs ({len(guard_inputs)}) "
                f"must match number of guards ({len(self._guards)})"
            )

        # Type validation
        for i, (guard, guard_input) in enumerate(zip(self._guards, guard_inputs)):
            try:
                signature = inspect.signature(guard)
            except (ValueError, TypeError):
                # Can't get signature (e.g., built-in functions)
                continue

            params = list(signature.parameters.values())
            if not params:
                continue

            first_param = params[0]
            expected_type = first_param.annotation

            # Skip if no type annotation (e.g., lambdas) or if type is Any
            if expected_type is inspect.Parameter.empty or expected_type is Any:
                continue

            try:
                TypeAdapter(expected_type).validate_python(guard_input)
            except Exception as e:
                guard_name = getattr(guard, "__name__", f"guard_{i}")
                error_detail = str(e).split('\n')[0] if str(e) else ""
                raise GuardInputTypeError(
                    message=(
                        f"Guard '{guard_name}' (index {i}) expected {expected_type}, "
                        f"but got {type(guard_input).__name__}. {error_detail}"
                    ),
                    guard_index=i,
                    expected_type=expected_type,
                    actual_type=type(guard_input),
                    validation_error=e,
                ) from e

    async def _run_single_guard(
        self,
        guard: Guard,
        guard_input: Any,
        index: int,
        tracer: Tracer,
    ) -> tuple[int, bool, Exception | None]:
        """Run a single guard with its own span and return (index, passed, exception)."""
        guard_name = getattr(guard, "__name__", f"guard_{index}")
        with tracer.start_as_current_span(f"{guard_name}.guard") as span:
            start_time = time.perf_counter()
            span.set_attribute(GenAIAttributes.GEN_AI_OPERATION_NAME, "guard")
            span.set_attribute(GEN_AI_GUARDRAIL_NAME, guard_name)

            # Capture guard input
            try:
                span.set_attribute(GEN_AI_GUARDRAIL_INPUT, json.dumps(guard_input))
            except (TypeError, ValueError):
                span.set_attribute(GEN_AI_GUARDRAIL_INPUT, str(guard_input))

            try:
                result = guard(guard_input)
                if asyncio.iscoroutine(result):
                    result = await result
                passed = bool(result)

                duration_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute(GEN_AI_GUARDRAIL_STATUS, "PASSED" if passed else "FAILED")
                span.set_attribute(GEN_AI_GUARDRAIL_DURATION, duration_ms)

                return (index, passed, None)
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute(GEN_AI_GUARDRAIL_STATUS, "FAILED")
                span.set_attribute(GEN_AI_GUARDRAIL_DURATION, duration_ms)
                span.set_attribute(GEN_AI_GUARDRAIL_ERROR_TYPE, type(e).__name__)
                span.set_attribute(GEN_AI_GUARDRAIL_ERROR_MESSAGE, str(e))
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                return (index, False, e)

    async def _run_guards(
        self,
        guard_inputs: list[Any],
        tracer: Tracer,
    ) -> list[tuple[int, bool, Exception | None]]:
        """Run guards either in parallel or sequentially based on configuration."""
        if self._parallel:
            tasks = [
                self._run_single_guard(guard, guard_input, i, tracer)
                for i, (guard, guard_input) in enumerate(zip(self._guards, guard_inputs))
            ]
            return await asyncio.gather(*tasks)

        # Sequential execution
        results = []
        for i, (guard, guard_input) in enumerate(zip(self._guards, guard_inputs)):
            result = await self._run_single_guard(guard, guard_input, i, tracer)
            results.append(result)
            _, passed, exception = result
            if (not passed or exception) and not self._run_all:
                break
        return results

    async def _execute_guards_with_tracing(
        self,
        guard_inputs: list[Any],
        tracer: Tracer,
        span: Optional[Span] = None,
        start_time: Optional[float] = None,
    ) -> tuple[bool, list[int]]:
        """
        Execute guards with tracing, handling validation, execution, and error checking.

        Args:
            guard_inputs: List of inputs for each guard
            tracer: The tracer instance
            span: The current tracing span (optional, for aggregated tracing)
            start_time: Start time for duration calculation (optional)

        Returns:
            tuple[bool, list[int]]: (all_passed, failed_indices)

        Raises:
            GuardExecutionError: If a guard function throws an exception
        """
        # Validate inputs
        self._validate_inputs(guard_inputs)

        # Run guards
        results = await self._run_guards(guard_inputs, tracer)

        # Check for execution errors
        for index, _, exception in results:
            if exception is not None:
                if span is not None:
                    span.set_status(Status(StatusCode.ERROR, str(exception)))
                    span.record_exception(exception)
                raise GuardExecutionError(
                    message="Guard execution failed",
                    original_exception=exception,
                    guard_input=guard_inputs[index],
                    guard_index=index,
                ) from exception

        # Check for failures
        failed_indices = [i for i, passed, _ in results if not passed]
        all_passed = len(failed_indices) == 0

        if span is not None and start_time is not None:
            duration_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute(GEN_AI_GUARDRAIL_STATUS, "PASSED" if all_passed else "FAILED")
            span.set_attribute(GEN_AI_GUARDRAIL_DURATION, duration_ms)
            span.set_attribute(GEN_AI_GUARDRAIL_GUARD_COUNT, len(self._guards))
            if failed_indices:
                span.set_attribute(GEN_AI_GUARDRAIL_FAILED_GUARD_COUNT, len(failed_indices))

        return all_passed, failed_indices

    async def run(
        self,
        func_to_guard: Callable[[], Awaitable[Any]],
        input_mapper: InputMapper | None = None,
    ) -> Any | FailureResult:
        """
        Execute a function with guardrail protection.

        Must call create() first to configure guards and on_failure.

        Args:
            func_to_guard: Async function that returns any type.
            input_mapper: Optional function to convert output to guard inputs.
                          If not provided, default mapper handles str and dict.

        Returns:
            The result from func_to_guard, or the on_failure return value.

        Raises:
            GuardValidationError: If any guard returns False and on_failure raises
            GuardExecutionError: If a guard function throws an exception
            ValueError: If create() was not called or guard_inputs length doesn't match

        Example:
            g = client.guardrails.create(
                guards=[Guards.toxicity_detector()],
                on_failure=OnFailure.raise_exception("Quality check failed"),
            )
            result = await g.run(generate_email)

            # With custom mapper
            result = await g.run(
                generate_custom_response,
                input_mapper=lambda r: [{"text": r.content}]
            )
        """
        if not self._guards or self._on_failure is None:
            raise ValueError("Must call create() before run()")

        with get_tracer() as tracer:
            span_name = f"{self._name}.guardrail" if self._name else "guardrail"
            with tracer.start_as_current_span(span_name) as span:
                start_time = time.perf_counter()
                span.set_attribute(GenAIAttributes.GEN_AI_OPERATION_NAME, "guardrail.run")
                if self._name:
                    span.set_attribute(GEN_AI_GUARDRAIL_NAME, self._name)

                # 1. Execute func_to_guard
                result = await func_to_guard()

                # 2. Convert result to guard inputs
                if input_mapper:
                    guard_inputs = input_mapper(result)
                else:
                    guard_inputs = default_input_mapper(result, len(self._guards))

                # 3. Execute guards with tracing
                all_passed, _ = await self._execute_guards_with_tracing(
                    guard_inputs, tracer, span, start_time
                )

                # 4. Handle failure
                if not all_passed:
                    guarded_result = GuardedResult(result=result, guard_inputs=guard_inputs)
                    failure_result = self._on_failure(guarded_result)
                    if asyncio.iscoroutine(failure_result):
                        failure_result = await failure_result
                    return cast(FailureResult, failure_result)

                # 5. All guards passed, return result
                return result

    async def validate(
        self,
        guard_inputs: list[Any],
        on_failure: Optional[OnFailureHandler] = None,
    ) -> bool:
        """
        Run guards on inputs directly, without wrapping in a function.

        Must call create() first to configure guards.

        Args:
            guard_inputs: List of inputs for each guard (must match number of guards)
            on_failure: Optional handler to override the class-configured on_failure

        Returns:
            bool: True if all guards pass, False if any guard fails

        Raises:
            GuardExecutionError: If a guard function throws an exception
            ValueError: If create() was not called or guard_inputs length doesn't match

        Example:
            g = client.guardrails.create(
                guards=[lambda z: z["score"] > 0.8],
                on_failure=OnFailure.log(),
            )
            passed = await g.validate([{"score": 0.9}])  # Returns True
        """
        if not self._guards:
            raise ValueError("Must call create() before validate()")

        failure_handler = on_failure if on_failure is not None else self._on_failure

        with get_tracer() as tracer:
            all_passed, _ = await self._execute_guards_with_tracing(
                guard_inputs, tracer
            )

            # Handle failure
            if not all_passed and failure_handler is not None:
                guarded_result = GuardedResult(result=None, guard_inputs=guard_inputs)
                failure_result = failure_handler(guarded_result)
                if asyncio.iscoroutine(failure_result):
                    await failure_result

            return all_passed
