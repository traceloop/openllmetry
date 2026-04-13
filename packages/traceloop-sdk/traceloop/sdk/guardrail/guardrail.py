"""
Guardrails class for running guarded operations.
"""
import asyncio
import httpx
import inspect
import json
import time
from typing import Any, Callable, Awaitable, Dict, Type, cast, Optional

from pydantic import BaseModel

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
from .on_failure import OnFailureInput, resolve_on_failure
from .default_mapper import default_input_mapper
from traceloop.sdk.evaluator.model import (
    GuardrailResponse,
)
from traceloop.sdk.evaluator.evaluator import _validate_evaluator_input, _extract_error_from_response
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

    Usage (constructor kwargs):
        g = Guardrails(
            pii_guard(),
            toxicity_guard(),
            on_failure="raise",
        )
        result = await g.run(my_function)

    Usage (builder pattern):
        g = Guardrails(
            pii_guard(),
            toxicity_guard(),
        ).raise_on_failure().parallel()
        result = await g.run(my_function)
    """

    _guards: list[Guard]
    _on_failure: OnFailureHandler
    _run_all: bool
    _parallel: bool
    _name: str

    def __init__(
        self,
        *guards: Guard,
        on_failure: OnFailureInput = "raise",
        name: str = "",
        run_all: bool = False,
        parallel: bool = True,
    ):
        """
        Create a new guardrail instance.

        Args:
            *guards: Guard functions as positional arguments. Each receives its
                     corresponding guard_input and returns bool. True = pass,
                     False = fail.
            on_failure: Called when any guard returns False. Can be:
                - "raise": Raise GuardValidationError (default)
                - "log": Log warning and return result
                - "ignore": Return result silently (shadow mode)
                - Any other string: Return that string as fallback
                - Callable: Custom OnFailureHandler
            name: Identifier for this guardrail configuration. Used as the
                  gen_ai.guardrail span attribute.
            run_all: If True, run all guards before handling failures.
                     If False (default), stop at first failure.
            parallel: If True (default), run guards in parallel.
                      If False, run guards sequentially.
        """
        self._guards = list(guards)
        self._on_failure = resolve_on_failure(on_failure)
        self._name = name
        self._run_all = run_all
        self._parallel = parallel

    # -- Builder methods (all return self for chaining) --

    def parallel(self) -> "Guardrails":
        """Run guards in parallel (default). Returns self for chaining."""
        self._parallel = True
        return self

    def sequential(self) -> "Guardrails":
        """Run guards sequentially. Returns self for chaining."""
        self._parallel = False
        return self

    def run_all(self) -> "Guardrails":
        """Run all guards even after a failure. Returns self for chaining."""
        self._run_all = True
        return self

    def fail_fast(self) -> "Guardrails":
        """Stop at the first guard failure (default). Returns self for chaining."""
        self._run_all = False
        return self

    def raise_on_failure(self) -> "Guardrails":
        """Raise GuardValidationError on failure. Returns self for chaining."""
        self._on_failure = resolve_on_failure("raise")
        return self

    def log_on_failure(self) -> "Guardrails":
        """Log a warning on failure and return the result. Returns self for chaining."""
        self._on_failure = resolve_on_failure("log")
        return self

    def ignore_on_failure(self) -> "Guardrails":
        """Silently return the result on failure (shadow mode). Returns self for chaining."""
        self._on_failure = resolve_on_failure("ignore")
        return self

    def on_failure(self, handler: OnFailureInput) -> "Guardrails":
        """Set a custom failure handler. Accepts a string or callable. Returns self for chaining."""
        self._on_failure = resolve_on_failure(handler)
        return self

    def named(self, name: str) -> "Guardrails":
        """Set the guardrail name (used in span attributes). Returns self for chaining."""
        self._name = name
        return self

    async def run(
        self,
        func_to_guard: Callable[..., Awaitable[Any]],
        *args: Any,
        input_mapper: InputMapper | None = None,
        **kwargs: Any,
    ) -> Any | FailureResult:
        """
        Execute a function with guardrail protection.

        Args:
            func_to_guard: Async function that returns any type.
            *args: Positional arguments to pass to func_to_guard.
            input_mapper: Optional function to convert output to guard inputs.
                          If not provided, default mapper handles str and dict.
            **kwargs: Keyword arguments to pass to func_to_guard.

        Returns:
            The result from func_to_guard, or the on_failure return value.

        Raises:
            GuardValidationError: If any guard returns False and on_failure raises
            GuardExecutionError: If a guard function throws an exception

        Example:
            g = Guardrails(
                toxicity_guard(),
                on_failure="raise",
            )
            result = await g.run(generate_email)

            # With arguments
            result = await g.run(generate_response, user_prompt)

            # With custom mapper
            result = await g.run(
                generate_custom_response,
                input_mapper=lambda r: [{"text": r.content}]
            )
        """

        with get_tracer() as tracer:
            span_name = f"{self._name}.guardrail" if self._name else "guardrail"
            with tracer.start_as_current_span(span_name) as span:
                start_time = time.perf_counter()
                span.set_attribute(GenAIAttributes.GEN_AI_OPERATION_NAME, "guardrail.run")
                if self._name:
                    span.set_attribute(GEN_AI_GUARDRAIL_NAME, self._name)

                # 1. Execute func_to_guard
                result = await func_to_guard(*args, **kwargs)

                # 2. Convert result to guard inputs
                if input_mapper:
                    mapped = input_mapper(result)
                    if isinstance(mapped, dict):
                        guard_inputs = self._resolve_dict_inputs(mapped)
                    else:
                        guard_inputs = mapped
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
        guard_inputs: list[Any] | dict[str, Any],
        on_failure: Optional[OnFailureHandler] = None,
    ) -> bool:
        """
        Run guards on inputs directly, without wrapping in a function.

        Args:
            guard_inputs: Inputs for each guard. Can be:
                - A list (positional, must match number of guards)
                - A dict keyed by guard name (order-independent)
            on_failure: Optional handler to override the configured on_failure

        Returns:
            bool: True if all guards pass, False if any guard fails

        Raises:
            GuardExecutionError: If a guard function throws an exception

        Example:
            g = Guardrails(
                lambda z: z["score"] > 0.8,
                on_failure="log",
            )
            passed = await g.validate([{"score": 0.9}])  # Returns True
        """

        if isinstance(guard_inputs, dict):
            guard_inputs = self._resolve_dict_inputs(guard_inputs)

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

    async def execute_evaluator(
        self,
        evaluator_slug: str,
        input: Dict[str, Any],
        async_http_client: httpx.AsyncClient,
        timeout_in_sec: int = 120,
        evaluator_config: Optional[Dict[str, Any]] = None,
        input_schema: Optional[Type[BaseModel]] = None,
    ) -> GuardrailResponse:
        """Execute an evaluator via /v2/guardrails/{slug}/execute."""
        _validate_evaluator_input(evaluator_slug, input)

        body: Dict[str, Any] = {"input": input}
        if evaluator_config is not None:
            body["config"] = evaluator_config

        if input_schema is not None:
            body = input_schema(**body).model_dump()

        return await self._execute_guardrail_request(
            evaluator_slug, body, async_http_client, timeout_in_sec
        )

    def _resolve_dict_inputs(self, mapped: dict[str, Any]) -> list[Any]:
        """
        Resolve a dict of guard inputs keyed by guard name to an ordered list.

        Args:
            mapped: Dict mapping guard names to their inputs.

        Returns:
            List of guard inputs in the same order as self._guards.

        Raises:
            ValueError: If a guard name is missing from the dict.
        """
        guard_inputs = []
        for i, guard in enumerate(self._guards):
            guard_name = getattr(guard, "__name__", f"guard_{i}")
            if guard_name not in mapped:
                available = list(mapped.keys())
                names = [
                    getattr(g, "__name__", f"guard_{j}")
                    for j, g in enumerate(self._guards)
                ]
                raise ValueError(
                    f"input_mapper dict missing key '{guard_name}' for guard at index {i}. "
                    f"Available keys: {available}. "
                    f"Guard names: {names}"
                )
            guard_inputs.append(mapped[guard_name])
        return guard_inputs

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

    async def _execute_guardrail_request(
        self,
        evaluator_slug: str,
        body: Dict[str, Any],
        async_http_client: httpx.AsyncClient,
        timeout_in_sec: int = 120,
    ) -> GuardrailResponse:
        """Execute guardrail evaluator request and return response."""
        full_url = f"/v2/guardrails/{evaluator_slug}/execute"
        response = await async_http_client.post(
            full_url, json=body, timeout=httpx.Timeout(timeout_in_sec)
        )
        if response.status_code != 200:
            error_detail = _extract_error_from_response(response)
            raise Exception(
                f"Failed to execute guardrail evaluator '{evaluator_slug}': "
                f"{response.status_code} - {error_detail}"
            )
        result_data = response.json()
        return GuardrailResponse(**result_data)
