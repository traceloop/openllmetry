"""Wrapper classes for FunctionCall tool execution instrumentation."""

import json
import time

from opentelemetry import context as context_api
from opentelemetry.instrumentation.agno.utils import dont_throw, should_send_prompts
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAIAttributes
from opentelemetry.semconv_ai import SpanAttributes, TraceloopSpanKindValues
from opentelemetry.trace import SpanKind
from opentelemetry.trace.status import Status, StatusCode


class _FunctionCallExecuteWrapper:
    """Wrapper for FunctionCall.execute() method to capture synchronous tool execution."""

    def __init__(self, tracer, duration_histogram, token_histogram):
        """Initialize the wrapper with OpenTelemetry instrumentation objects."""
        self._tracer = tracer
        self._duration_histogram = duration_histogram
        self._token_histogram = token_histogram

    @dont_throw
    def __call__(self, wrapped, instance, args, kwargs):
        """Wrap the FunctionCall.execute() call with tracing instrumentation."""
        if context_api.get_value(
            context_api._SUPPRESS_INSTRUMENTATION_KEY
        ) or context_api.get_value("suppress_agno_instrumentation"):
            return wrapped(*args, **kwargs)

        function_name = getattr(instance.function, 'name', 'unknown')
        span_name = f"{function_name}.tool"

        with self._tracer.start_as_current_span(
            span_name,
            kind=SpanKind.CLIENT,
        ) as span:
            try:
                span.set_attribute(GenAIAttributes.GEN_AI_SYSTEM, "agno")
                span.set_attribute(SpanAttributes.TRACELOOP_SPAN_KIND,
                                   TraceloopSpanKindValues.TOOL.value)
                span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, function_name)

                if hasattr(instance.function, 'description') and instance.function.description:
                    span.set_attribute("tool.description", instance.function.description)

                # Capture input arguments
                if should_send_prompts():
                    if hasattr(instance, 'arguments') and instance.arguments:
                        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_INPUT,
                                           json.dumps(instance.arguments))
                    elif kwargs:
                        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_INPUT,
                                           json.dumps(kwargs))

                start_time = time.time()

                result = wrapped(*args, **kwargs)

                duration = time.time() - start_time

                if result is not None and should_send_prompts():
                    span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, str(result))

                span.set_status(Status(StatusCode.OK))

                self._duration_histogram.record(
                    duration,
                    attributes={
                        GenAIAttributes.GEN_AI_SYSTEM: "agno",
                        SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.TOOL.value,
                    }
                )

                return result

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise


class _FunctionCallAExecuteWrapper:
    """Wrapper for FunctionCall.aexecute() method to capture asynchronous tool execution."""

    def __init__(self, tracer, duration_histogram, token_histogram):
        """Initialize the wrapper with OpenTelemetry instrumentation objects."""
        self._tracer = tracer
        self._duration_histogram = duration_histogram
        self._token_histogram = token_histogram

    @dont_throw
    async def __call__(self, wrapped, instance, args, kwargs):
        """Wrap the FunctionCall.aexecute() call with tracing instrumentation."""
        if context_api.get_value(
            context_api._SUPPRESS_INSTRUMENTATION_KEY
        ) or context_api.get_value("suppress_agno_instrumentation"):
            return await wrapped(*args, **kwargs)

        function_name = getattr(instance.function, 'name', 'unknown')
        span_name = f"{function_name}.tool"

        with self._tracer.start_as_current_span(
            span_name,
            kind=SpanKind.CLIENT,
        ) as span:
            try:
                span.set_attribute(GenAIAttributes.GEN_AI_SYSTEM, "agno")
                span.set_attribute(SpanAttributes.TRACELOOP_SPAN_KIND,
                                   TraceloopSpanKindValues.TOOL.value)
                span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, function_name)

                if hasattr(instance.function, 'description') and instance.function.description:
                    span.set_attribute("tool.description", instance.function.description)

                # Capture input arguments
                if should_send_prompts():
                    if hasattr(instance, 'arguments') and instance.arguments:
                        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_INPUT,
                                           json.dumps(instance.arguments))
                    elif kwargs:
                        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_INPUT,
                                           json.dumps(kwargs))

                start_time = time.time()

                result = await wrapped(*args, **kwargs)

                duration = time.time() - start_time

                if result is not None and should_send_prompts():
                    span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, str(result))

                span.set_status(Status(StatusCode.OK))

                self._duration_histogram.record(
                    duration,
                    attributes={
                        GenAIAttributes.GEN_AI_SYSTEM: "agno",
                        SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.TOOL.value,
                    }
                )

                return result

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
