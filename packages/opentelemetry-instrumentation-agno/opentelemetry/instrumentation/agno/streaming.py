import logging
import time

from opentelemetry.instrumentation.agno.utils import should_send_prompts
from opentelemetry.metrics import Histogram
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes, TraceloopSpanKindValues
from opentelemetry.trace.status import Status, StatusCode
from wrapt import ObjectProxy

logger = logging.getLogger(__name__)


class AgnoAsyncStream(ObjectProxy):
    """Wrapper for Agno async streaming responses that handles instrumentation"""

    def __init__(
        self,
        span,
        response,
        instance,
        start_time,
        duration_histogram: Histogram = None,
        token_histogram: Histogram = None,
    ):
        super().__init__(response)

        self._self_span = span
        self._self_instance = instance
        self._self_start_time = start_time
        self._self_duration_histogram = duration_histogram
        self._self_token_histogram = token_histogram
        self._self_events = []
        self._self_final_result = None
        self._self_instrumentation_completed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            event = await self.__wrapped__.__anext__()
        except StopAsyncIteration:
            if not self._self_instrumentation_completed:
                self._complete_instrumentation()
            raise
        except Exception as e:
            if not self._self_instrumentation_completed:
                if self._self_span and self._self_span.is_recording():
                    self._self_span.set_status(Status(StatusCode.ERROR, str(e)))
                    self._self_span.record_exception(e)
                    self._self_span.end()
                self._self_instrumentation_completed = True
            raise

        self._self_events.append(event)

        if hasattr(event, "event") and event.event == "run_response":
            self._self_final_result = event

        return event

    def _complete_instrumentation(self):
        """Complete the instrumentation when stream is fully consumed"""
        if self._self_instrumentation_completed:
            return

        try:
            duration = time.time() - self._self_start_time

            if self._self_final_result:
                result = self._self_final_result
                if hasattr(result, "content") and should_send_prompts():
                    self._self_span.set_attribute(
                        SpanAttributes.TRACELOOP_ENTITY_OUTPUT, str(result.content)
                    )

                if hasattr(result, "run_id"):
                    self._self_span.set_attribute("agno.run.id", result.run_id)

                if hasattr(result, "metrics"):
                    metrics = result.metrics
                    if hasattr(metrics, "input_tokens"):
                        self._self_span.set_attribute(
                            GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS,
                            metrics.input_tokens,
                        )
                    if hasattr(metrics, "output_tokens"):
                        self._self_span.set_attribute(
                            GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
                            metrics.output_tokens,
                        )
                    if hasattr(metrics, "total_tokens"):
                        self._self_span.set_attribute(
                            SpanAttributes.LLM_USAGE_TOTAL_TOKENS, metrics.total_tokens
                        )

            self._self_span.set_status(Status(StatusCode.OK))

            if self._self_duration_histogram:
                self._self_duration_histogram.record(
                    duration,
                    attributes={
                        GenAIAttributes.GEN_AI_SYSTEM: "agno",
                        SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.AGENT.value,
                    },
                )

        except Exception as e:
            logger.warning("Failed to complete instrumentation: %s", str(e))
        finally:
            if self._self_span.is_recording():
                self._self_span.end()
            self._self_instrumentation_completed = True


class AgnoStream(ObjectProxy):
    """Wrapper for Agno sync streaming responses that handles instrumentation"""

    def __init__(
        self,
        span,
        response,
        instance,
        start_time,
        duration_histogram: Histogram = None,
        token_histogram: Histogram = None,
    ):
        super().__init__(response)

        self._self_span = span
        self._self_instance = instance
        self._self_start_time = start_time
        self._self_duration_histogram = duration_histogram
        self._self_token_histogram = token_histogram
        self._self_events = []
        self._self_final_result = None
        self._self_instrumentation_completed = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            event = self.__wrapped__.__next__()
        except StopIteration:
            if not self._self_instrumentation_completed:
                self._complete_instrumentation()
            raise
        except Exception as e:
            if not self._self_instrumentation_completed:
                if self._self_span and self._self_span.is_recording():
                    self._self_span.set_status(Status(StatusCode.ERROR, str(e)))
                    self._self_span.record_exception(e)
                    self._self_span.end()
                self._self_instrumentation_completed = True
            raise

        self._self_events.append(event)

        if hasattr(event, "event") and event.event == "run_response":
            self._self_final_result = event

        return event

    def _complete_instrumentation(self):
        """Complete the instrumentation when stream is fully consumed"""
        if self._self_instrumentation_completed:
            return

        try:
            duration = time.time() - self._self_start_time

            if self._self_final_result:
                result = self._self_final_result
                if hasattr(result, "content") and should_send_prompts():
                    self._self_span.set_attribute(
                        SpanAttributes.TRACELOOP_ENTITY_OUTPUT, str(result.content)
                    )

                if hasattr(result, "run_id"):
                    self._self_span.set_attribute("agno.run.id", result.run_id)

                if hasattr(result, "metrics"):
                    metrics = result.metrics
                    if hasattr(metrics, "input_tokens"):
                        self._self_span.set_attribute(
                            GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS,
                            metrics.input_tokens,
                        )
                    if hasattr(metrics, "output_tokens"):
                        self._self_span.set_attribute(
                            GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
                            metrics.output_tokens,
                        )
                    if hasattr(metrics, "total_tokens"):
                        self._self_span.set_attribute(
                            SpanAttributes.LLM_USAGE_TOTAL_TOKENS, metrics.total_tokens
                        )

            self._self_span.set_status(Status(StatusCode.OK))

            if self._self_duration_histogram:
                self._self_duration_histogram.record(
                    duration,
                    attributes={
                        GenAIAttributes.GEN_AI_SYSTEM: "agno",
                        SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.AGENT.value,
                    },
                )

        except Exception as e:
            logger.warning("Failed to complete instrumentation: %s", str(e))
        finally:
            if self._self_span.is_recording():
                self._self_span.end()
            self._self_instrumentation_completed = True
