import logging
import time
from typing import Optional

from opentelemetry._logs import Logger
from opentelemetry.instrumentation.anthropic.config import Config
from opentelemetry.instrumentation.anthropic.event_emitter import (
    emit_streaming_response_events,
)
from opentelemetry.instrumentation.anthropic.span_utils import (
    set_streaming_response_attributes,
)
from opentelemetry.instrumentation.anthropic.utils import (
    count_prompt_tokens_from_request,
    dont_throw,
    error_metrics_attributes,
    set_span_attribute,
    shared_metrics_attributes,
    should_emit_events,
)
from opentelemetry.metrics import Counter, Histogram
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace.status import Status, StatusCode
from wrapt import ObjectProxy

logger = logging.getLogger(__name__)


@dont_throw
def _process_response_item(item, complete_response):
    if item.type == "message_start":
        complete_response["model"] = item.message.model
        complete_response["usage"] = dict(item.message.usage)
        complete_response["id"] = item.message.id
    elif item.type == "content_block_start":
        index = item.index
        if len(complete_response.get("events")) <= index:
            complete_response["events"].append(
                {"index": index, "text": "", "type": item.content_block.type}
            )
            if item.content_block.type == "tool_use":
                complete_response["events"][index]["id"] = item.content_block.id
                complete_response["events"][index]["name"] = item.content_block.name
                complete_response["events"][index]["input"] = """"""
    elif item.type == "content_block_delta":
        index = item.index
        if item.delta.type == "thinking_delta":
            complete_response["events"][index]["text"] += item.delta.thinking
        elif item.delta.type == "text_delta":
            complete_response["events"][index]["text"] += item.delta.text
        elif item.delta.type == "input_json_delta":
            complete_response["events"][index]["input"] += item.delta.partial_json
    elif item.type == "message_delta":
        for event in complete_response.get("events", []):
            event["finish_reason"] = item.delta.stop_reason
        if item.usage:
            if "usage" in complete_response:
                item_output_tokens = dict(item.usage).get("output_tokens", 0)
                existing_output_tokens = complete_response["usage"].get(
                    "output_tokens", 0
                )
                complete_response["usage"]["output_tokens"] = (
                    item_output_tokens + existing_output_tokens
                )
            else:
                complete_response["usage"] = dict(item.usage)


def _set_token_usage(
    span,
    complete_response,
    prompt_tokens,
    completion_tokens,
    metric_attributes: dict = {},
    token_histogram: Histogram = None,
    choice_counter: Counter = None,
):
    cache_read_tokens = (
        complete_response.get("usage", {}).get("cache_read_input_tokens", 0) or 0
    )
    cache_creation_tokens = (
        complete_response.get("usage", {}).get("cache_creation_input_tokens", 0) or 0
    )

    input_tokens = prompt_tokens + cache_read_tokens + cache_creation_tokens
    total_tokens = input_tokens + completion_tokens

    set_span_attribute(span, GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
    set_span_attribute(
        span, GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens
    )
    set_span_attribute(span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total_tokens)

    set_span_attribute(
        span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, complete_response.get("model")
    )

    if token_histogram and type(input_tokens) is int and input_tokens >= 0:
        token_histogram.record(
            input_tokens,
            attributes={
                **metric_attributes,
                GenAIAttributes.GEN_AI_TOKEN_TYPE: "input",
            },
        )

    if token_histogram and type(completion_tokens) is int and completion_tokens >= 0:
        token_histogram.record(
            completion_tokens,
            attributes={
                **metric_attributes,
                GenAIAttributes.GEN_AI_TOKEN_TYPE: "output",
            },
        )

    if type(complete_response.get("events")) is list and choice_counter:
        for event in complete_response.get("events"):
            choice_counter.add(
                1,
                attributes={
                    **metric_attributes,
                    SpanAttributes.LLM_RESPONSE_FINISH_REASON: event.get(
                        "finish_reason"
                    ),
                },
            )


def _handle_streaming_response(span, event_logger, complete_response):
    if should_emit_events() and event_logger:
        emit_streaming_response_events(event_logger, complete_response)
    else:
        if not span.is_recording():
            return
        set_streaming_response_attributes(span, complete_response.get("events"))


class AnthropicStream(ObjectProxy):
    """Wrapper for Anthropic streaming responses that handles instrumentation while preserving helper methods"""

    def __init__(
        self,
        span,
        response,
        instance,
        start_time,
        token_histogram: Histogram = None,
        choice_counter: Counter = None,
        duration_histogram: Histogram = None,
        exception_counter: Counter = None,
        event_logger: Optional[Logger] = None,
        kwargs: dict = {},
    ):
        super().__init__(response)

        self._span = span
        self._instance = instance
        self._start_time = start_time
        self._token_histogram = token_histogram
        self._choice_counter = choice_counter
        self._duration_histogram = duration_histogram
        self._exception_counter = exception_counter
        self._event_logger = event_logger
        self._kwargs = kwargs

        self._complete_response = {"events": [], "model": "", "usage": {}, "id": ""}
        self._instrumentation_completed = False

    def __getattr__(self, name):
        """Override helper methods to ensure they go through our instrumented iteration"""
        if name == 'get_final_message':
            return self._instrumented_get_final_message
        elif name == 'text_stream':
            return self._instrumented_text_stream
        elif name == 'until_done':
            return self._instrumented_until_done
        else:
            return super().__getattr__(name)

    def _instrumented_get_final_message(self):
        """Instrumented version of get_final_message that goes through our proxy"""
        for _ in self:
            pass
        original_get_final_message = getattr(self.__wrapped__, 'get_final_message')
        return original_get_final_message()

    @property
    def _instrumented_text_stream(self):
        """Instrumented version of text_stream that goes through our proxy"""
        def text_generator():
            for event in self:
                if (hasattr(event, 'delta') and
                    hasattr(event.delta, 'type') and
                    event.delta.type == 'text_delta' and
                        hasattr(event.delta, 'text')):
                    yield event.delta.text
        return text_generator()

    def _instrumented_until_done(self):
        """Instrumented version of until_done that goes through our proxy"""
        for _ in self:
            pass

    def __iter__(self):
        return self

    def __next__(self):
        try:
            item = self.__wrapped__.__next__()
        except StopIteration:
            # Stream is complete - handle instrumentation
            if not self._instrumentation_completed:
                self._complete_instrumentation()
            raise
        except Exception as e:
            attributes = error_metrics_attributes(e)
            if self._exception_counter:
                self._exception_counter.add(1, attributes=attributes)
            raise e
        _process_response_item(item, self._complete_response)
        return item

    def _handle_completion(self):
        """Handle completion logic"""
        metric_attributes = shared_metrics_attributes(self._complete_response)
        set_span_attribute(self._span, GenAIAttributes.GEN_AI_RESPONSE_ID, self._complete_response.get("id"))
        if self._duration_histogram:
            duration = time.time() - self._start_time
            self._duration_histogram.record(
                duration,
                attributes=metric_attributes,
            )

            # This mirrors the logic from build_from_streaming_response
            metric_attributes = shared_metrics_attributes(self._complete_response)
            set_span_attribute(self._span, GenAIAttributes.GEN_AI_RESPONSE_ID, self._complete_response.get("id"))

            if self._duration_histogram:
                duration = time.time() - self._start_time
                self._duration_histogram.record(
                    duration,
                    attributes=metric_attributes,
                )

            # Calculate token usage
            if Config.enrich_token_usage:
                try:
                    if usage := self._complete_response.get("usage"):
                        prompt_tokens = usage.get("input_tokens", 0) or 0
                    else:
                        prompt_tokens = count_prompt_tokens_from_request(self._instance, self._kwargs)

                    if usage := self._complete_response.get("usage"):
                        completion_tokens = usage.get("output_tokens", 0) or 0
                    else:
                        completion_content = ""
                        if self._complete_response.get("events"):
                            model_name = self._complete_response.get("model") or None
                            for event in self._complete_response.get("events"):
                                if event.get("text"):
                                    completion_content += event.get("text")

                            if model_name and hasattr(self._instance, "count_tokens"):
                                completion_tokens = self._instance.count_tokens(completion_content)

                    _set_token_usage(
                        self._span,
                        self._complete_response,
                        prompt_tokens,
                        completion_tokens,
                        metric_attributes,
                        self._token_histogram,
                        self._choice_counter,
                    )
                except Exception as e:
                    logger.warning("Failed to set token usage, error: %s", e)

            _handle_streaming_response(self._span, self._event_logger, self._complete_response)

            if self._span.is_recording():
                self._span.set_status(Status(StatusCode.OK))
                self._span.end()

            self._instrumentation_completed = True

    def _complete_instrumentation(self):
        """Complete the instrumentation when stream is fully consumed"""
        if self._instrumentation_completed:
            return
        self._handle_completion()


class AnthropicAsyncStream(ObjectProxy):
    """Wrapper for Anthropic async streaming responses that handles instrumentation while preserving helper methods"""

    def __init__(
        self,
        span,
        response,
        instance,
        start_time,
        token_histogram: Histogram = None,
        choice_counter: Counter = None,
        duration_histogram: Histogram = None,
        exception_counter: Counter = None,
        event_logger: Optional[Logger] = None,
        kwargs: dict = {},
    ):
        super().__init__(response)

        self._span = span
        self._instance = instance
        self._start_time = start_time
        self._token_histogram = token_histogram
        self._choice_counter = choice_counter
        self._duration_histogram = duration_histogram
        self._exception_counter = exception_counter
        self._event_logger = event_logger
        self._kwargs = kwargs

        self._complete_response = {"events": [], "model": "", "usage": {}, "id": ""}
        self._instrumentation_completed = False

    def __getattr__(self, name):
        """Override helper methods to ensure they go through our instrumented iteration"""
        if name == 'get_final_message':
            return self._instrumented_get_final_message
        elif name == 'text_stream':
            return self._instrumented_text_stream
        elif name == 'until_done':
            return self._instrumented_until_done
        else:
            return super().__getattr__(name)

    async def _instrumented_get_final_message(self):
        """Instrumented version of get_final_message that goes through our proxy"""
        # Consume the entire stream through our instrumentation
        async for _ in self:
            pass
        # Now call the original method to get the final message
        # We need to access the original method directly
        original_get_final_message = getattr(self.__wrapped__, 'get_final_message')
        return await original_get_final_message()

    @property
    def _instrumented_text_stream(self):
        """Instrumented version of text_stream that goes through our proxy"""
        async def text_generator():
            async for event in self:
                if (hasattr(event, 'delta') and
                    hasattr(event.delta, 'type') and
                    event.delta.type == 'text_delta' and
                        hasattr(event.delta, 'text')):
                    yield event.delta.text
        return text_generator()

    async def _instrumented_until_done(self):
        """Instrumented version of until_done that goes through our proxy"""
        async for _ in self:
            pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            item = await self.__wrapped__.__anext__()
        except StopAsyncIteration:
            # Stream is complete - handle instrumentation
            if not self._instrumentation_completed:
                self._complete_instrumentation()
            raise
        except Exception as e:
            # Handle errors during streaming
            if not self._instrumentation_completed:
                attributes = error_metrics_attributes(e)
                if self._exception_counter:
                    self._exception_counter.add(1, attributes=attributes)
                if self._span and self._span.is_recording():
                    self._span.set_status(Status(StatusCode.ERROR, str(e)))
                self._span.end()
                self._instrumentation_completed = True
            raise
        else:
            # Process the item for instrumentation
            _process_response_item(item, self._complete_response)
            return item

    def _complete_instrumentation(self):
        """Complete the instrumentation when stream is fully consumed"""
        if self._instrumentation_completed:
            return

        # This mirrors the logic from abuild_from_streaming_response
        metric_attributes = shared_metrics_attributes(self._complete_response)
        set_span_attribute(self._span, GenAIAttributes.GEN_AI_RESPONSE_ID, self._complete_response.get("id"))

        if self._duration_histogram:
            duration = time.time() - self._start_time
            self._duration_histogram.record(
                duration,
                attributes=metric_attributes,
            )

        # Calculate token usage
        if Config.enrich_token_usage:
            try:
                if usage := self._complete_response.get("usage"):
                    prompt_tokens = usage.get("input_tokens", 0)
                else:
                    prompt_tokens = count_prompt_tokens_from_request(self._instance, self._kwargs)

                if usage := self._complete_response.get("usage"):
                    completion_tokens = usage.get("output_tokens", 0)
                else:
                    completion_content = ""
                    if self._complete_response.get("events"):
                        model_name = self._complete_response.get("model") or None
                        for event in self._complete_response.get("events"):
                            if event.get("text"):
                                completion_content += event.get("text")

                        if model_name and hasattr(self._instance, "count_tokens"):
                            completion_tokens = self._instance.count_tokens(completion_content)

                _set_token_usage(
                    self._span,
                    self._complete_response,
                    prompt_tokens,
                    completion_tokens,
                    metric_attributes,
                    self._token_histogram,
                    self._choice_counter,
                )
            except Exception as e:
                logger.warning("Failed to set token usage, error: %s", str(e))

        _handle_streaming_response(self._span, self._event_logger, self._complete_response)

        if self._span.is_recording():
            self._span.set_status(Status(StatusCode.OK))
            self._span.end()

        self._instrumentation_completed = True


class WrappedMessageStreamManager:
    """Wrapper for MessageStreamManager that handles instrumentation"""

    def __init__(
        self,
        stream_manager,
        span,
        instance,
        start_time,
        token_histogram,
        choice_counter,
        duration_histogram,
        exception_counter,
        event_logger,
        kwargs,
    ):
        self._stream_manager = stream_manager
        self._span = span
        self._instance = instance
        self._start_time = start_time
        self._token_histogram = token_histogram
        self._choice_counter = choice_counter
        self._duration_histogram = duration_histogram
        self._exception_counter = exception_counter
        self._event_logger = event_logger
        self._kwargs = kwargs

    def __enter__(self):
        # Call the original stream manager's __enter__ to get the actual stream
        stream = self._stream_manager.__enter__()
        # Return the proxy that preserves helper methods
        return AnthropicStream(
            self._span,
            stream,
            self._instance,
            self._start_time,
            self._token_histogram,
            self._choice_counter,
            self._duration_histogram,
            self._exception_counter,
            self._event_logger,
            self._kwargs,
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._stream_manager.__exit__(exc_type, exc_val, exc_tb)

    def __getattr__(self, name):
        if name == '_complete_instrumentation':
            return self._complete_instrumentation
        return getattr(self._stream_manager, name)

    def _complete_instrumentation(self):
        """Complete the instrumentation when stream is fully consumed"""
        pass


class WrappedAsyncMessageStreamManager:
    """Wrapper for AsyncMessageStreamManager that handles instrumentation"""

    def __init__(
        self,
        stream_manager,
        span,
        instance,
        start_time,
        token_histogram,
        choice_counter,
        duration_histogram,
        exception_counter,
        event_logger,
        kwargs,
    ):
        self._stream_manager = stream_manager
        self._span = span
        self._instance = instance
        self._start_time = start_time
        self._token_histogram = token_histogram
        self._choice_counter = choice_counter
        self._duration_histogram = duration_histogram
        self._exception_counter = exception_counter
        self._event_logger = event_logger
        self._kwargs = kwargs

    async def __aenter__(self):
        # Call the original stream manager's __aenter__ to get the actual stream
        stream = await self._stream_manager.__aenter__()
        # Return the proxy that preserves helper methods
        return AnthropicAsyncStream(
            self._span,
            stream,
            self._instance,
            self._start_time,
            self._token_histogram,
            self._choice_counter,
            self._duration_histogram,
            self._exception_counter,
            self._event_logger,
            self._kwargs,
        )

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self._stream_manager.__aexit__(exc_type, exc_val, exc_tb)

    def __getattr__(self, name):
        if name == '_complete_instrumentation':
            return self._complete_instrumentation
        return getattr(self._stream_manager, name)

    def _complete_instrumentation(self):
        """Complete the instrumentation when stream is fully consumed"""
        pass
