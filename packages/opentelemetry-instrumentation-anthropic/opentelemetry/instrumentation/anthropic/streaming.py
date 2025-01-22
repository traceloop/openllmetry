import logging
import time

from opentelemetry.instrumentation.anthropic.config import Config
from opentelemetry.instrumentation.anthropic.events import create_completion_event
from opentelemetry.instrumentation.anthropic.utils import (
    acount_prompt_tokens_from_request,
    dont_throw,
    error_metrics_attributes,
    count_prompt_tokens_from_request,
    set_span_attribute,
    shared_metrics_attributes,
    should_send_prompts,
)
from opentelemetry.metrics import Counter, Histogram
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace.status import Status, StatusCode

logger = logging.getLogger(__name__)


@dont_throw
def _process_response_item(item, complete_response):
    if item.type == "message_start":
        complete_response["model"] = item.message.model
        complete_response["usage"] = dict(item.message.usage)
    elif item.type == "content_block_start":
        index = item.index
        if len(complete_response.get("events")) <= index:
            complete_response["events"].append({"index": index, "text": ""})
    elif item.type == "content_block_delta" and item.delta.type == "text_delta":
        index = item.index
        complete_response.get("events")[index]["text"] += item.delta.text
    elif item.type == "message_delta":
        for event in complete_response.get("events", []):
            event["finish_reason"] = item.delta.stop_reason
        if item.usage:
            if "usage" in complete_response:
                item_output_tokens = dict(item.usage).get("output_tokens", 0)
                existing_output_tokens = complete_response["usage"].get("output_tokens", 0)
                complete_response["usage"]["output_tokens"] = item_output_tokens + existing_output_tokens
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
    cache_read_tokens = complete_response.get("usage", {}).get("cache_read_input_tokens", 0)
    cache_creation_tokens = complete_response.get("usage", {}).get("cache_creation_input_tokens", 0)

    input_tokens = prompt_tokens + cache_read_tokens + cache_creation_tokens
    total_tokens = input_tokens + completion_tokens

    set_span_attribute(span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, input_tokens)
    set_span_attribute(
        span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completion_tokens
    )
    set_span_attribute(span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total_tokens)

    set_span_attribute(
        span, SpanAttributes.LLM_RESPONSE_MODEL, complete_response.get("model")
    )
    set_span_attribute(
        span, SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS, cache_read_tokens
    )
    set_span_attribute(
        span, SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS, cache_creation_tokens
    )

    if token_histogram and type(input_tokens) is int and input_tokens >= 0:
        token_histogram.record(
            input_tokens,
            attributes={
                **metric_attributes,
                SpanAttributes.LLM_TOKEN_TYPE: "input",
            },
        )

    if token_histogram and type(completion_tokens) is int and completion_tokens >= 0:
        token_histogram.record(
            completion_tokens,
            attributes={
                **metric_attributes,
                SpanAttributes.LLM_TOKEN_TYPE: "output",
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


def _set_completions(span, events, event_logger=None):
    if not events:
        return

    try:
        for event in events:
            index = event.get("index")
            
            # Legacy attribute-based approach
            if Config.use_legacy_attributes and span.is_recording():
                prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
                set_span_attribute(
                    span, f"{prefix}.finish_reason", event.get("finish_reason")
                )
                set_span_attribute(span, f"{prefix}.content", event.get("text"))
            
            # Event-based approach
            if event_logger and not Config.use_legacy_attributes:
                event_logger.emit(
                    create_completion_event(
                        content=event.get("text", ""),
                        role="assistant",
                        finish_reason=event.get("finish_reason"),
                        span_ctx=span.get_span_context(),
                        index=index,
                    )
                )
    except Exception as e:
        logger.warning("Failed to set completion attributes, error: %s", str(e))


@dont_throw
def build_from_streaming_response(
    response,
    span,
    event_logger=None,
    token_histogram: Histogram = None,
    choice_counter: Counter = None,
):
    complete_response = {"events": [], "model": "", "usage": {}}
    for item in response:
        try:
            yield item
        except Exception as e:
            attributes = error_metrics_attributes(e)
            raise e
        _process_response_item(item, complete_response)

    metric_attributes = shared_metrics_attributes(complete_response)

    # calculate token usage
    if Config.enrich_token_usage:
        try:
            completion_tokens = -1
            # prompt_usage
            if usage := complete_response.get("usage"):
                prompt_tokens = usage.get("input_tokens", 0)
            else:
                prompt_tokens = count_prompt_tokens_from_request(kwargs)

            # completion_usage
            if usage := complete_response.get("usage"):
                completion_tokens = usage.get("output_tokens", 0)
            else:
                completion_content = ""
                if complete_response.get("events"):
                    model_name = complete_response.get("model") or None
                    for event in complete_response.get("events"):  # type: dict
                        if event.get("text"):
                            completion_content += event.get("text")

            _set_token_usage(
                span,
                complete_response,
                prompt_tokens,
                completion_tokens,
                metric_attributes,
                token_histogram,
                choice_counter,
            )
        except Exception as e:
            logger.warning("Failed to set token usage, error: %s", e)

    if should_send_prompts():
        _set_completions(span, complete_response.get("events"), event_logger)

    span.set_status(Status(StatusCode.OK))
    span.end()


@dont_throw
async def abuild_from_streaming_response(
    response,
    span,
    event_logger=None,
    token_histogram: Histogram = None,
    choice_counter: Counter = None,
):
    complete_response = {"events": [], "model": "", "usage": {}}
    async for item in response:
        try:
            yield item
        except Exception as e:
            attributes = error_metrics_attributes(e)
            raise e
        _process_response_item(item, complete_response)

    metric_attributes = shared_metrics_attributes(complete_response)

    # calculate token usage
    if Config.enrich_token_usage:
        try:
            # prompt_usage
            if usage := complete_response.get("usage"):
                prompt_tokens = usage.get("input_tokens", 0)
            else:
                prompt_tokens = await acount_prompt_tokens_from_request(kwargs)

            # completion_usage
            if usage := complete_response.get("usage"):
                completion_tokens = usage.get("output_tokens", 0)
            else:
                completion_content = ""
                if complete_response.get("events"):
                    model_name = complete_response.get("model") or None
                    for event in complete_response.get("events"):  # type: dict
                        if event.get("text"):
                            completion_content += event.get("text")

            _set_token_usage(
                span,
                complete_response,
                prompt_tokens,
                completion_tokens,
                metric_attributes,
                token_histogram,
                choice_counter,
            )
        except Exception as e:
            logger.warning("Failed to set token usage, error: %s", e)

    if should_send_prompts():
        _set_completions(span, complete_response.get("events"), event_logger)

    span.set_status(Status(StatusCode.OK))
    span.end()
