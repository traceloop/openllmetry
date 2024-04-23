import logging
import time

from opentelemetry.instrumentation.anthropic.config import Config
from opentelemetry.instrumentation.anthropic.utils import (
    dont_throw,
    set_span_attribute,
    should_send_prompts,
)
from opentelemetry.metrics import Counter, Histogram
from opentelemetry.semconv.ai import SpanAttributes
from opentelemetry.trace.status import Status, StatusCode

logger = logging.getLogger(__name__)


@dont_throw
def _process_response_item(item, complete_response):
    if item.type == "message_start":
        complete_response["model"] = item.message.model
        complete_response["usage"] = item.message.usage
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


def _set_token_usage(
    span,
    complete_response,
    prompt_tokens,
    completion_tokens,
    metric_attributes: dict = {},
    token_counter: Counter = None,
    choice_counter: Counter = None,
):
    total_tokens = prompt_tokens + completion_tokens
    set_span_attribute(span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, prompt_tokens)
    set_span_attribute(
        span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completion_tokens
    )
    set_span_attribute(span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total_tokens)

    set_span_attribute(
        span, SpanAttributes.LLM_RESPONSE_MODEL, complete_response.get("model")
    )

    if token_counter and type(prompt_tokens) is int and prompt_tokens >= 0:
        token_counter.add(
            prompt_tokens,
            attributes={
                **metric_attributes,
                "llm.usage.token_type": "prompt",
            },
        )

    if token_counter and type(completion_tokens) is int and completion_tokens >= 0:
        token_counter.add(
            completion_tokens,
            attributes={
                **metric_attributes,
                "llm.usage.token_type": "completion",
            },
        )

    if type(complete_response.get("events")) is list and choice_counter:
        for event in complete_response.get("events"):
            choice_counter.add(
                1,
                attributes={
                    **metric_attributes,
                    "llm.response.finish_reason": event.get("finish_reason"),
                },
            )


def _set_completions(span, events):
    if not span.is_recording() or not events:
        return

    try:
        for event in events:
            index = event.get("index")
            prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
            set_span_attribute(
                span, f"{prefix}.finish_reason", event.get("finish_reason")
            )
            set_span_attribute(span, f"{prefix}.content", event.get("text"))
    except Exception as e:
        logger.warning("Failed to set completion attributes, error: %s", str(e))


@dont_throw
def build_from_streaming_response(
    span,
    response,
    instance,
    start_time,
    token_counter: Counter = None,
    choice_counter: Counter = None,
    duration_histogram: Histogram = None,
    exception_counter: Counter = None,
    kwargs: dict = {},
):
    complete_response = {"events": [], "model": "", "usage": {}}
    for item in response:
        try:
            yield item
        except Exception as e:
            attributes = {
                "error.type": e.__class__.__name__,
            }
            if exception_counter:
                exception_counter.add(1, attributes=attributes)
            raise e
        _process_response_item(item, complete_response)

    metric_attributes = {
        "gen_ai.response.model": complete_response.get("model"),
    }

    if duration_histogram:
        duration = time.time() - start_time
        duration_histogram.record(
            duration,
            attributes=metric_attributes,
        )

    # calculate token usage
    if Config.enrich_token_usage:
        try:
            prompt_tokens = -1
            completion_tokens = -1

            # prompt_usage
            if kwargs.get("prompt"):
                prompt_tokens = instance.count_tokens(kwargs.get("prompt"))
            elif kwargs.get("messages"):
                prompt_tokens = sum(
                    [
                        instance.count_tokens(m.get("content"))
                        for m in kwargs.get("messages")
                    ]
                )

            # completion_usage
            completion_content = ""
            if complete_response.get("events"):
                model_name = complete_response.get("model") or None
                for event in complete_response.get("events"):  # type: dict
                    if event.get("text"):
                        completion_content += event.get("text")

                if model_name:
                    completion_tokens = instance.count_tokens(completion_content)

            _set_token_usage(
                span,
                complete_response,
                prompt_tokens,
                completion_tokens,
                metric_attributes,
                token_counter,
                choice_counter,
            )
        except Exception as e:
            logger.warning("Failed to set token usage, error: %s", str(e))

    if should_send_prompts():
        _set_completions(span, complete_response.get("events"))

    span.set_status(Status(StatusCode.OK))
    span.end()


@dont_throw
async def abuild_from_streaming_response(
    span,
    response,
    instance,
    start_time,
    token_counter: Counter = None,
    choice_counter: Counter = None,
    duration_histogram: Histogram = None,
    exception_counter: Counter = None,
    kwargs: dict = {},
):
    complete_response = {"events": [], "model": ""}
    async for item in response:
        try:
            yield item
        except Exception as e:
            attributes = {
                "error.type": e.__class__.__name__,
            }
            if exception_counter:
                exception_counter.add(1, attributes=attributes)
            raise e
        _process_response_item(item, complete_response)

    metric_attributes = {
        "gen_ai.response.model": complete_response.get("model"),
    }

    if duration_histogram:
        duration = time.time() - start_time
        duration_histogram.record(
            duration,
            attributes=metric_attributes,
        )

    # calculate token usage
    if Config.enrich_token_usage:
        try:
            prompt_tokens = -1
            completion_tokens = -1

            # prompt_usage
            if kwargs.get("prompt"):
                prompt_tokens = await instance.count_tokens(kwargs.get("prompt"))
            elif kwargs.get("messages"):
                prompt_tokens = sum(
                    [
                        await instance.count_tokens(m.get("content"))
                        for m in kwargs.get("messages")
                    ]
                )

            # completion_usage
            completion_content = ""
            if complete_response.get("events"):
                model_name = complete_response.get("model") or None
                for event in complete_response.get("events"):  # type: dict
                    if event.get("text"):
                        completion_content += event.get("text")

                if model_name:
                    completion_tokens = await instance.count_tokens(completion_content)

            _set_token_usage(
                span,
                complete_response,
                prompt_tokens,
                completion_tokens,
                metric_attributes,
                token_counter,
                choice_counter,
            )
        except Exception as e:
            logger.warning("Failed to set token usage, error: %s", str(e))

    if should_send_prompts():
        _set_completions(span, complete_response.get("events"))

    span.set_status(Status(StatusCode.OK))
    span.end()
