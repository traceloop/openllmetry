import logging
from opentelemetry.instrumentation.anthropic.utils import set_span_attribute, should_send_prompts
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.semconv.ai import SpanAttributes

from opentelemetry.instrumentation.anthropic.config import Config

logger = logging.getLogger(__name__)


def _process_response_item(item, complete_response):
    try:
        if item.type == 'message_start':
            complete_response["model"] = item.message.model
            complete_response["usage"] = item.message.usage
        elif item.type == 'content_block_start':
            index = item.index
            if len(complete_response.get("events")) <= index:
                complete_response["events"].append({"index": index, "text": ""})
        elif item.type == 'content_block_delta' and item.delta.type == 'text_delta':
            index = item.index
            complete_response.get("events")[index]["text"] += item.delta.text
        elif item.type == 'message_delta':
            for event in complete_response.get("events", []):
                event["finish_reason"] = item.delta.stop_reason
    except Exception as ex:
        logger.warning("Failed to process response item, error: %s", str(ex))


def _set_token_usage(span, complete_response, prompt_tokens, completion_tokens):
    total_tokens = prompt_tokens + completion_tokens
    set_span_attribute(span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, prompt_tokens)
    set_span_attribute(
        span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completion_tokens
    )
    set_span_attribute(span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total_tokens)

    set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, complete_response.get("model"))


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


def _build_from_streaming_response(
        span,
        response,
        instance,
        kwargs
):
    complete_response = {"events": [], "model": "", "usage": {}}
    for item in response:
        yield item
        _process_response_item(item, complete_response)

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
                    [instance.count_tokens(m.get("content")) for m in kwargs.get("messages")]
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

            _set_token_usage(span, complete_response, prompt_tokens, completion_tokens)
        except Exception as e:
            logger.warning("Failed to set token usage, error: %s", str(e))

    if should_send_prompts():
        _set_completions(span, complete_response.get("events"))

    span.set_status(Status(StatusCode.OK))
    span.end()


async def _abuild_from_streaming_response(
        span,
        response,
        instance,
        kwargs
):
    complete_response = {"events": [], "model": ""}
    async for item in response:
        yield item
        _process_response_item(item, complete_response)

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
                    [await instance.count_tokens(m.get("content")) for m in kwargs.get("messages")]
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

            _set_token_usage(span, complete_response, prompt_tokens, completion_tokens)
        except Exception as e:
            logger.warning("Failed to set token usage, error: %s", str(e))

    if should_send_prompts():
        _set_completions(span, complete_response.get("events"))

    span.set_status(Status(StatusCode.OK))
    span.end()
