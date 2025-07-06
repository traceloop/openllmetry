import json
import logging
from typing import Any, Dict

from opentelemetry.instrumentation.anthropic.config import Config
from opentelemetry.instrumentation.anthropic.utils import (
    JSONEncoder,
    dont_throw,
    model_as_dict,
    should_send_prompts,
)
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_RESPONSE_ID,
)
from opentelemetry.semconv_ai import SpanAttributes

logger = logging.getLogger(__name__)


def _is_base64_image(item: Dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False

    if not isinstance(item.get("source"), dict):
        return False

    if item.get("type") != "image" or item["source"].get("type") != "base64":
        return False

    return True


async def _process_image_item(item, trace_id, span_id, message_index, content_index):
    if not Config.upload_base64_image:
        return item

    image_format = item.get("source").get("media_type").split("/")[1]
    image_name = f"message_{message_index}_content_{content_index}.{image_format}"
    base64_string = item.get("source").get("data")
    url = await Config.upload_base64_image(trace_id, span_id, image_name, base64_string)

    return {"type": "image_url", "image_url": {"url": url}}


async def _dump_content(message_index, content, span):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # If the content is a list of text blocks, concatenate them.
        # This is more commonly used in prompt caching.
        if all([model_as_dict(item).get("type") == "text" for item in content]):
            return "".join([model_as_dict(item).get("text") for item in content])

        content = [
            (
                await _process_image_item(
                    model_as_dict(item),
                    span.context.trace_id,
                    span.context.span_id,
                    message_index,
                    j,
                )
                if _is_base64_image(model_as_dict(item))
                else model_as_dict(item)
            )
            for j, item in enumerate(content)
        ]

        return json.dumps(content, cls=JSONEncoder)


@dont_throw
async def aset_input_attributes(span, kwargs):
    from opentelemetry.instrumentation.anthropic import set_span_attribute

    set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("model"))
    set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, kwargs.get("max_tokens_to_sample")
    )
    set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_TEMPERATURE, kwargs.get("temperature")
    )
    set_span_attribute(span, SpanAttributes.LLM_REQUEST_TOP_P, kwargs.get("top_p"))
    set_span_attribute(
        span, SpanAttributes.LLM_FREQUENCY_PENALTY, kwargs.get("frequency_penalty")
    )
    set_span_attribute(
        span, SpanAttributes.LLM_PRESENCE_PENALTY, kwargs.get("presence_penalty")
    )
    set_span_attribute(span, SpanAttributes.LLM_IS_STREAMING, kwargs.get("stream"))

    if should_send_prompts():
        if kwargs.get("prompt") is not None:
            set_span_attribute(
                span, f"{SpanAttributes.LLM_PROMPTS}.0.user", kwargs.get("prompt")
            )

        elif kwargs.get("messages") is not None:
            has_system_message = False
            if kwargs.get("system"):
                has_system_message = True
                set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.0.content",
                    await _dump_content(
                        message_index=0, span=span, content=kwargs.get("system")
                    ),
                )
                set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.0.role",
                    "system",
                )
            for i, message in enumerate(kwargs.get("messages")):
                prompt_index = i + (1 if has_system_message else 0)
                set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.{prompt_index}.content",
                    await _dump_content(
                        message_index=i, span=span, content=message.get("content")
                    ),
                )
                set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.{prompt_index}.role",
                    message.get("role"),
                )

        if kwargs.get("tools") is not None:
            for i, tool in enumerate(kwargs.get("tools")):
                prefix = f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{i}"
                set_span_attribute(span, f"{prefix}.name", tool.get("name"))
                set_span_attribute(
                    span, f"{prefix}.description", tool.get("description")
                )
                input_schema = tool.get("input_schema")
                if input_schema is not None:
                    set_span_attribute(
                        span, f"{prefix}.input_schema", json.dumps(input_schema)
                    )


def _set_span_completions(span, response):
    if not should_send_prompts():
        return
    from opentelemetry.instrumentation.anthropic import set_span_attribute

    index = 0
    prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
    set_span_attribute(span, f"{prefix}.finish_reason", response.get("stop_reason"))
    if response.get("role"):
        set_span_attribute(span, f"{prefix}.role", response.get("role"))

    if response.get("completion"):
        set_span_attribute(span, f"{prefix}.content", response.get("completion"))
    elif response.get("content"):
        tool_call_index = 0
        text = ""
        for content in response.get("content"):
            content_block_type = content.type
            # usually, Antrhopic responds with just one text block,
            # but the API allows for multiple text blocks, so concatenate them
            if content_block_type == "text":
                text += content.text
            elif content_block_type == "thinking":
                content = dict(content)
                # override the role to thinking
                set_span_attribute(
                    span,
                    f"{prefix}.role",
                    "thinking",
                )
                set_span_attribute(
                    span,
                    f"{prefix}.content",
                    content.get("thinking"),
                )
                # increment the index for subsequent content blocks
                index += 1
                prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
                # set the role to the original role on the next completions
                set_span_attribute(
                    span,
                    f"{prefix}.role",
                    response.get("role"),
                )
            elif content_block_type == "tool_use":
                content = dict(content)
                set_span_attribute(
                    span,
                    f"{prefix}.tool_calls.{tool_call_index}.id",
                    content.get("id"),
                )
                set_span_attribute(
                    span,
                    f"{prefix}.tool_calls.{tool_call_index}.name",
                    content.get("name"),
                )
                tool_arguments = content.get("input")
                if tool_arguments is not None:
                    set_span_attribute(
                        span,
                        f"{prefix}.tool_calls.{tool_call_index}.arguments",
                        json.dumps(tool_arguments),
                    )
                tool_call_index += 1
        set_span_attribute(span, f"{prefix}.content", text)


@dont_throw
def set_response_attributes(span, response):
    from opentelemetry.instrumentation.anthropic import set_span_attribute

    if not isinstance(response, dict):
        response = response.__dict__
    set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, response.get("model"))
    set_span_attribute(span, GEN_AI_RESPONSE_ID, response.get("id"))

    if response.get("usage"):
        prompt_tokens = response.get("usage").input_tokens
        completion_tokens = response.get("usage").output_tokens
        set_span_attribute(span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, prompt_tokens)
        set_span_attribute(
            span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completion_tokens
        )
        set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
            prompt_tokens + completion_tokens,
        )

    _set_span_completions(span, response)


@dont_throw
def set_streaming_response_attributes(span, complete_response_events):
    if not should_send_prompts():
        return

    from opentelemetry.instrumentation.anthropic import set_span_attribute

    if not span.is_recording() or not complete_response_events:
        return

    try:
        for event in complete_response_events:
            index = event.get("index")
            prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
            set_span_attribute(
                span, f"{prefix}.finish_reason", event.get("finish_reason")
            )
            role = "thinking" if event.get("type") == "thinking" else "assistant"
            set_span_attribute(span, f"{prefix}.role", role)
            set_span_attribute(span, f"{prefix}.content", event.get("text"))
    except Exception as e:
        logger.warning("Failed to set completion attributes, error: %s", str(e))
