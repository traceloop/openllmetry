import json
import logging
from typing import Any, Dict

from opentelemetry.instrumentation.anthropic.config import Config
from opentelemetry.instrumentation.anthropic.utils import (
    JSONEncoder,
    dont_throw,
    model_as_dict,
    should_send_prompts,
    _extract_response_data,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
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

    set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_MODEL, kwargs.get("model"))
    set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS, kwargs.get("max_tokens_to_sample") or kwargs.get("max_tokens")
    )
    set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE, kwargs.get("temperature")
    )
    set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_TOP_P, kwargs.get("top_p"))
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
                span,
                GenAIAttributes.GEN_AI_INPUT_MESSAGES,
                json.dumps([{"role": "user", "content": kwargs.get("prompt")}]),
            )

        elif kwargs.get("messages") is not None:
            if kwargs.get("system"):
                set_span_attribute(
                    span,
                    GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS,
                    await _dump_content(
                        message_index=0, span=span, content=kwargs.get("system")
                    ),
                )

            input_messages = []
            for i, message in enumerate(kwargs.get("messages")):
                content = message.get("content")
                tool_use_blocks = []
                non_tool_use_content = content
                if isinstance(content, list):
                    tool_use_blocks = [
                        dict(block)
                        for block in content
                        if dict(block).get("type") == "tool_use"
                    ]
                    non_tool_use_content = [
                        block
                        for block in content
                        if dict(block).get("type") != "tool_use"
                    ] or None

                msg_obj = {
                    "role": message.get("role"),
                    "content": await _dump_content(
                        message_index=i, span=span, content=non_tool_use_content
                    ),
                }
                if tool_use_blocks:
                    msg_obj["tool_calls"] = [
                        {
                            "id": block.get("id"),
                            "name": block.get("name"),
                            "arguments": json.dumps(block.get("input")),
                        }
                        for block in tool_use_blocks
                    ]
                input_messages.append(msg_obj)

            set_span_attribute(
                span,
                GenAIAttributes.GEN_AI_INPUT_MESSAGES,
                json.dumps(input_messages, cls=JSONEncoder),
            )

        if kwargs.get("tools") is not None:
            tool_defs = []
            for tool in kwargs.get("tools"):
                tool_def = {"name": tool.get("name")}
                if tool.get("description"):
                    tool_def["description"] = tool.get("description")
                if tool.get("input_schema") is not None:
                    tool_def["input_schema"] = tool.get("input_schema")
                tool_defs.append(tool_def)
            set_span_attribute(
                span,
                GenAIAttributes.GEN_AI_TOOL_DEFINITIONS,
                json.dumps(tool_defs, cls=JSONEncoder),
            )

        output_format = kwargs.get("output_format")
        if output_format and isinstance(output_format, dict):
            if output_format.get("type") == "json_schema":
                schema = output_format.get("schema")
                if schema:
                    set_span_attribute(
                        span,
                        "gen_ai.request.structured_output_schema",
                        json.dumps(schema),
                    )


async def _aset_span_completions(span, response):
    from opentelemetry.instrumentation.anthropic import set_span_attribute
    from opentelemetry.instrumentation.anthropic.utils import _aextract_response_data

    response = await _aextract_response_data(response)
    stop_reason = response.get("stop_reason")

    if stop_reason:
        span.set_attribute(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS, [stop_reason])

    if not should_send_prompts():
        return

    output_messages = []

    if response.get("completion"):
        output_messages.append({
            "role": response.get("role", "assistant"),
            "content": response.get("completion"),
        })
    elif response.get("content"):
        tool_calls = []
        text = ""
        thinking_messages = []
        for content in response.get("content"):
            content_block_type = content.type
            if content_block_type == "text" and hasattr(content, "text"):
                text += content.text
            elif content_block_type == "thinking":
                thinking_messages.append({
                    "role": "thinking",
                    "content": getattr(content, "thinking", None),
                })
            elif content_block_type == "tool_use":
                tool_arguments = getattr(content, "input", None)
                tool_calls.append({
                    "id": getattr(content, "id", None),
                    "name": getattr(content, "name", None),
                    "arguments": json.dumps(tool_arguments) if tool_arguments is not None else None,
                })

        output_messages.extend(thinking_messages)
        msg = {"role": response.get("role", "assistant"), "content": text}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        output_messages.append(msg)

    if output_messages:
        set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
            json.dumps(output_messages, cls=JSONEncoder),
        )


def _set_span_completions(span, response):
    from opentelemetry.instrumentation.anthropic import set_span_attribute

    response = _extract_response_data(response)
    stop_reason = response.get("stop_reason")

    if stop_reason:
        span.set_attribute(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS, [stop_reason])

    if not should_send_prompts():
        return

    output_messages = []

    if response.get("completion"):
        output_messages.append({
            "role": response.get("role", "assistant"),
            "content": response.get("completion"),
        })
    elif response.get("content"):
        tool_calls = []
        text = ""
        thinking_messages = []
        for content in response.get("content"):
            content_block_type = content.type
            # usually, Anthropic responds with just one text block,
            # but the API allows for multiple text blocks, so concatenate them
            if content_block_type == "text" and hasattr(content, "text"):
                text += content.text
            elif content_block_type == "thinking":
                thinking_messages.append({
                    "role": "thinking",
                    "content": getattr(content, "thinking", None),
                })
            elif content_block_type == "tool_use":
                tool_arguments = getattr(content, "input", None)
                tool_calls.append({
                    "id": getattr(content, "id", None),
                    "name": getattr(content, "name", None),
                    "arguments": json.dumps(tool_arguments) if tool_arguments is not None else None,
                })

        output_messages.extend(thinking_messages)
        msg = {"role": response.get("role", "assistant"), "content": text}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        output_messages.append(msg)

    if output_messages:
        set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
            json.dumps(output_messages, cls=JSONEncoder),
        )


@dont_throw
async def aset_response_attributes(span, response):
    from opentelemetry.instrumentation.anthropic import set_span_attribute
    from opentelemetry.instrumentation.anthropic.utils import _aextract_response_data

    response = await _aextract_response_data(response)
    set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, response.get("model"))
    set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_ID, response.get("id"))

    if response.get("usage"):
        prompt_tokens = response.get("usage").input_tokens
        completion_tokens = response.get("usage").output_tokens
        set_span_attribute(span, GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens)
        set_span_attribute(
            span, GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens
        )
        set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
            prompt_tokens + completion_tokens,
        )

    await _aset_span_completions(span, response)


@dont_throw
def set_response_attributes(span, response):
    from opentelemetry.instrumentation.anthropic import set_span_attribute

    response = _extract_response_data(response)
    set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, response.get("model"))
    set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_ID, response.get("id"))

    if response.get("usage"):
        prompt_tokens = response.get("usage").input_tokens
        completion_tokens = response.get("usage").output_tokens
        set_span_attribute(span, GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens)
        set_span_attribute(
            span, GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens
        )
        set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
            prompt_tokens + completion_tokens,
        )

    _set_span_completions(span, response)


@dont_throw
def set_streaming_response_attributes(span, complete_response_events):
    from opentelemetry.instrumentation.anthropic import set_span_attribute

    if not span.is_recording() or not complete_response_events:
        return

    output_messages = []
    finish_reasons = []

    for event in complete_response_events:
        finish_reason = event.get("finish_reason")
        if finish_reason and finish_reason not in finish_reasons:
            finish_reasons.append(finish_reason)

        if should_send_prompts():
            if event.get("type") == "thinking":
                output_messages.append({
                    "role": "thinking",
                    "content": event.get("text"),
                })
            elif event.get("type") == "tool_use":
                tool_arguments = event.get("input")
                output_messages.append({
                    "role": "assistant",
                    "tool_calls": [{
                        "id": event.get("id"),
                        "name": event.get("name"),
                        "arguments": json.dumps(tool_arguments) if tool_arguments is not None else None,
                    }],
                })
            else:
                output_messages.append({
                    "role": "assistant",
                    "content": event.get("text"),
                })

    if finish_reasons:
        span.set_attribute(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons)

    if output_messages:
        set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
            json.dumps(output_messages, cls=JSONEncoder),
        )
