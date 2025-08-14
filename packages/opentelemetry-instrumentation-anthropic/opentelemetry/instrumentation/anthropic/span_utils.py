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
                content = message.get("content")
                tool_use_blocks = []
                other_blocks = []
                if isinstance(content, list):
                    for block in content:
                        if dict(block).get("type") == "tool_use":
                            tool_use_blocks.append(dict(block))
                        else:
                            other_blocks.append(block)
                    content = other_blocks
                set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.{prompt_index}.content",
                    await _dump_content(message_index=i, span=span, content=content),
                )
                set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.{prompt_index}.role",
                    message.get("role"),
                )
                if tool_use_blocks:
                    for tool_num, tool_use_block in enumerate(tool_use_blocks):
                        set_span_attribute(
                            span,
                            f"{SpanAttributes.LLM_PROMPTS}.{prompt_index}.tool_calls.{tool_num}.id",
                            tool_use_block.get("id"),
                        )
                        set_span_attribute(
                            span,
                            f"{SpanAttributes.LLM_PROMPTS}.{prompt_index}.tool_calls.{tool_num}.name",
                            tool_use_block.get("name"),
                        )
                        set_span_attribute(
                            span,
                            f"{SpanAttributes.LLM_PROMPTS}.{prompt_index}.tool_calls.{tool_num}.arguments",
                            json.dumps(tool_use_block.get("input")),
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


async def _aset_span_completions(span, response):
    if not should_send_prompts():
        return
    from opentelemetry.instrumentation.anthropic import set_span_attribute
    import inspect

    # If we get a coroutine, await it
    if inspect.iscoroutine(response):
        try:
            response = await response
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.debug(f"Failed to await coroutine response: {e}")
            return

    # Work directly with the response object to preserve its structure
    index = 0
    prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"

    # Safely get attributes without extracting the whole object
    stop_reason = getattr(response, "stop_reason", None)
    role = getattr(response, "role", None)
    completion = getattr(response, "completion", None)
    content = getattr(response, "content", None)

    set_span_attribute(span, f"{prefix}.finish_reason", stop_reason)
    if role:
        set_span_attribute(span, f"{prefix}.role", role)

    if completion:
        set_span_attribute(span, f"{prefix}.content", completion)
    elif content:
        tool_call_index = 0
        text = ""
        for content_item in content:
            content_block_type = getattr(content_item, "type", None)
            # usually, Antrhopic responds with just one text block,
            # but the API allows for multiple text blocks, so concatenate them
            if content_block_type == "text" and hasattr(content_item, "text"):
                text += content_item.text
            elif content_block_type == "thinking":
                content_dict = (
                    dict(content_item)
                    if hasattr(content_item, "__dict__")
                    else content_item
                )
                # override the role to thinking
                set_span_attribute(
                    span,
                    f"{prefix}.role",
                    "thinking",
                )
                thinking_content = (
                    content_dict.get("thinking")
                    if isinstance(content_dict, dict)
                    else getattr(content_item, "thinking", None)
                )
                set_span_attribute(
                    span,
                    f"{prefix}.content",
                    thinking_content,
                )
                # increment the index for subsequent content blocks
                index += 1
                prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
                # set the role to the original role on the next completions
                set_span_attribute(
                    span,
                    f"{prefix}.role",
                    role,
                )
            elif content_block_type == "tool_use":
                content_dict = (
                    dict(content_item)
                    if hasattr(content_item, "__dict__")
                    else content_item
                )
                tool_id = (
                    content_dict.get("id")
                    if isinstance(content_dict, dict)
                    else getattr(content_item, "id", None)
                )
                tool_name = (
                    content_dict.get("name")
                    if isinstance(content_dict, dict)
                    else getattr(content_item, "name", None)
                )
                tool_input = (
                    content_dict.get("input")
                    if isinstance(content_dict, dict)
                    else getattr(content_item, "input", None)
                )

                set_span_attribute(
                    span,
                    f"{prefix}.tool_calls.{tool_call_index}.id",
                    tool_id,
                )
                set_span_attribute(
                    span,
                    f"{prefix}.tool_calls.{tool_call_index}.name",
                    tool_name,
                )
                if tool_input is not None:
                    set_span_attribute(
                        span,
                        f"{prefix}.tool_calls.{tool_call_index}.arguments",
                        json.dumps(tool_input),
                    )
                tool_call_index += 1
        set_span_attribute(span, f"{prefix}.content", text)


def _set_span_completions(span, response):
    if not should_send_prompts():
        return
    from opentelemetry.instrumentation.anthropic import set_span_attribute
    import inspect

    # If we get a coroutine, we cannot process it in sync context
    if inspect.iscoroutine(response):
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(
            f"_set_span_completions received coroutine {response} - span processing skipped"
        )
        return

    # Work directly with the response object to preserve its structure
    index = 0
    prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"

    # Safely get attributes without extracting the whole object
    stop_reason = getattr(response, "stop_reason", None)
    role = getattr(response, "role", None)
    completion = getattr(response, "completion", None)
    content = getattr(response, "content", None)

    set_span_attribute(span, f"{prefix}.finish_reason", stop_reason)
    if role:
        set_span_attribute(span, f"{prefix}.role", role)

    if completion:
        set_span_attribute(span, f"{prefix}.content", completion)
    elif content:
        tool_call_index = 0
        text = ""
        for content_item in content:
            content_block_type = getattr(content_item, "type", None)
            # usually, Antrhopic responds with just one text block,
            # but the API allows for multiple text blocks, so concatenate them
            if content_block_type == "text" and hasattr(content_item, "text"):
                text += content_item.text
            elif content_block_type == "thinking":
                content_dict = (
                    dict(content_item)
                    if hasattr(content_item, "__dict__")
                    else content_item
                )
                # override the role to thinking
                set_span_attribute(
                    span,
                    f"{prefix}.role",
                    "thinking",
                )
                thinking_content = (
                    content_dict.get("thinking")
                    if isinstance(content_dict, dict)
                    else getattr(content_item, "thinking", None)
                )
                set_span_attribute(
                    span,
                    f"{prefix}.content",
                    thinking_content,
                )
                # increment the index for subsequent content blocks
                index += 1
                prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
                # set the role to the original role on the next completions
                set_span_attribute(
                    span,
                    f"{prefix}.role",
                    role,
                )
            elif content_block_type == "tool_use":
                content_dict = (
                    dict(content_item)
                    if hasattr(content_item, "__dict__")
                    else content_item
                )
                tool_id = (
                    content_dict.get("id")
                    if isinstance(content_dict, dict)
                    else getattr(content_item, "id", None)
                )
                tool_name = (
                    content_dict.get("name")
                    if isinstance(content_dict, dict)
                    else getattr(content_item, "name", None)
                )
                tool_input = (
                    content_dict.get("input")
                    if isinstance(content_dict, dict)
                    else getattr(content_item, "input", None)
                )

                set_span_attribute(
                    span,
                    f"{prefix}.tool_calls.{tool_call_index}.id",
                    tool_id,
                )
                set_span_attribute(
                    span,
                    f"{prefix}.tool_calls.{tool_call_index}.name",
                    tool_name,
                )
                if tool_input is not None:
                    set_span_attribute(
                        span,
                        f"{prefix}.tool_calls.{tool_call_index}.arguments",
                        json.dumps(tool_input),
                    )
                tool_call_index += 1
        set_span_attribute(span, f"{prefix}.content", text)


@dont_throw
async def aset_response_attributes(span, response):
    from opentelemetry.instrumentation.anthropic import set_span_attribute
    import inspect

    # If we get a coroutine, await it
    if inspect.iscoroutine(response):
        try:
            response = await response
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.debug(f"Failed to await coroutine response: {e}")
            return

    # Work directly with the response object
    model = getattr(response, "model", None)
    response_id = getattr(response, "id", None)
    usage = getattr(response, "usage", None)

    set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, model)
    set_span_attribute(span, GEN_AI_RESPONSE_ID, response_id)

    if usage:
        prompt_tokens = getattr(usage, "input_tokens", None)
        completion_tokens = getattr(usage, "output_tokens", None)
        if prompt_tokens is not None:
            set_span_attribute(
                span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, prompt_tokens
            )
        if completion_tokens is not None:
            set_span_attribute(
                span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completion_tokens
            )
        if prompt_tokens is not None and completion_tokens is not None:
            set_span_attribute(
                span,
                SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
                prompt_tokens + completion_tokens,
            )

    await _aset_span_completions(span, response)


@dont_throw
def set_response_attributes(span, response):
    from opentelemetry.instrumentation.anthropic import set_span_attribute
    import inspect

    # If we get a coroutine, we cannot process it in sync context
    if inspect.iscoroutine(response):
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(
            f"set_response_attributes received coroutine {response} - response processing skipped"
        )
        return

    # Work directly with the response object
    model = getattr(response, "model", None)
    response_id = getattr(response, "id", None)
    usage = getattr(response, "usage", None)

    set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, model)
    set_span_attribute(span, GEN_AI_RESPONSE_ID, response_id)

    if usage:
        prompt_tokens = getattr(usage, "input_tokens", None)
        completion_tokens = getattr(usage, "output_tokens", None)
        if prompt_tokens is not None:
            set_span_attribute(
                span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, prompt_tokens
            )
        if completion_tokens is not None:
            set_span_attribute(
                span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completion_tokens
            )
        if prompt_tokens is not None and completion_tokens is not None:
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

    index = 0
    for event in complete_response_events:
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
        set_span_attribute(span, f"{prefix}.finish_reason", event.get("finish_reason"))
        role = "thinking" if event.get("type") == "thinking" else "assistant"
        # Thinking is added as a separate completion, so we need to increment the index
        if event.get("type") == "thinking":
            index += 1
        set_span_attribute(span, f"{prefix}.role", role)
        if event.get("type") == "tool_use":
            set_span_attribute(
                span,
                f"{prefix}.tool_calls.0.id",
                event.get("id"),
            )
            set_span_attribute(
                span,
                f"{prefix}.tool_calls.0.name",
                event.get("name"),
            )
            tool_arguments = event.get("input")
            if tool_arguments is not None:
                set_span_attribute(
                    span,
                    f"{prefix}.tool_calls.0.arguments",
                    tool_arguments,
                )
        else:
            set_span_attribute(span, f"{prefix}.content", event.get("text"))
