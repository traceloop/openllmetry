import copy
import json
import base64
import logging
import asyncio
import threading
from opentelemetry.instrumentation.vertexai.utils import dont_throw, should_send_prompts
from opentelemetry.instrumentation.vertexai.config import Config
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes


logger = logging.getLogger(__name__)


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


def _is_base64_image_part(item):
    """Check if item is a VertexAI Part object containing image data"""
    try:
        # Check if it has the Part attributes we expect
        if not hasattr(item, 'inline_data') or not hasattr(item, 'mime_type'):
            return False

        # Check if it's an image mime type and has inline data
        if item.mime_type and 'image/' in item.mime_type and item.inline_data:
            # Check if the inline_data has actual data
            if hasattr(item.inline_data, 'data') and item.inline_data.data:
                return True

        return False
    except Exception:
        return False


async def _process_image_part(item, trace_id, span_id, content_index):
    """Process a VertexAI Part object containing image data"""
    if not Config.upload_base64_image:
        return None

    try:
        # Extract format from mime type (e.g., 'image/jpeg' -> 'jpeg')
        image_format = item.mime_type.split('/')[1] if item.mime_type else 'unknown'
        image_name = f"content_{content_index}.{image_format}"

        # Convert binary data to base64 string for upload
        binary_data = item.inline_data.data
        base64_string = base64.b64encode(binary_data).decode('utf-8')

        # Upload the base64 data - convert IDs to strings
        url = await Config.upload_base64_image(str(trace_id), str(span_id), image_name, base64_string)

        # Return OpenAI-compatible format for consistency across LLM providers
        return {
            "type": "image_url",
            "image_url": {"url": url}
        }
    except Exception as e:
        logger.warning(f"Failed to process image part: {e}")
        # Return None to skip adding this image to the span
        return None


def run_async(method):
    """Handle async method in sync context, following OpenAI's battle-tested approach"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        thread = threading.Thread(target=lambda: asyncio.run(method))
        thread.start()
        thread.join()
    else:
        asyncio.run(method)


def _process_image_part_sync(item, trace_id, span_id, content_index):
    """Synchronous version of image part processing using OpenAI's pattern"""
    if not Config.upload_base64_image:
        return None

    try:
        # Extract format from mime type (e.g., 'image/jpeg' -> 'jpeg')
        image_format = item.mime_type.split('/')[1] if item.mime_type else 'unknown'
        image_name = f"content_{content_index}.{image_format}"

        # Convert binary data to base64 string for upload
        binary_data = item.inline_data.data
        base64_string = base64.b64encode(binary_data).decode('utf-8')

        # Use OpenAI's run_async pattern to handle the async upload function
        url = None

        async def upload_task():
            nonlocal url
            url = await Config.upload_base64_image(str(trace_id), str(span_id), image_name, base64_string)

        run_async(upload_task())

        return {
            "type": "image_url",
            "image_url": {"url": url}
        }
    except Exception as e:
        logger.warning(f"Failed to process image part sync: {e}")
        # Return None to skip adding this image to the span
        return None


async def _process_vertexai_argument(argument, span):
    """Process a single argument for VertexAI, handling different types"""
    if isinstance(argument, str):
        # Simple text argument in OpenAI format
        return [{"type": "text", "text": argument}]

    elif isinstance(argument, list):
        # List of mixed content (text strings and Part objects) - deep copy and process
        content_list = copy.deepcopy(argument)
        processed_items = []

        for item_index, content_item in enumerate(content_list):
            processed_item = await _process_content_item_vertexai(content_item, span, item_index)
            if processed_item is not None:
                processed_items.append(processed_item)

        return processed_items

    else:
        # Single Part object - convert to OpenAI format
        processed_item = await _process_content_item_vertexai(argument, span, 0)
        return [processed_item] if processed_item is not None else []


async def _process_content_item_vertexai(content_item, span, item_index):
    """Process a single content item for VertexAI"""
    if isinstance(content_item, str):
        # Convert text to OpenAI format
        return {"type": "text", "text": content_item}

    elif _is_base64_image_part(content_item):
        # Process image part
        return await _process_image_part(
            content_item, span.context.trace_id, span.context.span_id, item_index
        )

    elif hasattr(content_item, 'text'):
        # Text part to OpenAI format
        return {"type": "text", "text": content_item.text}

    else:
        # Other types as text
        return {"type": "text", "text": str(content_item)}


def _process_vertexai_argument_sync(argument, span):
    """Synchronous version of argument processing for VertexAI"""
    if isinstance(argument, str):
        # Simple text argument in OpenAI format
        return [{"type": "text", "text": argument}]

    elif isinstance(argument, list):
        # List of mixed content (text strings and Part objects) - deep copy and process
        content_list = copy.deepcopy(argument)
        processed_items = []

        for item_index, content_item in enumerate(content_list):
            processed_item = _process_content_item_vertexai_sync(content_item, span, item_index)
            if processed_item is not None:
                processed_items.append(processed_item)

        return processed_items

    else:
        # Single Part object - convert to OpenAI format
        processed_item = _process_content_item_vertexai_sync(argument, span, 0)
        return [processed_item] if processed_item is not None else []


def _process_content_item_vertexai_sync(content_item, span, item_index):
    """Synchronous version of content item processing for VertexAI"""
    if isinstance(content_item, str):
        # Convert text to OpenAI format
        return {"type": "text", "text": content_item}

    elif _is_base64_image_part(content_item):
        # Process image part
        return _process_image_part_sync(
            content_item, span.context.trace_id, span.context.span_id, item_index
        )

    elif hasattr(content_item, 'text'):
        # Text part to OpenAI format
        return {"type": "text", "text": content_item.text}

    else:
        # Other types as text
        return {"type": "text", "text": str(content_item)}


@dont_throw
async def set_input_attributes(span, args):
    """Process input arguments, handling both text and image content"""
    if not span.is_recording():
        return
    if should_send_prompts() and args is not None and len(args) > 0:
        # Process each argument using extracted helper methods
        for arg_index, argument in enumerate(args):
            processed_content = await _process_vertexai_argument(argument, span)

            if processed_content:
                _set_span_attribute(
                    span,
                    f"{GenAIAttributes.GEN_AI_PROMPT}.{arg_index}.role",
                    "user"
                )
                _set_span_attribute(
                    span,
                    f"{GenAIAttributes.GEN_AI_PROMPT}.{arg_index}.content",
                    json.dumps(processed_content)
                )


# Sync version with image processing support
@dont_throw
def set_input_attributes_sync(span, args):
    """Synchronous version with image processing support"""
    if not span.is_recording():
        return
    if should_send_prompts() and args is not None and len(args) > 0:
        # Process each argument using extracted helper methods
        for arg_index, argument in enumerate(args):
            processed_content = _process_vertexai_argument_sync(argument, span)

            if processed_content:
                _set_span_attribute(
                    span,
                    f"{GenAIAttributes.GEN_AI_PROMPT}.{arg_index}.role",
                    "user"
                )
                _set_span_attribute(
                    span,
                    f"{GenAIAttributes.GEN_AI_PROMPT}.{arg_index}.content",
                    json.dumps(processed_content)
                )


@dont_throw
def set_model_input_attributes(span, kwargs, llm_model):
    if not span.is_recording():
        return
    _set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_MODEL, llm_model)
    _set_span_attribute(
        span, f"{GenAIAttributes.GEN_AI_PROMPT}.0.user", kwargs.get("prompt")
    )
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE, kwargs.get("temperature")
    )
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS, kwargs.get("max_output_tokens")
    )
    _set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_TOP_P, kwargs.get("top_p"))
    _set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_TOP_K, kwargs.get("top_k"))
    _set_span_attribute(
        span, SpanAttributes.LLM_PRESENCE_PENALTY, kwargs.get("presence_penalty")
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_FREQUENCY_PENALTY, kwargs.get("frequency_penalty")
    )


@dont_throw
def set_response_attributes(span, llm_model, generation_text):
    if not span.is_recording() or not should_send_prompts():
        return
    _set_span_attribute(span, f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role", "assistant")
    _set_span_attribute(
        span,
        f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content",
        generation_text,
    )


@dont_throw
def set_model_response_attributes(span, llm_model, token_usage):
    if not span.is_recording():
        return
    _set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, llm_model)

    if token_usage:
        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
            token_usage.total_token_count,
        )
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
            token_usage.candidates_token_count,
        )
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS,
            token_usage.prompt_token_count,
        )
