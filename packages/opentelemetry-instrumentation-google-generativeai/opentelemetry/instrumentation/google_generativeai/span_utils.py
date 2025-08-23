import json
import base64
import logging
import asyncio
from opentelemetry.instrumentation.google_generativeai.utils import (
    dont_throw,
    should_send_prompts,
)
from opentelemetry.instrumentation.google_generativeai.config import Config
from opentelemetry.semconv_ai import (
    SpanAttributes,
)
from opentelemetry.trace.status import Status, StatusCode


logger = logging.getLogger(__name__)


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


def _is_image_part(item):
    """Check if item is a Google GenAI Part object containing image data"""
    try:
        # Check if it has the Part attributes we expect for new Google GenAI SDK
        if hasattr(item, 'inline_data') and item.inline_data is not None:
            # Check if it's an image mime type and has data
            if (hasattr(item.inline_data, 'mime_type') and
                    item.inline_data.mime_type and
                    'image/' in item.inline_data.mime_type and
                    hasattr(item.inline_data, 'data') and
                    item.inline_data.data):
                return True
        return False
    except Exception:
        return False


async def _process_image_part(item, trace_id, span_id, content_index):
    """Process a Google GenAI Part object containing image data"""
    if not Config.upload_base64_image:
        return None

    try:
        # Extract format from mime type (e.g., 'image/jpeg' -> 'jpeg')
        image_format = item.inline_data.mime_type.split('/')[1] if item.inline_data.mime_type else 'unknown'
        image_name = f"content_{content_index}.{image_format}"

        # Convert binary data to base64 string for upload
        binary_data = item.inline_data.data
        base64_string = base64.b64encode(binary_data).decode('utf-8')

        # Upload the base64 data
        url = await Config.upload_base64_image(trace_id, span_id, image_name, base64_string)

        # Return OpenAI-compatible format for consistency across LLM providers
        return {
            "type": "image_url",
            "image_url": {"url": url}
        }
    except Exception as e:
        logger.warning(f"Failed to process image part: {e}")
        # Return None to skip adding this image to the span
        return None


def _process_image_part_sync(item, trace_id, span_id, content_index):
    """Synchronous version of image part processing"""
    if not Config.upload_base64_image:
        return None

    try:
        # Extract format from mime type (e.g., 'image/jpeg' -> 'jpeg')
        image_format = item.inline_data.mime_type.split('/')[1] if item.inline_data.mime_type else 'unknown'
        image_name = f"content_{content_index}.{image_format}"

        # Convert binary data to base64 string for upload
        binary_data = item.inline_data.data
        base64_string = base64.b64encode(binary_data).decode('utf-8')

        # Use asyncio.run to call the async upload function in sync context
        url = asyncio.run(Config.upload_base64_image(trace_id, span_id, image_name, base64_string))

        return {
            "type": "image_url",
            "image_url": {"url": url}
        }
    except Exception as e:
        logger.warning(f"Failed to process image part sync: {e}")
        # Return None to skip adding this image to the span
        return None


@dont_throw
async def set_input_attributes(span, args, kwargs, llm_model):
    """Process input arguments, handling both text and image content"""
    if not span.is_recording():
        return

    if not should_send_prompts():
        return

    if "contents" in kwargs:
        contents = kwargs["contents"]
        if isinstance(contents, str):
            # Simple string content in OpenAI format
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_PROMPTS}.0.content",
                json.dumps([{"type": "text", "text": contents}]),
            )
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_PROMPTS}.0.role",
                "user",
            )
        elif isinstance(contents, list):
            # Process content list - could be mixed text and Part objects
            for i, content in enumerate(contents):
                processed_content = []

                if hasattr(content, "parts"):
                    # Content with parts (Google GenAI Content object)
                    for j, part in enumerate(content.parts):
                        if hasattr(part, "text") and part.text:
                            processed_content.append({"type": "text", "text": part.text})
                        elif _is_image_part(part):
                            processed_image = await _process_image_part(
                                part, span.context.trace_id, span.context.span_id, j
                            )
                            if processed_image is not None:
                                processed_content.append(processed_image)
                        else:
                            # Other part types
                            processed_content.append({"type": "text", "text": str(part)})
                elif isinstance(content, str):
                    # Direct string in the list
                    processed_content.append({"type": "text", "text": content})
                elif _is_image_part(content):
                    # Direct Part object that's an image
                    processed_image = await _process_image_part(
                        content, span.context.trace_id, span.context.span_id, 0
                    )
                    if processed_image is not None:
                        processed_content.append(processed_image)
                else:
                    # Other content types
                    processed_content.append({"type": "text", "text": str(content)})

                if processed_content:
                    _set_span_attribute(
                        span,
                        f"{SpanAttributes.LLM_PROMPTS}.{i}.content",
                        json.dumps(processed_content),
                    )
                    _set_span_attribute(
                        span,
                        f"{SpanAttributes.LLM_PROMPTS}.{i}.role",
                        getattr(content, "role", "user"),
                    )
    elif args and len(args) > 0:
        # Handle args - process each argument
        for i, arg in enumerate(args):
            processed_content = []

            if isinstance(arg, str):
                processed_content.append({"type": "text", "text": arg})
            elif isinstance(arg, list):
                for j, subarg in enumerate(arg):
                    if isinstance(subarg, str):
                        processed_content.append({"type": "text", "text": subarg})
                    elif _is_image_part(subarg):
                        processed_image = await _process_image_part(
                            subarg, span.context.trace_id, span.context.span_id, j
                        )
                        if processed_image is not None:
                            processed_content.append(processed_image)
                    else:
                        processed_content.append({"type": "text", "text": str(subarg)})
            elif _is_image_part(arg):
                processed_image = await _process_image_part(
                    arg, span.context.trace_id, span.context.span_id, 0
                )
                if processed_image is not None:
                    processed_content.append(processed_image)
            else:
                processed_content.append({"type": "text", "text": str(arg)})

            if processed_content:
                _set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.{i}.content",
                    json.dumps(processed_content),
                )
                _set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.{i}.role",
                    "user",
                )
    elif "prompt" in kwargs:
        _set_span_attribute(
            span, f"{SpanAttributes.LLM_PROMPTS}.0.content",
            json.dumps([{"type": "text", "text": kwargs["prompt"]}])
        )
        _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")


# Keep sync version for backward compatibility
@dont_throw
def set_input_attributes_sync(span, args, kwargs, llm_model):
    """Synchronous version with image processing support"""
    if not span.is_recording():
        return

    if not should_send_prompts():
        return

    if "contents" in kwargs:
        contents = kwargs["contents"]
        if isinstance(contents, str):
            # Simple string content in OpenAI format
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_PROMPTS}.0.content",
                json.dumps([{"type": "text", "text": contents}]),
            )
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_PROMPTS}.0.role",
                "user",
            )
        elif isinstance(contents, list):
            # Process content list - could be mixed text and Part objects
            for i, content in enumerate(contents):
                processed_content = []

                if hasattr(content, "parts"):
                    # Content with parts (Google GenAI Content object)
                    for j, part in enumerate(content.parts):
                        if hasattr(part, "text") and part.text:
                            processed_content.append({"type": "text", "text": part.text})
                        elif _is_image_part(part):
                            processed_image = _process_image_part_sync(
                                part, span.context.trace_id, span.context.span_id, j
                            )
                            if processed_image is not None:
                                processed_content.append(processed_image)
                        else:
                            # Other part types
                            processed_content.append({"type": "text", "text": str(part)})
                elif isinstance(content, str):
                    # Direct string in the list
                    processed_content.append({"type": "text", "text": content})
                elif _is_image_part(content):
                    # Direct Part object that's an image
                    processed_image = _process_image_part_sync(
                        content, span.context.trace_id, span.context.span_id, 0
                    )
                    if processed_image is not None:
                        processed_content.append(processed_image)
                else:
                    # Other content types
                    processed_content.append({"type": "text", "text": str(content)})

                if processed_content:
                    _set_span_attribute(
                        span,
                        f"{SpanAttributes.LLM_PROMPTS}.{i}.content",
                        json.dumps(processed_content),
                    )
                    _set_span_attribute(
                        span,
                        f"{SpanAttributes.LLM_PROMPTS}.{i}.role",
                        getattr(content, "role", "user"),
                    )
    elif args and len(args) > 0:
        # Handle args - process each argument
        for i, arg in enumerate(args):
            processed_content = []

            if isinstance(arg, str):
                processed_content.append({"type": "text", "text": arg})
            elif isinstance(arg, list):
                for j, subarg in enumerate(arg):
                    if isinstance(subarg, str):
                        processed_content.append({"type": "text", "text": subarg})
                    elif _is_image_part(subarg):
                        processed_image = _process_image_part_sync(
                            subarg, span.context.trace_id, span.context.span_id, j
                        )
                        if processed_image is not None:
                            processed_content.append(processed_image)
                    else:
                        processed_content.append({"type": "text", "text": str(subarg)})
            elif _is_image_part(arg):
                processed_image = _process_image_part_sync(
                    arg, span.context.trace_id, span.context.span_id, 0
                )
                if processed_image is not None:
                    processed_content.append(processed_image)
            else:
                processed_content.append({"type": "text", "text": str(arg)})

            if processed_content:
                _set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.{i}.content",
                    json.dumps(processed_content),
                )
                _set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.{i}.role",
                    "user",
                )
    elif "prompt" in kwargs:
        _set_span_attribute(
            span, f"{SpanAttributes.LLM_PROMPTS}.0.content",
            json.dumps([{"type": "text", "text": kwargs["prompt"]}])
        )
        _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")


def set_model_request_attributes(span, kwargs, llm_model):
    if not span.is_recording():
        return
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, llm_model)
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_TEMPERATURE, kwargs.get("temperature")
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, kwargs.get("max_output_tokens")
    )
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_TOP_P, kwargs.get("top_p"))
    _set_span_attribute(span, SpanAttributes.LLM_TOP_K, kwargs.get("top_k"))
    _set_span_attribute(
        span, SpanAttributes.LLM_PRESENCE_PENALTY, kwargs.get("presence_penalty")
    )
    _set_span_attribute(
        span, SpanAttributes.LLM_FREQUENCY_PENALTY, kwargs.get("frequency_penalty")
    )


@dont_throw
def set_response_attributes(span, response, llm_model):
    if not should_send_prompts():
        return
    if hasattr(response, "usage_metadata"):
        if isinstance(response.text, list):
            for index, item in enumerate(response):
                prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
                _set_span_attribute(span, f"{prefix}.content", item.text)
                _set_span_attribute(span, f"{prefix}.role", "assistant")
        elif isinstance(response.text, str):
            _set_span_attribute(
                span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content", response.text
            )
            _set_span_attribute(
                span, f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant"
            )
    else:
        if isinstance(response, list):
            for index, item in enumerate(response):
                prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
                _set_span_attribute(span, f"{prefix}.content", item)
                _set_span_attribute(span, f"{prefix}.role", "assistant")
        elif isinstance(response, str):
            _set_span_attribute(
                span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content", response
            )
            _set_span_attribute(
                span, f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant"
            )


def set_model_response_attributes(span, response, llm_model):
    if not span.is_recording():
        return

    _set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, llm_model)

    if hasattr(response, "usage_metadata"):
        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
            response.usage_metadata.total_token_count,
        )
        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
            response.usage_metadata.candidates_token_count,
        )
        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
            response.usage_metadata.prompt_token_count,
        )

    span.set_status(Status(StatusCode.OK))
