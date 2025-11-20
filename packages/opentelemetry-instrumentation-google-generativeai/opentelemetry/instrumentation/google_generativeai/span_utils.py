import json
import base64
import logging
import asyncio
import threading
from opentelemetry.instrumentation.google_generativeai.utils import (
    dont_throw,
    should_send_prompts,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
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
        image_format = item.inline_data.mime_type.split('/')[1] if item.inline_data.mime_type else 'unknown'
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


async def _process_content_item(content_item, span):
    """Process a single content item, handling different types (Content objects, strings, Parts)"""
    processed_content = []

    if hasattr(content_item, "parts"):
        # Content with parts (Google GenAI Content object)
        for part_index, part in enumerate(content_item.parts):
            processed_part = await _process_content_part(part, span, part_index)
            if processed_part is not None:
                processed_content.append(processed_part)
    elif isinstance(content_item, str):
        # Direct string in the list
        processed_content.append({"type": "text", "text": content_item})
    elif _is_image_part(content_item):
        # Direct Part object that's an image
        processed_image = await _process_image_part(
            content_item, span.context.trace_id, span.context.span_id, 0
        )
        if processed_image is not None:
            processed_content.append(processed_image)
    else:
        # Other content types
        processed_content.append({"type": "text", "text": str(content_item)})

    return processed_content


async def _process_content_part(part, span, part_index):
    """Process a single part within a Content object"""
    if hasattr(part, "text") and part.text:
        return {"type": "text", "text": part.text}
    elif _is_image_part(part):
        return await _process_image_part(
            part, span.context.trace_id, span.context.span_id, part_index
        )
    else:
        # Other part types
        return {"type": "text", "text": str(part)}


def _set_prompt_attributes(span, prompt_index, processed_content, content_item):
    """Set span attributes for a processed prompt"""
    _set_span_attribute(
        span,
        f"{SpanAttributes.LLM_PROMPTS}.{prompt_index}.content",
        json.dumps(processed_content),
    )
    _set_span_attribute(
        span,
        f"{SpanAttributes.LLM_PROMPTS}.{prompt_index}.role",
        getattr(content_item, "role", "user"),
    )


async def _process_argument(argument, span):
    """Process a single argument from args list"""
    processed_content = []

    if isinstance(argument, str):
        processed_content.append({"type": "text", "text": argument})
    elif isinstance(argument, list):
        for sub_index, sub_item in enumerate(argument):
            if isinstance(sub_item, str):
                processed_content.append({"type": "text", "text": sub_item})
            elif _is_image_part(sub_item):
                processed_image = await _process_image_part(
                    sub_item, span.context.trace_id, span.context.span_id, sub_index
                )
                if processed_image is not None:
                    processed_content.append(processed_image)
            else:
                processed_content.append({"type": "text", "text": str(sub_item)})
    elif _is_image_part(argument):
        processed_image = await _process_image_part(
            argument, span.context.trace_id, span.context.span_id, 0
        )
        if processed_image is not None:
            processed_content.append(processed_image)
    else:
        processed_content.append({"type": "text", "text": str(argument)})

    return processed_content


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
                f"{GenAIAttributes.GEN_AI_PROMPT}.0.content",
                contents,
            )
            _set_span_attribute(
                span,
                f"{GenAIAttributes.GEN_AI_PROMPT}.0.role",
                "user",
            )
        elif isinstance(contents, list):
            for prompt_index, content_item in enumerate(contents):
                processed_content = await _process_content_item(content_item, span)

                if processed_content:
                    _set_prompt_attributes(span, prompt_index, processed_content, content_item)

    elif args and len(args) > 0:
        # Handle args - process each argument
        for arg_index, argument in enumerate(args):
            processed_content = await _process_argument(argument, span)

            if processed_content:
                _set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.{arg_index}.content",
                    json.dumps(processed_content),
                )
                _set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.{arg_index}.role",
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
                f"{GenAIAttributes.GEN_AI_PROMPT}.0.content",
                json.dumps([{"type": "text", "text": contents}]),
            )
            _set_span_attribute(
                span,
                f"{GenAIAttributes.GEN_AI_PROMPT}.0.role",
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
            span, f"{GenAIAttributes.GEN_AI_PROMPT}.0.content",
            json.dumps([{"type": "text", "text": kwargs["prompt"]}])
        )
        _set_span_attribute(span, f"{GenAIAttributes.GEN_AI_PROMPT}.0.role", "user")


def set_model_request_attributes(span, kwargs, llm_model):
    if not span.is_recording():
        return
    _set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_MODEL, llm_model)
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

    generation_config = kwargs.get("generation_config")
    if generation_config and hasattr(generation_config, "response_schema"):
        try:
            _set_span_attribute(
                span,
                SpanAttributes.LLM_REQUEST_STRUCTURED_OUTPUT_SCHEMA,
                json.dumps(generation_config.response_schema),
            )
        except Exception:
            pass

    if "response_schema" in kwargs:
        try:
            _set_span_attribute(
                span,
                SpanAttributes.LLM_REQUEST_STRUCTURED_OUTPUT_SCHEMA,
                json.dumps(kwargs.get("response_schema")),
            )
        except Exception:
            pass


@dont_throw
def set_response_attributes(span, response, llm_model):
    if not should_send_prompts():
        return
    if hasattr(response, "usage_metadata"):
        if isinstance(response.text, list):
            for index, item in enumerate(response):
                prefix = f"{GenAIAttributes.GEN_AI_COMPLETION}.{index}"
                _set_span_attribute(span, f"{prefix}.content", item.text)
                _set_span_attribute(span, f"{prefix}.role", "assistant")
        elif isinstance(response.text, str):
            _set_span_attribute(
                span, f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content", response.text
            )
            _set_span_attribute(
                span, f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role", "assistant"
            )
    else:
        if isinstance(response, list):
            for index, item in enumerate(response):
                prefix = f"{GenAIAttributes.GEN_AI_COMPLETION}.{index}"
                _set_span_attribute(span, f"{prefix}.content", item)
                _set_span_attribute(span, f"{prefix}.role", "assistant")
        elif isinstance(response, str):
            _set_span_attribute(
                span, f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content", response
            )
            _set_span_attribute(
                span, f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role", "assistant"
            )


def set_model_response_attributes(span, response, llm_model):
    if not span.is_recording():
        return

    _set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, llm_model)

    if hasattr(response, "usage_metadata"):
        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
            response.usage_metadata.total_token_count,
        )
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
            response.usage_metadata.candidates_token_count,
        )
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS,
            response.usage_metadata.prompt_token_count,
        )

    span.set_status(Status(StatusCode.OK))
