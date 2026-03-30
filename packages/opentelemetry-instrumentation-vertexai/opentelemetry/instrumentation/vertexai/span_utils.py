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
        if item.mime_type and "image/" in item.mime_type and item.inline_data:
            data = getattr(item.inline_data, "data", None)
            if isinstance(data, (bytes, bytearray)) and len(data) > 0:
                return True

        return False
    except Exception:
        return False


def _map_vertex_finish_reason(finish_reason):
    if finish_reason is None:
        return None
    name = getattr(finish_reason, "name", None) or str(finish_reason)
    mapping = {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
        "BLOCKLIST": "content_filter",
        "PROHIBITED_CONTENT": "content_filter",
        "SPII": "content_filter",
        "IMAGE_SAFETY": "content_filter",
        "IMAGE_PROHIBITED_CONTENT": "content_filter",
        "IMAGE_RECITATION": "content_filter",
        "LANGUAGE": "content_filter",
        "FINISH_REASON_UNSPECIFIED": None,
        "MALFORMED_FUNCTION_CALL": "error",
        "OTHER": "error",
        "UNEXPECTED_TOOL_CALL": "error",
        "NO_IMAGE": "error",
        "IMAGE_OTHER": "error",
    }
    return mapping.get(name)


def _normalize_message_role(role):
    if role is None:
        return "user"
    r = str(role).lower()
    if r == "model":
        return "assistant"
    return r


def _parse_vertex_function_args(args):
    if args is None:
        return {}
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            return json.loads(args)
        except (json.JSONDecodeError, TypeError):
            return {"_raw": args}
    return {"_value": args}


def _vertex_function_response_to_str(response):
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    try:
        return json.dumps(response, default=str)
    except TypeError:
        return str(response)


def accumulate_vertex_stream_finish_reasons(ordered, seen, chunk):
    for cand in getattr(chunk, "candidates", None) or []:
        mapped = _map_vertex_finish_reason(getattr(cand, "finish_reason", None))
        if mapped is not None and mapped not in seen:
            seen.add(mapped)
            ordered.append(mapped)


def _parts_from_vertex_part_sync(part, span, part_index):
    out = []
    t = getattr(part, "text", None)
    if isinstance(t, str) and t:
        out.append({"type": "text", "content": t})
    img = _otel_image_part_vertex_sync(part, span, part_index)
    if img:
        out.append(img)
    fc = getattr(part, "function_call", None)
    if fc is not None:
        fid = getattr(fc, "id", None) or ""
        fname = getattr(fc, "name", None) or ""
        out.append(
            {
                "type": "tool_call",
                "id": fid,
                "name": fname,
                "arguments": _parse_vertex_function_args(getattr(fc, "args", None)),
            }
        )
    fr = getattr(part, "function_response", None)
    if fr is not None:
        fid = getattr(fr, "id", None) or ""
        out.append(
            {
                "type": "tool_call_response",
                "id": fid,
                "response": _vertex_function_response_to_str(
                    getattr(fr, "response", None)
                ),
            }
        )
    if not out:
        out.append({"type": "text", "content": str(part)})
    return out


async def _parts_from_vertex_part_async(part, span, part_index):
    out = []
    t = getattr(part, "text", None)
    if isinstance(t, str) and t:
        out.append({"type": "text", "content": t})
    if _is_base64_image_part(part):
        img = await _otel_image_part_vertex_async(part, span, part_index)
        if img:
            out.append(img)
    fc = getattr(part, "function_call", None)
    if fc is not None:
        fid = getattr(fc, "id", None) or ""
        fname = getattr(fc, "name", None) or ""
        out.append(
            {
                "type": "tool_call",
                "id": fid,
                "name": fname,
                "arguments": _parse_vertex_function_args(getattr(fc, "args", None)),
            }
        )
    fr = getattr(part, "function_response", None)
    if fr is not None:
        fid = getattr(fr, "id", None) or ""
        out.append(
            {
                "type": "tool_call_response",
                "id": fid,
                "response": _vertex_function_response_to_str(
                    getattr(fr, "response", None)
                ),
            }
        )
    if not out:
        out.append({"type": "text", "content": str(part)})
    return out


def _vertex_parts_list_from_content_sync(content, span):
    parts_acc = []
    for idx, p in enumerate(content.parts):
        parts_acc.extend(_parts_from_vertex_part_sync(p, span, idx))
    return parts_acc


async def _vertex_parts_list_from_content_async(content, span):
    parts_acc = []
    for idx, p in enumerate(content.parts):
        parts_acc.extend(await _parts_from_vertex_part_async(p, span, idx))
    return parts_acc


def _vertex_expand_arg_to_messages_sync(argument, span):
    messages = []
    if hasattr(argument, "parts") and hasattr(argument, "role"):
        parts = _vertex_parts_list_from_content_sync(argument, span)
        if parts:
            messages.append(
                {"role": _normalize_message_role(argument.role), "parts": parts}
            )
        return messages
    if isinstance(argument, list):
        if argument and all(
            hasattr(x, "parts") and hasattr(x, "role") for x in argument
        ):
            for c in argument:
                parts = _vertex_parts_list_from_content_sync(c, span)
                if parts:
                    messages.append(
                        {"role": _normalize_message_role(c.role), "parts": parts}
                    )
            return messages
    processed = _process_vertexai_argument_sync(argument, span)
    if processed:
        messages.append({"role": "user", "parts": processed})
    return messages


async def _vertex_expand_arg_to_messages_async(argument, span):
    messages = []
    if hasattr(argument, "parts") and hasattr(argument, "role"):
        parts = await _vertex_parts_list_from_content_async(argument, span)
        if parts:
            messages.append(
                {"role": _normalize_message_role(argument.role), "parts": parts}
            )
        return messages
    if isinstance(argument, list):
        if argument and all(
            hasattr(x, "parts") and hasattr(x, "role") for x in argument
        ):
            for c in argument:
                parts = await _vertex_parts_list_from_content_async(c, span)
                if parts:
                    messages.append(
                        {"role": _normalize_message_role(c.role), "parts": parts}
                    )
            return messages
    processed = await _process_vertexai_argument(argument, span)
    if processed:
        messages.append({"role": "user", "parts": processed})
    return messages


def _output_messages_from_vertex_response(span, response):
    messages = []
    for cand in getattr(response, "candidates", None) or []:
        parts = []
        content = getattr(cand, "content", None)
        if content and getattr(content, "parts", None):
            for idx, p in enumerate(content.parts):
                parts.extend(_parts_from_vertex_part_sync(p, span, idx))
        fr = _map_vertex_finish_reason(getattr(cand, "finish_reason", None))
        role = "assistant"
        if content and getattr(content, "role", None):
            role = _normalize_message_role(content.role)
        msg = {"role": role, "parts": parts, "finish_reason": fr}
        messages.append(msg)
    if not messages and getattr(response, "text", None):
        messages.append(
            {
                "role": "assistant",
                "parts": [{"type": "text", "content": response.text}],
                "finish_reason": None,
            }
        )
    return messages


def _vertex_system_instruction_to_parts(si, span):
    if isinstance(si, str):
        return [{"type": "text", "content": si}]
    if hasattr(si, "parts") and si.parts:
        out = []
        for idx, p in enumerate(si.parts):
            out.extend(_parts_from_vertex_part_sync(p, span, idx))
        return out
    return [{"type": "text", "content": str(si)}]


async def _otel_image_part_vertex_async(item, span, item_index):
    if not _is_base64_image_part(item):
        return None
    mime = item.mime_type or "application/octet-stream"
    if Config.upload_base64_image:
        uploaded = await _process_image_part(
            item, span.context.trace_id, span.context.span_id, item_index
        )
        if uploaded and uploaded.get("image_url", {}).get("url"):
            return {
                "type": "uri",
                "modality": "image",
                "uri": uploaded["image_url"]["url"],
            }
    binary_data = item.inline_data.data
    b64 = base64.b64encode(binary_data).decode("utf-8")
    return {"type": "blob", "modality": "image", "mime_type": mime, "content": b64}


def _otel_image_part_vertex_sync(item, span, item_index):
    if not _is_base64_image_part(item):
        return None
    mime = item.mime_type or "application/octet-stream"
    if Config.upload_base64_image:
        uploaded = _process_image_part_sync(
            item, span.context.trace_id, span.context.span_id, item_index
        )
        if uploaded and uploaded.get("image_url", {}).get("url"):
            return {
                "type": "uri",
                "modality": "image",
                "uri": uploaded["image_url"]["url"],
            }
    binary_data = item.inline_data.data
    b64 = base64.b64encode(binary_data).decode("utf-8")
    return {"type": "blob", "modality": "image", "mime_type": mime, "content": b64}


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
    """Sync upload path; uses ``run_async`` + thread when a loop is already running.

    Can deadlock if the upload depends on the same event loop (e.g. some ASGI/Jupyter
    setups). Prefer async request paths where possible.
    """
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
        return [{"type": "text", "content": argument}]

    elif isinstance(argument, list):
        content_list = copy.deepcopy(argument)
        processed_items = []

        for item_index, content_item in enumerate(content_list):
            processed_item = await _process_content_item_vertexai(content_item, span, item_index)
            if processed_item is not None:
                processed_items.append(processed_item)

        return processed_items

    else:
        processed_item = await _process_content_item_vertexai(argument, span, 0)
        return [processed_item] if processed_item is not None else []


async def _process_content_item_vertexai(content_item, span, item_index):
    """Process a single content item for VertexAI"""
    if isinstance(content_item, str):
        return {"type": "text", "content": content_item}

    elif _is_base64_image_part(content_item):
        return await _otel_image_part_vertex_async(content_item, span, item_index)

    elif hasattr(content_item, "text"):
        t = getattr(content_item, "text", None)
        if isinstance(t, str) and t:
            return {"type": "text", "content": t}

    return {"type": "text", "content": str(content_item)}


def _process_vertexai_argument_sync(argument, span):
    """Synchronous version of argument processing for VertexAI"""
    if isinstance(argument, str):
        return [{"type": "text", "content": argument}]

    elif isinstance(argument, list):
        content_list = copy.deepcopy(argument)
        processed_items = []

        for item_index, content_item in enumerate(content_list):
            processed_item = _process_content_item_vertexai_sync(content_item, span, item_index)
            if processed_item is not None:
                processed_items.append(processed_item)

        return processed_items

    else:
        processed_item = _process_content_item_vertexai_sync(argument, span, 0)
        return [processed_item] if processed_item is not None else []


def _process_content_item_vertexai_sync(content_item, span, item_index):
    """Synchronous version of content item processing for VertexAI"""
    if isinstance(content_item, str):
        return {"type": "text", "content": content_item}

    elif _is_base64_image_part(content_item):
        return _otel_image_part_vertex_sync(content_item, span, item_index)

    elif hasattr(content_item, "text"):
        t = getattr(content_item, "text", None)
        if isinstance(t, str) and t:
            return {"type": "text", "content": t}

    return {"type": "text", "content": str(content_item)}


@dont_throw
async def set_input_attributes(span, args):
    """Process input arguments, handling both text and image content"""
    if not span.is_recording():
        return
    if should_send_prompts() and args is not None and len(args) > 0:
        messages = []
        for argument in args:
            messages.extend(await _vertex_expand_arg_to_messages_async(argument, span))
        if messages:
            _set_span_attribute(
                span, GenAIAttributes.GEN_AI_INPUT_MESSAGES, json.dumps(messages)
            )


# Sync version with image processing support
@dont_throw
def set_input_attributes_sync(span, args):
    """Synchronous version with image processing support"""
    if not span.is_recording():
        return
    if should_send_prompts() and args is not None and len(args) > 0:
        messages = []
        for argument in args:
            messages.extend(_vertex_expand_arg_to_messages_sync(argument, span))
        if messages:
            _set_span_attribute(
                span, GenAIAttributes.GEN_AI_INPUT_MESSAGES, json.dumps(messages)
            )


@dont_throw
def set_model_input_attributes(span, kwargs, llm_model):
    if not span.is_recording():
        return
    _set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_MODEL, llm_model)

    prompt = kwargs.get("prompt")
    if prompt:
        messages = [
            {"role": "user", "parts": [{"type": "text", "content": prompt}]},
        ]
        _set_span_attribute(
            span, GenAIAttributes.GEN_AI_INPUT_MESSAGES, json.dumps(messages)
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
        span, GenAIAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY, kwargs.get("presence_penalty")
    )
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY, kwargs.get("frequency_penalty")
    )

    if should_send_prompts():
        si = _vertex_system_instruction_from_kwargs(kwargs)
        if si is not None:
            try:
                parts = _vertex_system_instruction_to_parts(si, span)
                if parts:
                    _set_span_attribute(
                        span,
                        GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS,
                        json.dumps(parts),
                    )
            except Exception:
                pass


def _vertex_system_instruction_from_kwargs(kwargs):
    si = kwargs.get("system_instruction")
    if si is not None:
        return si
    config = kwargs.get("config")
    if config is not None:
        si = getattr(config, "system_instruction", None)
        if si is not None:
            return si
    return None


@dont_throw
def set_response_attributes(span, llm_model, generation_or_text, finish_reason_otel=None):
    if not span.is_recording() or not should_send_prompts():
        return
    if hasattr(generation_or_text, "candidates"):
        messages = _output_messages_from_vertex_response(span, generation_or_text)
        if messages:
            _set_span_attribute(
                span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES, json.dumps(messages)
            )
        return
    parts = []
    if generation_or_text:
        parts.append({"type": "text", "content": generation_or_text})
    msg = {"role": "assistant", "parts": parts, "finish_reason": finish_reason_otel}
    if parts or finish_reason_otel is not None:
        _set_span_attribute(
            span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES, json.dumps([msg])
        )


@dont_throw
def set_model_response_attributes(
    span, llm_model, token_usage, response_meta=None, stream_finish_reasons=None
):
    if not span.is_recording():
        return
    _set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, llm_model)

    if response_meta is not None:
        if stream_finish_reasons is not None:
            reasons = stream_finish_reasons
        else:
            reasons = []
            for cand in getattr(response_meta, "candidates", None) or []:
                mapped = _map_vertex_finish_reason(getattr(cand, "finish_reason", None))
                if mapped is not None:
                    reasons.append(mapped)
        if reasons:
            _set_span_attribute(
                span, GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS, reasons
            )
        rid = getattr(response_meta, "response_id", None)
        if rid:
            _set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_ID, rid)

    if token_usage:
        _set_span_attribute(
            span,
            SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS,
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
