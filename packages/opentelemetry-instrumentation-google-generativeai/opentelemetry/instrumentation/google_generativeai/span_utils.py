import json
import base64
import logging
import asyncio
import threading
from opentelemetry.instrumentation.google_generativeai.config import Config
from opentelemetry.instrumentation.google_generativeai.utils import (
    dont_throw,
    should_send_prompts,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes
)
from opentelemetry.semconv_ai import 
    SpanAttributes,
)
from opentelemetry.trace.status import Status, StatusCode

logger = logging.getLogger(__name__)


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


def _map_gemini_finish_reason(finish_reason):
    """Map Gemini FinishReason to OTel finish reason string.

    Returns ``""`` for ``None``, unspecified, and unmapped values.
    """
    if finish_reason is None:
        return ""
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
        "FINISH_REASON_UNSPECIFIED": "",
        "MALFORMED_FUNCTION_CALL": "error",
        "OTHER": "error",
        "UNEXPECTED_TOOL_CALL": "error",
        "NO_IMAGE": "error",
        "IMAGE_OTHER": "error",
    }
    return mapping.get(name, "")


def _normalize_message_role(role):
    if role is None:
        return "user"
    r = str(role).lower()
    if r == "model":
        return "assistant"
    return r


def _parse_function_call_arguments(args):
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


def _function_response_to_str(response):
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    try:
        return json.dumps(response, default=str)
    except TypeError:
        return str(response)


def _is_image_part(item):
    """Check if item is a Google GenAI Part object containing image data"""
    try:
        if hasattr(item, "inline_data") and item.inline_data is not None:
            if (
                hasattr(item.inline_data, "mime_type")
                and item.inline_data.mime_type
                and "image/" in item.inline_data.mime_type
                and hasattr(item.inline_data, "data")
                and item.inline_data.data
            ):
                return True
        return False
    except Exception:
        return False


def _otel_image_part_from_genai_part(part, span, part_index, sync):
    """BlobPart after upload attempt, or UriPart when upload is configured and succeeds."""
    if not _is_image_part(part):
        return None
    mime = (
        part.inline_data.mime_type
        if hasattr(part, "inline_data") and part.inline_data
        else None
    ) or "application/octet-stream"
    if Config.upload_base64_image:
        if sync:
            uploaded = _process_image_part_sync(
                part, span.context.trace_id, span.context.span_id, part_index
            )
        else:
            return None  # async path must use _otel_image_part_from_genai_part_async
        if uploaded and uploaded.get("image_url") and uploaded["image_url"].get("url"):
            return {
                "type": "uri",
                "modality": "image",
                "uri": uploaded["image_url"]["url"],
            }
    binary_data = part.inline_data.data
    b64 = base64.b64encode(binary_data).decode("utf-8")
    return {"type": "blob", "modality": "image", "mime_type": mime, "content": b64}


async def _otel_image_part_from_genai_part_async(part, span, part_index):
    if not _is_image_part(part):
        return None
    mime = (
        part.inline_data.mime_type
        if hasattr(part, "inline_data") and part.inline_data
        else None
    ) or "application/octet-stream"
    if Config.upload_base64_image:
        uploaded = await _process_image_part(
            part, span.context.trace_id, span.context.span_id, part_index
        )
        if uploaded and uploaded.get("image_url") and uploaded["image_url"].get("url"):
            return {
                "type": "uri",
                "modality": "image",
                "uri": uploaded["image_url"]["url"],
            }
    binary_data = part.inline_data.data
    b64 = base64.b64encode(binary_data).decode("utf-8")
    return {"type": "blob", "modality": "image", "mime_type": mime, "content": b64}


async def _process_image_part(item, trace_id, span_id, content_index):
    """Upload image when configured; used only from async content processing."""
    if not Config.upload_base64_image:
        return None
    try:
        image_format = (
            item.inline_data.mime_type.split("/")[1]
            if item.inline_data.mime_type
            else ""
        )
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
        return None


def run_async(method):
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
    """Sync path for optional image upload.

    Uses ``run_async`` in a new thread when the current thread already has a
    running event loop (e.g. FastAPI/Jupyter). That pattern can deadlock if the
    async upload waits on the same loop — prefer async instrumentation paths
    where possible.
    """
    if not Config.upload_base64_image:
        return None
    try:
        image_format = (
            item.inline_data.mime_type.split("/")[1]
            if item.inline_data.mime_type
            else ""
        )
        image_name = f"content_{content_index}.{image_format}"
        # Convert binary data to base64 string for upload
        binary_data = item.inline_data.data
        base64_string = base64.b64encode(binary_data).decode('utf-8')
        # Use OpenAI's run_async pattern to handle the async upload function
        url = None

        async def upload_task():
            nonlocal url
            url = await Config.upload_base64_image(
                str(trace_id), str(span_id), image_name, base64_string
            )

        run_async(upload_task())
        return {
            "type": "image_url",
            "image_url": {"url": url}
        }
    except Exception as e:
        logger.warning(f"Failed to process image part sync: {e}")
        return None


async def _process_content_item(content_item, span):
    """Process a single content item, handling different types (Content objects, strings, Parts)"""
    processed_content = []
    if hasattr(content_item, "parts"):
        # Content with parts (Google GenAI Content object)
        for part_index, part in enumerate(content_item.parts):
            result = await _process_content_part(part, span, part_index)
            if result:
                processed_content.append(result)
    elif isinstance(content_item, str):
        # Direct string in the list
        processed_content.append({"type": "text", "content": content_item})
    elif _is_image_part(content_item):
        # Direct Part object that's an image
        processed_image = await _process_image_part(
            content_item,
            span.context.trace_id,
            span.context.span_id,
            0
        )
        if processed_image is not None:
            processed_content.append(processed_image)
    else:
        # Other content types
        processed_content.append({"type": "text", "content": str(content_item)})
    return processed_content


async def _process_content_part(part, span, part_index):
    """Process a single part within a Content object"""
    if hasattr(part, "text") and part.text:
        return {"type": "text", "content": part.text}
    elif _is_image_part(part):
        return await _process_image_part(
            part,
            span.context.trace_id,
            span.context.span_id,
            part_index
        )
    else:
        # Other part types
        return {"type": "text", "content": str(part)}


async def _process_argument(argument, span):
    """Process a single argument from args list"""
    processed_content = []
    if isinstance(argument, str):
        processed_content.append({"type": "text", "content": argument})
    elif isinstance(argument, list):
        for sub_index, sub_item in enumerate(argument):
            if isinstance(sub_item, str):
                processed_content.append({"type": "text", "content": sub_item})
            elif _is_image_part(sub_item):
                processed_image = await _process_image_part(
                    sub_item,
                    span.context.trace_id,
                    span.context.span_id,
                    sub_index
                )
                if processed_image is not None:
                    processed_content.append(processed_image)
            else:
                processed_content.append({"type": "text", "content": str(sub_item)})
    elif _is_image_part(argument):
        processed_image = await _process_image_part(
            argument,
            span.context.trace_id,
            span.context.span_id,
            0
        )
        if processed_image is not None:
            processed_content.append(processed_image)
    else:
        processed_content.append({"type": "text", "content": str(argument)})
    return processed_content


@dont_throw
async def set_input_attributes(span, args, kwargs, llm_model):
    if not span.is_recording():
        return
    if not should_send_prompts():
        return

    input_messages = []

    if "contents" in kwargs:
        contents = kwargs["contents"]
        if isinstance(contents, str):
            # Simple string content
            input_messages.append({
                "role": "user",
                "parts": [{"type": "text", "content": contents}],
            })
        elif hasattr(contents, "parts"):
            # Single Content object (not a list)
            processed_content = await _process_content_item(contents, span)
            if processed_content:
                input_messages.append({
                    "role": getattr(contents, "role", "user"),
                    "parts": processed_content,
                })
        elif isinstance(contents, list):
            if contents and hasattr(contents[0], "parts"):
                # Multi-turn: list of Content objects
                for content_item in contents:
                    processed_content = await _process_content_item(content_item, span)
                    if processed_content:
                        role = getattr(content_item, "role", "user")
                        input_messages.append({
                            "role": role,
                            "parts": processed_content,
                        })
            else:
                # Single-turn: list of Part objects or strings
                parts = []
                for content_item in contents:
                    items = await _process_content_item(content_item, span)
                    parts.extend(items)
                if parts:
                    input_messages.append({
                        "role": "user",
                        "parts": parts,
                    })
    elif args and len(args) > 0:
        # Handle args - process each argument
        for argument in args:
            processed_content = await _process_argument(argument, span)
            if processed_content:
                input_messages.append({
                    "role": "user",
                    "parts": processed_content,
                })
    elif "prompt" in kwargs:
        input_messages.append({
            "role": "user",
            "parts": [{"type": "text", "content": kwargs["prompt"]}],
        })

    if input_messages:
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_INPUT_MESSAGES,
            json.dumps(input_messages),
        )


@dont_throw
def set_input_attributes_sync(span, args, kwargs, llm_model):
    if not span.is_recording():
        return
    if not should_send_prompts():
        return

    input_messages = []

    if "contents" in kwargs:
        contents = kwargs["contents"]
        if isinstance(contents, str):
            # Simple string content
            input_messages.append({
                "role": "user",
                "parts": [{"type": "text", "content": contents}],
            })
        elif hasattr(contents, "parts"):
            # Single Content object (not a list)
            processed_content = []
            for j, part in enumerate(contents.parts):
                if hasattr(part, "text") and part.text:
                    processed_content.append({"type": "text", "content": part.text})
                elif _is_image_part(part):
                    processed_image = _process_image_part_sync(
                        part, span.context.trace_id, span.context.span_id, j
                    )
                    if processed_image is not None:
                        processed_content.append(processed_image)
                else:
                    processed_content.append({"type": "text", "content": str(part)})
            if processed_content:
                input_messages.append({
                    "role": getattr(contents, "role", "user"),
                    "parts": processed_content,
                })
        elif isinstance(contents, list):
            if contents and hasattr(contents[0], "parts"):
                # Multi-turn: list of Content objects
                for content in contents:
                    processed_content = []
                    for j, part in enumerate(content.parts):
                        if hasattr(part, "text") and part.text:
                            processed_content.append({"type": "text", "content": part.text})
                        elif _is_image_part(part):
                            processed_image = _process_image_part_sync(
                                part, span.context.trace_id, span.context.span_id, j
                            )
                            if processed_image is not None:
                                processed_content.append(processed_image)
                        else:
                            processed_content.append({"type": "text", "content": str(part)})
                    if processed_content:
                        input_messages.append({
                            "role": getattr(content, "role", "user"),
                            "parts": processed_content,
                        })
            else:
                # Single-turn: list of Part objects or strings
                parts = []
                for content in contents:
                    if isinstance(content, str):
                        parts.append({"type": "text", "content": content})
                    elif _is_image_part(content):
                        processed_image = _process_image_part_sync(
                            content, span.context.trace_id, span.context.span_id, 0
                        )
                        if processed_image is not None:
                            parts.append(processed_image)
                    else:
                        parts.append({"type": "text", "content": str(content)})
                if parts:
                    input_messages.append({
                        "role": "user",
                        "parts": parts,
                    })
    elif args and len(args) > 0:
        # Handle args - process each argument
        for arg in args:
            processed_content = []
            if isinstance(arg, str):
                processed_content.append({"type": "text", "content": arg})
            elif isinstance(arg, list):
                for j, subarg in enumerate(arg):
                    if isinstance(subarg, str):
                        processed_content.append({"type": "text", "content": subarg})
                    elif _is_image_part(subarg):
                        processed_image = _process_image_part_sync(
                            subarg,
                            span.context.trace_id,
                            span.context.span_id,
                            j
                        )
                        if processed_image is not None:
                            processed_content.append(processed_image)
                    else:
                        processed_content.append({"type": "text", "content": str(subarg)})
            elif _is_image_part(arg):
                processed_image = _process_image_part_sync(
                    arg,
                    span.context.trace_id,
                    span.context.span_id,
                    0
                )
                if processed_image is not None:
                    processed_content.append(processed_image)
            else:
                processed_content.append({"type": "text", "content": str(arg)})

            if processed_content:
                input_messages.append({
                    "role": "user",
                    "parts": processed_content,
                })
    elif "prompt" in kwargs:
        input_messages.append({
            "role": "user",
            "parts": [{"type": "text", "content": kwargs["prompt"]}],
        })

    if input_messages:
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_INPUT_MESSAGES,
            json.dumps(input_messages),
        )


def set_model_request_attributes(span, kwargs, llm_model):
    if not span.is_recording():
        return
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE, kwargs.get("temperature")
    )
    _set_span_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS, kwargs.get("max_output_tokens")
    )
    _set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_TOP_P, kwargs.get("top_p"))
    _set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_TOP_K, kwargs.get("top_k"))
    _set_span_attribute(
        span,
        SpanAttributes.LLM_PRESENCE_PENALTY,
        kwargs.get("presence_penalty"),
    )
    _set_span_attribute(
        span,
        SpanAttributes.LLM_FREQUENCY_PENALTY,
        kwargs.get("frequency_penalty"),
    )
    generation_config = kwargs.get("generation_config")
    if generation_config and hasattr(generation_config, "response_schema"):
        try:
            _set_span_attribute(
                span,
                SpanAttributes.GEN_AI_REQUEST_STRUCTURED_OUTPUT_SCHEMA,
                json.dumps(generation_config.response_schema),
            )
        except Exception:
            pass
    if "response_schema" in kwargs:
        try:
            _set_span_attribute(
                span,
                SpanAttributes.GEN_AI_REQUEST_STRUCTURED_OUTPUT_SCHEMA,
                json.dumps(kwargs.get("response_schema")),
            )
        except Exception:
            pass

    if should_send_prompts():
        si = _system_instruction_from_kwargs(kwargs)
        if si is not None:
            try:
                parts = _system_instruction_to_parts(si, span)
                if parts:
                    _set_span_attribute(
                        span,
                        GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS,
                        json.dumps(parts),
                    )
            except Exception:
                pass

        tools = _tool_definitions_from_kwargs(kwargs)
        if tools:
            try:
                _set_span_attribute(
                    span,
                    GenAIAttributes.GEN_AI_TOOL_DEFINITIONS,
                    json.dumps(tools),
                )
            except Exception:
                pass


def _system_instruction_from_kwargs(kwargs):
    """Top-level kwarg or unified-client ``config.system_instruction``."""
    si = kwargs.get("system_instruction")
    if si is not None:
        return si
    config = kwargs.get("config")
    if config is not None:
        si = getattr(config, "system_instruction", None)
        if si is not None:
            return si
    return None


def _system_instruction_to_parts(si, span):
    """OTel: flat array of parts for gen_ai.system_instructions."""
    if isinstance(si, str):
        return [{"type": "text", "content": si}]
    if hasattr(si, "parts") and si.parts:
        out = []
        for idx, p in enumerate(si.parts):
            out.extend(_parts_from_genai_part_sync(p, span, idx))
        return out
    return [{"type": "text", "content": str(si)}]


def _tool_definitions_from_kwargs(kwargs):
    """Extract tool definitions from kwargs — source system representation per OTel spec."""
    tools = kwargs.get("tools")
    if tools is None:
        config = kwargs.get("config")
        if config is not None:
            tools = getattr(config, "tools", None)
    if not tools:
        return None
    defs = []
    for tool in tools:
        if hasattr(tool, "model_dump"):
            defs.append(tool.model_dump(exclude_none=True, mode="json"))
        elif isinstance(tool, dict):
            defs.append(tool)
        elif callable(tool):
            defs.append({"name": getattr(tool, "__name__", str(tool))})
    return defs or None



def _serialize_response_part(part):
    """Serialize a single response Part to a dict for gen_ai.output.messages."""
    if hasattr(part, "text") and part.text:
        return {"type": "text", "content": part.text}
    if hasattr(part, "function_call") and part.function_call:
        fc = part.function_call
        return {
            "type": "function_call",
            "name": fc.name,
            "arguments": dict(fc.args) if hasattr(fc, "args") else {},
        }
    return {"type": "text", "content": str(part)}


@dont_throw
def set_response_attributes(span, response, llm_model, stream_last_chunk=None):
    if not span.is_recording():
        return
    if not should_send_prompts():
        return

    output_messages = []

    if hasattr(response, "usage_metadata"):
        # Non-streaming: iterate over candidates, serialize all part types, and
        # use each candidate's own finish_reason.
        _candidates = getattr(response, "candidates", None)
        if _candidates:
            for candidate in _candidates:
                _finish_reason = None
                fr = getattr(candidate, "finish_reason", None)
                if fr:
                    _finish_reason = _map_gemini_finish_reason(fr) or None
                parts = []
                content = getattr(candidate, "content", None)
                if content and hasattr(content, "parts"):
                    for part in content.parts:
                        parts.append(_serialize_response_part(part))
                if not parts:
                    # Fallback: try response.text for simple single-part responses
                    try:
                        text = response.text
                        if text:
                            parts = [{"type": "text", "content": text}]
                    except Exception:
                        pass
                if parts:
                    msg = {"role": "assistant", "parts": parts}
                    if _finish_reason:
                        msg["finish_reason"] = _finish_reason
                    output_messages.append(msg)
        else:
            # No candidates field: fall back to response.text
            try:
                if isinstance(response.text, list):
                    for item in response:
                        output_messages.append({
                            "role": "assistant",
                            "parts": [{"type": "text", "content": item.text}],
                        })
                elif isinstance(response.text, str):
                    output_messages.append({
                        "role": "assistant",
                        "parts": [{"type": "text", "content": response.text}],
                    })
            except Exception:
                pass
    else:
        # Streaming path: omit finish_reason (not reliably available per-chunk)
        if isinstance(response, list):
            for index, item in enumerate(response):
                output_messages.append({
                    "role": "assistant",
                    "parts": [{"type": "text", "content": item}],
                })
        elif isinstance(response, str):
            output_messages.append({
                "role": "assistant",
                "parts": [{"type": "text", "content": response}],
            })

    if output_messages:
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
            json.dumps(output_messages),
        )


def set_model_response_attributes(
    span, response, llm_model, token_histogram, stream_finish_reasons=None
):
    if not span.is_recording():
        return
    _set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, llm_model)
    if hasattr(response, "usage_metadata"):
        _set_span_attribute(
            span,
            SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS,
            um.total_token_count,
        )
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,
            um.candidates_token_count,
        )
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS,
            um.prompt_token_count,
        )
    if token_histogram and hasattr(response, "usage_metadata"):
        token_histogram.record(
            um.prompt_token_count,
            attributes={
                GenAIAttributes.GEN_AI_PROVIDER_NAME: _GCP_GEN_AI,
                GenAIAttributes.GEN_AI_OPERATION_NAME: _GEN_CONTENT,
                GenAIAttributes.GEN_AI_REQUEST_MODEL: llm_model,
                GenAIAttributes.GEN_AI_TOKEN_TYPE: "input",
                GenAIAttributes.GEN_AI_RESPONSE_MODEL: llm_model,
            },
        )
        token_histogram.record(
            response.usage_metadata.candidates_token_count,
            attributes={
                GenAIAttributes.GEN_AI_PROVIDER_NAME: "Google",
                GenAIAttributes.GEN_AI_TOKEN_TYPE: "output",
                GenAIAttributes.GEN_AI_RESPONSE_MODEL: llm_model,
            },
        )
    span.set_status(Status(StatusCode.OK))
