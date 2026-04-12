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
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import (
    SpanAttributes,
)
from opentelemetry.trace.status import Status, StatusCode

_GCP_GEN_AI = GenAIAttributes.GenAiProviderNameValues.GCP_GEN_AI.value
_GEN_CONTENT = GenAIAttributes.GenAiOperationNameValues.GENERATE_CONTENT.value

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
        binary_data = item.inline_data.data
        base64_string = base64.b64encode(binary_data).decode("utf-8")
        url = await Config.upload_base64_image(
            str(trace_id), str(span_id), image_name, base64_string
        )
        return {"type": "image_url", "image_url": {"url": url}}
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
        binary_data = item.inline_data.data
        base64_string = base64.b64encode(binary_data).decode("utf-8")
        url = None

        async def upload_task():
            nonlocal url
            url = await Config.upload_base64_image(
                str(trace_id), str(span_id), image_name, base64_string
            )

        run_async(upload_task())

        return {"type": "image_url", "image_url": {"url": url}}
    except Exception as e:
        logger.warning(f"Failed to process image part sync: {e}")
        return None


def _extract_non_image_parts(part):
    """Extract all non-image OTel parts from a GenAI Part object (shared by sync/async)."""
    out = []
    if getattr(part, "thought", False) and getattr(part, "text", None):
        out.append({"type": "reasoning", "content": part.text})
    elif getattr(part, "text", None):
        out.append({"type": "text", "content": part.text})
    fc = getattr(part, "function_call", None)
    if fc is not None:
        fid = getattr(fc, "id", None) or ""
        fname = getattr(fc, "name", None) or ""
        out.append(
            {
                "type": "tool_call",
                "id": fid,
                "name": fname,
                "arguments": _parse_function_call_arguments(getattr(fc, "args", None)),
            }
        )
    fr = getattr(part, "function_response", None)
    if fr is not None:
        fid = getattr(fr, "id", None) or ""
        out.append(
            {
                "type": "tool_call_response",
                "id": fid,
                "response": _function_response_to_str(getattr(fr, "response", None)),
            }
        )
    return out


def _fallback_part(part):
    """Fallback when no structured parts were extracted."""
    if getattr(part, "executable_code", None):
        return {"type": "text", "content": str(part.executable_code)}
    if getattr(part, "code_execution_result", None):
        return {"type": "text", "content": str(part.code_execution_result)}
    return {"type": "text", "content": str(part)}


def _parts_from_genai_part_sync(part, span, part_index):
    out = _extract_non_image_parts(part)
    img = _otel_image_part_from_genai_part(part, span, part_index, sync=True)
    if img:
        out.append(img)
    return out or [_fallback_part(part)]


async def _parts_from_genai_part_async(part, span, part_index):
    out = _extract_non_image_parts(part)
    if _is_image_part(part):
        img = await _otel_image_part_from_genai_part_async(part, span, part_index)
        if img:
            out.append(img)
    return out or [_fallback_part(part)]


def _map_dict_part(p):
    """Map a dict-based part to OTel part schema, preserving semantics."""
    text = p.get("text")
    if text is not None:
        return {"type": "text", "content": text}
    fc = p.get("functionCall")
    if fc is not None:
        return {
            "type": "tool_call",
            "id": fc.get("id", ""),
            "name": fc.get("name", ""),
            "arguments": _parse_function_call_arguments(fc.get("args")),
        }
    fr = p.get("functionResponse")
    if fr is not None:
        return {
            "type": "tool_call_response",
            "id": fr.get("id", ""),
            "response": _function_response_to_str(fr.get("response")),
        }
    inline = p.get("inlineData")
    if inline is not None:
        mime = inline.get("mimeType", "application/octet-stream")
        data = inline.get("data", "")
        modality = next((m for m in ("image", "video", "audio") if f"{m}/" in mime), "data")
        return {"type": "blob", "modality": modality, "mime_type": mime, "content": data}
    return {"type": "text", "content": str(p)}


async def _process_content_item(content_item, span):
    parts_acc = []
    if isinstance(content_item, dict):
        for p in content_item.get("parts", []):
            if isinstance(p, dict):
                parts_acc.append(_map_dict_part(p))
            else:
                parts_acc.extend(await _parts_from_genai_part_async(p, span, 0))
    elif hasattr(content_item, "parts"):
        for part_index, part in enumerate(content_item.parts):
            parts_acc.extend(await _parts_from_genai_part_async(part, span, part_index))
    elif isinstance(content_item, str):
        parts_acc.append({"type": "text", "content": content_item})
    elif _is_image_part(content_item):
        img = await _otel_image_part_from_genai_part_async(content_item, span, 0)
        if img:
            parts_acc.append(img)
    else:
        parts_acc.append({"type": "text", "content": str(content_item)})
    return parts_acc


async def _process_argument(argument, span):
    if isinstance(argument, str):
        return [{"type": "text", "content": argument}]
    if isinstance(argument, list):
        parts_acc = []
        for sub_index, sub_item in enumerate(argument):
            if isinstance(sub_item, str):
                parts_acc.append({"type": "text", "content": sub_item})
            elif _is_image_part(sub_item):
                img = await _otel_image_part_from_genai_part_async(sub_item, span, sub_index)
                if img:
                    parts_acc.append(img)
            else:
                parts_acc.append({"type": "text", "content": str(sub_item)})
        return parts_acc
    if _is_image_part(argument):
        img = await _otel_image_part_from_genai_part_async(argument, span, 0)
        return [img] if img else []
    return [{"type": "text", "content": str(argument)}]


def _process_content_item_sync(content_item, span):
    parts_acc = []
    if isinstance(content_item, dict):
        for p in content_item.get("parts", []):
            if isinstance(p, dict):
                parts_acc.append(_map_dict_part(p))
            else:
                parts_acc.extend(_parts_from_genai_part_sync(p, span, 0))
    elif hasattr(content_item, "parts"):
        for part_index, part in enumerate(content_item.parts):
            parts_acc.extend(_parts_from_genai_part_sync(part, span, part_index))
    elif isinstance(content_item, str):
        parts_acc.append({"type": "text", "content": content_item})
    elif _is_image_part(content_item):
        img = _otel_image_part_from_genai_part(content_item, span, 0, sync=True)
        if img:
            parts_acc.append(img)
    else:
        parts_acc.append({"type": "text", "content": str(content_item)})
    return parts_acc


def _process_argument_sync(argument, span):
    if isinstance(argument, str):
        return [{"type": "text", "content": argument}]
    if isinstance(argument, list):
        parts_acc = []
        for sub_index, sub_item in enumerate(argument):
            if isinstance(sub_item, str):
                parts_acc.append({"type": "text", "content": sub_item})
            elif _is_image_part(sub_item):
                img = _otel_image_part_from_genai_part(sub_item, span, sub_index, sync=True)
                if img:
                    parts_acc.append(img)
            else:
                parts_acc.append({"type": "text", "content": str(sub_item)})
        return parts_acc
    if _is_image_part(argument):
        img = _otel_image_part_from_genai_part(argument, span, 0, sync=True)
        return [img] if img else []
    return [{"type": "text", "content": str(argument)}]


def _collect_finish_reasons_from_response(response):
    reasons = []
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        mapped = _map_gemini_finish_reason(getattr(cand, "finish_reason", None))
        reasons.append(mapped)
    return reasons



def _output_messages_from_generate_response(span, response):
    messages = []
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        parts = []
        content = getattr(cand, "content", None)
        if content and getattr(content, "parts", None):
            for idx, p in enumerate(content.parts):
                parts.extend(_parts_from_genai_part_sync(p, span, idx))
        fr = _map_gemini_finish_reason(getattr(cand, "finish_reason", None))
        role = "assistant"
        if content and getattr(content, "role", None):
            role = _normalize_message_role(content.role)
        msg = {"role": role, "parts": parts, "finish_reason": fr}
        messages.append(msg)
    if not messages and hasattr(response, "text") and response.text:
        text = response.text
        if isinstance(text, str) and text:
            msg = {"role": "assistant", "parts": [{"type": "text", "content": text}], "finish_reason": ""}
            messages.append(msg)
    return messages


@dont_throw
async def set_input_attributes(span, args, kwargs, llm_model):
    if not span.is_recording():
        return
    if not should_send_prompts():
        return

    messages = []
    if "contents" in kwargs:
        contents = kwargs["contents"]
        if isinstance(contents, str):
            messages.append(
                {
                    "role": "user",
                    "parts": [{"type": "text", "content": contents}],
                }
            )
        elif isinstance(contents, list):
            for content_item in contents:
                parts = await _process_content_item(content_item, span)
                if parts:
                    messages.append(
                        {
                            "role": _normalize_message_role(
                                content_item.get("role", "user") if isinstance(content_item, dict)
                                else getattr(content_item, "role", "user")
                            ),
                            "parts": parts,
                        }
                    )
    elif args and len(args) > 0:
        for argument in args:
            parts = await _process_argument(argument, span)
            if parts:
                messages.append({"role": "user", "parts": parts})
    elif "prompt" in kwargs:
        messages.append(
            {
                "role": "user",
                "parts": [{"type": "text", "content": kwargs["prompt"]}],
            }
        )

    if messages:
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_INPUT_MESSAGES,
            json.dumps(messages),
        )


@dont_throw
def set_input_attributes_sync(span, args, kwargs, llm_model):
    if not span.is_recording():
        return
    if not should_send_prompts():
        return

    messages = []
    if "contents" in kwargs:
        contents = kwargs["contents"]
        if isinstance(contents, str):
            messages.append(
                {
                    "role": "user",
                    "parts": [{"type": "text", "content": contents}],
                }
            )
        elif isinstance(contents, list):
            for content in contents:
                parts = _process_content_item_sync(content, span)
                if parts:
                    messages.append(
                        {
                            "role": _normalize_message_role(
                                content.get("role", "user") if isinstance(content, dict)
                                else getattr(content, "role", "user")
                            ),
                            "parts": parts,
                        }
                    )
    elif args and len(args) > 0:
        for arg in args:
            parts = _process_argument_sync(arg, span)
            if parts:
                messages.append({"role": "user", "parts": parts})
    elif "prompt" in kwargs:
        messages.append(
            {
                "role": "user",
                "parts": [{"type": "text", "content": kwargs["prompt"]}],
            }
        )

    if messages:
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_INPUT_MESSAGES,
            json.dumps(messages),
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
        GenAIAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY,
        kwargs.get("presence_penalty"),
    )
    _set_span_attribute(
        span,
        GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY,
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


@dont_throw
def set_response_attributes(span, response, llm_model, stream_last_chunk=None):
    if not span.is_recording():
        return
    if not should_send_prompts():
        return

    if isinstance(response, str):
        fr = ""
        if stream_last_chunk is not None:
            reasons = _collect_finish_reasons_from_response(stream_last_chunk)
            fr = reasons[0] if reasons else ""
        msg = {"role": "assistant", "parts": []}
        if response:
            msg["parts"].append({"type": "text", "content": response})
        msg["finish_reason"] = fr
        if not msg["parts"] and not fr:
            return
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
            json.dumps([msg]),
        )
        return

    if isinstance(response, list):
        messages = []
        for item in response:
            if isinstance(item, str):
                messages.append(
                    {
                        "role": "assistant",
                        "parts": [{"type": "text", "content": item}],
                        "finish_reason": "",
                    }
                )
            else:
                messages.extend(_output_messages_from_generate_response(span, item))
    else:
        messages = _output_messages_from_generate_response(span, response)

    if messages:
        _set_span_attribute(
            span,
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
            json.dumps(messages),
        )


def set_model_response_attributes(
    span, response, llm_model, token_histogram, stream_finish_reasons=None
):
    if not span.is_recording():
        return

    _set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, llm_model)

    rid = getattr(response, "response_id", None)
    if rid:
        _set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_ID, rid)

    if stream_finish_reasons is not None:
        reasons = stream_finish_reasons
    else:
        reasons = _collect_finish_reasons_from_response(response)
    if reasons and any(reason != "" for reason in reasons):
        _set_span_attribute(
            span, GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS, reasons
        )

    um = getattr(response, "usage_metadata", None)
    if um is not None:
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

    if token_histogram and um is not None:
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
            um.candidates_token_count,
            attributes={
                GenAIAttributes.GEN_AI_PROVIDER_NAME: _GCP_GEN_AI,
                GenAIAttributes.GEN_AI_OPERATION_NAME: _GEN_CONTENT,
                GenAIAttributes.GEN_AI_REQUEST_MODEL: llm_model,
                GenAIAttributes.GEN_AI_TOKEN_TYPE: "output",
                GenAIAttributes.GEN_AI_RESPONSE_MODEL: llm_model,
            },
        )

    span.set_status(Status(StatusCode.OK))
