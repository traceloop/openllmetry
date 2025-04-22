import logging
import traceback
import json

from opentelemetry.instrumentation.google_genai.config import Config
from google.genai import types
from google.genai._common import BaseModel
import pydantic
from opentelemetry.trace import Span
from typing import Any, Optional, Union


def set_span_attribute(span: Span, name: str, value: str):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


def dont_throw(func):
    """
    A decorator that wraps the passed in function and logs exceptions instead of throwing them.

    @param func: The function to wrap
    @return: The wrapper function
    """
    # Obtain a logger specific to the function's module
    logger = logging.getLogger(func.__module__)

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.debug(
                "OpenLLMetry failed to trace in %s, error: %s",
                func.__name__,
                traceback.format_exc(),
            )
            if Config.exception_logger:
                Config.exception_logger(e)

    return wrapper


def to_dict(obj: Union[BaseModel, pydantic.BaseModel, dict]) -> dict[str, Any]:
    try:
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        elif isinstance(obj, pydantic.BaseModel):
            return obj.model_dump()
        elif isinstance(obj, dict):
            return obj
        else:
            return dict(obj)
    except Exception:
        return dict(obj)


def process_content_union(
    content: Union[types.ContentUnion, types.ContentUnionDict],
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    message_index: int = 0,
) -> Optional[str]:
    parts = _process_content_union(content, trace_id, span_id, message_index)
    if parts is None:
        return None
    if isinstance(parts, str):
        return parts
    elif isinstance(parts, list):
        if len(parts) == 1 and isinstance(parts[0], str):
            return parts[0]
        return json.dumps([
            {
                "type": "text",
                "text": part
            }
            if isinstance(part, str)
            else part
            for part in parts
        ])
    else:
        return None


def _process_content_union(
    content: Union[types.ContentUnion, types.ContentUnionDict],
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    message_index: int = 0,
) -> Union[str, list[str], None]:
    if isinstance(content, types.Content):
        parts = to_dict(content).get("parts", [])
        return [_process_part(part) for part in parts]
    elif isinstance(content, list):
        return [_process_part_union(item) for item in content]
    elif isinstance(content, (types.Part, types.File, str)):
        return _process_part_union(content)
    elif isinstance(content, dict):
        if "parts" in content:
            return [
                _process_part_union(item, trace_id, span_id, message_index, content_index)
                for content_index, item
                in enumerate(content.get("parts", []))
            ]
        else:
            # Assume it's PartDict
            return _process_part_union(content, trace_id, span_id, message_index)
    else:
        return None


def _process_part_union(
    content: Union[types.PartDict, types.File, types.Part, str],
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    message_index: int = 0,
    content_index: int = 0,
) -> Optional[str]:
    if isinstance(content, str):
        return content
    elif isinstance(content, types.File):
        content_dict = to_dict(content)
        name = content_dict.get("name") or content_dict.get("display_name") or content_dict.get("uri")
        return f"files/{name}"
    elif isinstance(content, (types.Part, dict)):
        return _process_part(content, trace_id, span_id, message_index, content_index)
    else:
        return None


def _process_part(
    content: types.Part,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    message_index: int = 0,
    content_index: int = 0,
) -> Optional[str]:
    part_dict = to_dict(content)
    if part_dict.get("text") is not None:
        return part_dict.get("text")
    elif part_dict.get("inline_data"):
        blob = to_dict(part_dict.get("inline_data"))
        if blob.get("mime_type").startswith("image/"):
            return _process_image_item(blob, trace_id, span_id, message_index, content_index)
        else:
            # currently, only images are supported
            return blob.get("mime_type") or "unknown_media"
    else:
        return None


def role_from_content_union(content: Union[types.ContentUnion, types.ContentUnionDict]) -> Optional[str]:
    if isinstance(content, types.Content):
        return to_dict(content).get("role")
    elif isinstance(content, list) and len(content) > 0:
        return role_from_content_union(content[0])
    else:
        return None


def with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


def _run_async(method):
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        return loop.run_until_complete(method)
    else:
        return asyncio.run(method)


def _process_image_item(blob: dict[str, Any], trace_id: str, span_id: str, message_index: int, content_index: int):

    if not Config.upload_base64_image:
        # Convert to openai format, so backends can handle it
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{blob.get('mime_type').split('/')[1]};base64,{blob.get('data')}",
            }
        } if Config.convert_image_to_openai_format else blob

    image_format = blob.get("mime_type").split("/")[1]
    image_name = f"message_{message_index}_content_{content_index}.{image_format}"
    base64_string = blob.get("data")
    url = _run_async(Config.upload_base64_image(trace_id, span_id, image_name, base64_string))

    return {"type": "image_url", "image_url": {"url": url}}
