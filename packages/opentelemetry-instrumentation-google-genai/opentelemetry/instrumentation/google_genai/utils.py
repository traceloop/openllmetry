import logging
import traceback

from opentelemetry.instrumentation.google_genai.config import Config
from google.genai import types
from google.genai._common import BaseModel
import pydantic
from opentelemetry.trace import Span
from typing import Any, Optional, Union


JOIN_PARTS_SEPARATOR = " "


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


def text_from_content_union(content: Union[types.ContentUnion, types.ContentUnionDict]) -> Optional[str]:
    if isinstance(content, types.Content):
        parts = to_dict(content).get("parts", [])
        return JOIN_PARTS_SEPARATOR.join([_text_from_part(part) for part in parts])
    elif isinstance(content, list):
        return JOIN_PARTS_SEPARATOR.join([_text_from_part_union(item) for item in content])
    elif isinstance(content, (types.Part, types.File, str)):
        return _text_from_part_union(content)
    elif isinstance(content, dict):
        if "parts" in content:
            return JOIN_PARTS_SEPARATOR.join([_text_from_part_union(item) for item in content.get("parts", [])])
        else:
            # Assume it's PartDict
            return _text_from_part_union(content)
    else:
        return None


def _text_from_part_union(content: Union[types.PartDict, types.File, types.Part, str]) -> Optional[str]:
    if isinstance(content, str):
        return content
    elif isinstance(content, types.File):
        content_dict = to_dict(content)
        name = content_dict.get("name") or content_dict.get("display_name") or content_dict.get("uri")
        return f"files/{name}"
    elif isinstance(content, (types.Part, dict)):
        return _text_from_part(content)
    else:
        return None


def _text_from_part(content: types.Part) -> Optional[str]:
    return to_dict(content).get("text")


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
