"""Shared utilities for MCP instrumentation."""

import asyncio
import json
import logging
import traceback


class Config:
    exception_logger = None


def dont_throw(func):
    """
    A decorator that wraps the passed in function and logs exceptions instead of throwing them.
    Works for both synchronous and asynchronous functions.
    """
    logger = logging.getLogger(func.__module__)

    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            _handle_exception(e, func, logger)

    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            _handle_exception(e, func, logger)

    def _handle_exception(e, func, logger):
        logger.debug(
            "OpenLLMetry failed to trace in %s, error: %s",
            func.__name__,
            traceback.format_exc(),
        )
        if Config.exception_logger:
            Config.exception_logger(e)

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


def serialize_mcp_result(result, json_encoder_cls=None, truncate_func=None):
    """
    Serialize MCP tool result to JSON string.

    Args:
        result: The result object to serialize
        json_encoder_cls: Optional JSON encoder class
        truncate_func: Optional function to truncate the output

    Returns:
        Serialized and optionally truncated string representation
    """
    if not result:
        return None

    def _serialize_object(obj):
        """Recursively serialize an object to a JSON-compatible format."""
        # Try direct JSON serialization first
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            pass

        # Handle Pydantic models
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()

        # Handle objects with __dict__
        if hasattr(obj, '__dict__'):
            result_dict = {}
            for key, value in obj.__dict__.items():
                if not key.startswith('_'):
                    result_dict[key] = _serialize_object(value)
            return result_dict

        # Handle lists/tuples
        if isinstance(obj, (list, tuple)):
            return [_serialize_object(item) for item in obj]

        # Fallback to string representation
        return str(obj)

    try:
        # Handle FastMCP result types - prioritize extracting content
        # If result is a list, serialize it directly
        if isinstance(result, list):
            serialized = _serialize_object(result)
        # If result has .content attribute, extract and serialize just the content
        elif hasattr(result, 'content') and result.content:
            serialized = _serialize_object(result.content)
        else:
            # For other objects, serialize the whole thing
            serialized = _serialize_object(result)

        json_output = json.dumps(serialized, cls=json_encoder_cls)
        return truncate_func(json_output) if truncate_func else json_output
    except (TypeError, ValueError):
        # Final fallback: return raw result as string
        return str(result)
