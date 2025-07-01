import dataclasses
import datetime
import importlib.util
import json
import logging
import os
import traceback

from opentelemetry import context as context_api
from opentelemetry._events import EventLogger
from opentelemetry.instrumentation.langchain.config import Config
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from pydantic import BaseModel

TRACELOOP_TRACE_CONTENT = "TRACELOOP_TRACE_CONTENT"

EVENT_ATTRIBUTES = {GenAIAttributes.GEN_AI_SYSTEM: "langchain"}


class CallbackFilteredJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, dict):
            if "callbacks" in o:
                del o["callbacks"]
                return o

        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)

        if hasattr(o, "to_json"):
            return o.to_json()

        if isinstance(o, BaseModel) and hasattr(o, "model_dump_json"):
            return o.model_dump_json()

        if isinstance(o, datetime.datetime):
            return o.isoformat()

        try:
            return str(o)
        except Exception:
            logger = logging.getLogger(__name__)
            logger.debug("Failed to serialize object of type: %s", type(o).__name__)
            return ""


def should_send_prompts():
    return (
        os.getenv(TRACELOOP_TRACE_CONTENT) or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")


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


def should_emit_events() -> bool:
    """
    Checks if the instrumentation isn't using the legacy attributes
    and if the event logger is not None.
    """
    return not Config.use_legacy_attributes and isinstance(
        Config.event_logger, EventLogger
    )


def is_package_available(package_name):
    return importlib.util.find_spec(package_name) is not None
