import logging
import traceback
from os import environ

from opentelemetry._events import EventLogger
from opentelemetry.instrumentation.google_generativeai.config import Config

OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT = (
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
)


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


def is_content_enabled() -> bool:
    capture_content = environ.get(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "false"
    )

    return capture_content.lower() == "true"


def should_emit_events() -> bool:
    """
    Checks if the instrumentation isn't using the legacy attributes
    and if the event logger is not None.
    """

    return not Config.use_legacy_attributes and isinstance(
        Config.event_logger, EventLogger
    )


def part_to_dict(part):
    response = {}

    if part.text:
        response["text"] = part.text
    if part.inline_data:
        response["inline_data"] = part.inline_data
    if part.function_call:
        response["function_call"] = part.function_call
    if part.function_response:
        response["function_response"] = part.function_response
    if part.file_data:
        response["file_data"] = part.file_data
    if part.executable_code:
        response["executable_code"] = part.executable_code
    if part.code_execution_result:
        response["code_execution_result"] = part.code_execution_result

    return response
