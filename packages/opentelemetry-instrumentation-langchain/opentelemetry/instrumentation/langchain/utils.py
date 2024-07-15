import dataclasses
import json
import logging
import os
import traceback
from opentelemetry import context as context_api
from opentelemetry.instrumentation.langchain.config import Config

logger = logging.getLogger(__name__)


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

        return super().default(o)


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


def should_send_prompts():
    return (
        os.getenv("TRACELOOP_TRACE_CONTENT") or "true"
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


# tiktoken encodings map for different model, key is model_name, value is tiktoken encoding
tiktoken_encodings = {}


def get_token_count_from_string(string: str, model_name: str):
    if Config.enrich_token_usage:
        return None

    import tiktoken

    if tiktoken_encodings.get(model_name) is None:
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError as ex:
            # no such model_name in tiktoken
            logger.warning(
                f"Failed to get tiktoken encoding for model_name {model_name}, error: {str(ex)}"
            )
            return None

        tiktoken_encodings[model_name] = encoding
    else:
        encoding = tiktoken_encodings.get(model_name)

    token_count = len(encoding.encode(string))
    return token_count
