from importlib.metadata import version
from contextlib import asynccontextmanager
import logging
import os

import openai
from opentelemetry.instrumentation.openai.shared.config import Config


def is_openai_v1():
    return version("openai") >= "1.0.0"


def is_azure_openai(instance):
    return is_openai_v1() and isinstance(
        instance._client, (openai.AsyncAzureOpenAI, openai.AzureOpenAI)
    )


def is_metrics_enabled() -> bool:
    return (os.getenv("TRACELOOP_METRICS_ENABLED") or "true").lower() == "true"


def should_record_stream_token_usage():
    return Config.enrich_token_usage


def _with_image_gen_metric_wrapper(func):
    def _with_metric(duration_histogram, exception_counter):
        def wrapper(wrapped, instance, args, kwargs):
            return func(
                duration_histogram, exception_counter, wrapped, instance, args, kwargs
            )

        return wrapper

    return _with_metric


def _with_embeddings_telemetry_wrapper(func):
    def _with_embeddings_telemetry(
        tracer,
        token_counter,
        vector_size_counter,
        duration_histogram,
        exception_counter,
    ):
        def wrapper(wrapped, instance, args, kwargs):
            return func(
                tracer,
                token_counter,
                vector_size_counter,
                duration_histogram,
                exception_counter,
                wrapped,
                instance,
                args,
                kwargs,
            )

        return wrapper

    return _with_embeddings_telemetry


def _with_chat_telemetry_wrapper(func):
    def _with_chat_telemetry(
        tracer,
        token_counter,
        choice_counter,
        duration_histogram,
        exception_counter,
        streaming_time_to_first_token,
        streaming_time_to_generate,
    ):
        def wrapper(wrapped, instance, args, kwargs):
            return func(
                tracer,
                token_counter,
                choice_counter,
                duration_histogram,
                exception_counter,
                streaming_time_to_first_token,
                streaming_time_to_generate,
                wrapped,
                instance,
                args,
                kwargs,
            )

        return wrapper

    return _with_chat_telemetry


def _with_tracer_wrapper(func):
    def _with_tracer(tracer):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


@asynccontextmanager
async def start_as_current_span_async(tracer, *args, **kwargs):
    with tracer.start_as_current_span(*args, **kwargs) as span:
        yield span


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
            logger.warning("Failed to execute %s, error: %s", func.__name__, str(e))
            if Config.exception_logger:
                Config.exception_logger(e)

    return wrapper
