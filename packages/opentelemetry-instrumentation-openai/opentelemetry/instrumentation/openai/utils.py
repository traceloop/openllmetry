from importlib.metadata import version
from contextlib import asynccontextmanager
import os


def is_openai_v1():
    return version("openai") >= "1.0.0"


def is_metrics_enabled() -> bool:
    return (os.getenv("TRACELOOP_METRICS_ENABLED") or "true").lower() == "true"


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
