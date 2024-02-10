from importlib.metadata import version
from contextlib import asynccontextmanager


def is_openai_v1():
    return version("openai") >= "1.0.0"


def _with_chat_metric_wrapper(func):
    def _with_metric(token_counter, choice_counter, duration_histogram):
        def wrapper(wrapped, instance, args, kwargs):
            return func(token_counter, choice_counter, duration_histogram, wrapped, instance, args, kwargs)

        return wrapper

    return _with_metric


def _with_embeddings_metric_wrapper(func):
    def _with_metric(token_counter, vector_size_counter, duration_histogram):
        def wrapper(wrapped, instance, args, kwargs):
            return func(token_counter, vector_size_counter, duration_histogram, wrapped, instance, args, kwargs)

        return wrapper

    return _with_metric


def _with_image_gen_metric_wrapper(func):
    def _with_metric(duration_histogram):
        def wrapper(wrapped, instance, args, kwargs):
            return func(duration_histogram, wrapped, instance, args, kwargs)

        return wrapper

    return _with_metric


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
