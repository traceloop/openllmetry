import time

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY

from opentelemetry.metrics import Histogram

from opentelemetry.instrumentation.openai import is_openai_v1
from opentelemetry.instrumentation.openai.shared import _get_openai_base_url, model_as_dict
from opentelemetry.instrumentation.openai.utils import _with_image_gen_metric_wrapper


@_with_image_gen_metric_wrapper
def image_gen_metrics_wrapper(duration_histogram: Histogram,
                              wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    try:
        # record time for duration
        start_time = time.time()
        response = wrapped(*args, **kwargs)
        end_time = time.time()
    except Exception as e:  # pylint: disable=broad-except
        end_time = time.time()
        duration = end_time - start_time if 'start_time' in locals() else 0
        attributes = {
            "error.type": e.__class__.__name__,
            "server.address": _get_openai_base_url(),
        }

        if duration > 0:
            duration_histogram.record(duration, attributes=attributes)

        raise e

    if is_openai_v1():
        response_dict = model_as_dict(response)
    else:
        response_dict = response

    shared_attributes = {
        # not provide response.model in ImagesResponse response, use model in request kwargs
        "llm.response.model": kwargs.get("model") or None,
        "server.address": _get_openai_base_url(),
    }

    duration = end_time - start_time
    duration_histogram.record(duration, attributes=shared_attributes)

    return response
