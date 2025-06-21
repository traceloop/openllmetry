import json
import time
import asyncio

from opentelemetry import context as context_api
from opentelemetry.instrumentation.openai import is_openai_v1
from opentelemetry.instrumentation.openai.shared import (
    _get_openai_base_url,
    metric_shared_attributes,
    model_as_dict,
    _set_span_attribute,
    _set_client_attributes,
    _set_request_attributes,
    _set_response_attributes,
)
from opentelemetry.instrumentation.openai.shared.config import Config
from opentelemetry.instrumentation.openai.utils import (
    _with_image_gen_wrapper,
    dont_throw,
    should_send_prompts,
    run_async,
)
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.metrics import Counter, Histogram
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)
from opentelemetry.trace import SpanKind, Tracer
from opentelemetry.trace.status import Status, StatusCode


SPAN_NAME = "openai.generate"


@dont_throw
def _handle_image_gen_request(span, kwargs, instance):
    _set_request_attributes(span, kwargs)
    _set_client_attributes(span, instance)

    if should_send_prompts():
        _set_image_gen_prompt(span, kwargs.get("prompt"))


def _set_image_gen_prompt(span, prompt):
    if not span.is_recording() or prompt is None:
        return

    _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.content", prompt)


@dont_throw
async def _handle_image_gen_response(
    response,
    span,
    instance=None,
    duration_histogram=None,
    duration=None,
):
    if is_openai_v1():
        response_dict = model_as_dict(response)
    else:
        response_dict = response

    shared_attributes = metric_shared_attributes(
        response_model=response_dict.get("model") or "unknown",
        operation="image_gen",
        server_address=_get_openai_base_url(instance),
    )

    if duration and isinstance(duration, (float, int)) and duration_histogram:
        duration_histogram.record(duration, attributes=shared_attributes)

    # _set_response_attributes(span, response_dict)

    if should_send_prompts():
        if Config.upload_base64_image and response.data:
            url = await Config.upload_base64_image(
                span.context.trace_id,
                span.context.span_id,
                "generated_image",
                response.data[0].b64_json,
            )
            span.set_attribute(
                f"{SpanAttributes.LLM_COMPLETIONS}.0.content",
                json.dumps({"type": "image_url", "image_url": {"url": url}}),
            )
            span.set_attribute(
                f"{SpanAttributes.LLM_COMPLETIONS}.0.role",
                "assistant",
            )


@_with_image_gen_wrapper
def image_gen_wrapper(
    tracer: Tracer,
    duration_histogram: Histogram,
    exception_counter: Counter,
    wrapped,
    instance,
    args,
    kwargs,
):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return wrapped(*args, **kwargs)

    span = tracer.start_span(
        SPAN_NAME,
        kind=SpanKind.CLIENT,
    )

    _handle_image_gen_request(span, kwargs, instance)

    try:
        start_time = time.time()
        response = wrapped(*args, **kwargs)
        end_time = time.time()
    except Exception as e:  # pylint: disable=broad-except
        end_time = time.time()
        duration = end_time - start_time if "start_time" in locals() else 0

        attributes = {
            "error.type": e.__class__.__name__,
        }

        if duration > 0 and duration_histogram:
            duration_histogram.record(duration, attributes=attributes)
        if exception_counter:
            exception_counter.add(1, attributes=attributes)

        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.end()

        raise e

    duration = end_time - start_time

    run_async(
        _handle_image_gen_response(
            response,
            span,
            instance,
            duration_histogram,
            duration,
        )
    )

    span.end()

    return response


@_with_image_gen_wrapper
async def aimage_gen_wrapper(
    tracer: Tracer,
    duration_histogram: Histogram,
    exception_counter: Counter,
    wrapped,
    instance,
    args,
    kwargs,
):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return await wrapped(*args, **kwargs)

    span = tracer.start_span(
        SPAN_NAME,
        kind=SpanKind.CLIENT,
    )

    _handle_image_gen_request(span, kwargs, instance)

    try:
        start_time = time.time()
        response = await wrapped(*args, **kwargs)
        end_time = time.time()
    except Exception as e:  # pylint: disable=broad-except
        end_time = time.time()
        duration = end_time - start_time if "start_time" in locals() else 0

        attributes = {
            "error.type": e.__class__.__name__,
        }

        if duration > 0 and duration_histogram:
            duration_histogram.record(duration, attributes=attributes)
        if exception_counter:
            exception_counter.add(1, attributes=attributes)

        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.end()

        raise e

    duration = end_time - start_time

    await _handle_image_gen_response(
        response,
        span,
        instance,
        duration_histogram,
        duration,
    )

    span.end()

    return response


@dont_throw
async def _process_generated_image(image_data, trace_id, span_id, image_index):
    if not Config.upload_base64_image:
        return image_data

    if not hasattr(image_data, "b64_json") or not image_data.b64_json:
        return image_data

    image_name = f"generated_image_{image_index}.png"
    url = await Config.upload_base64_image(
        trace_id, span_id, image_name, image_data.b64_json
    )

    return {"url": url, "b64_json": image_data.b64_json}
