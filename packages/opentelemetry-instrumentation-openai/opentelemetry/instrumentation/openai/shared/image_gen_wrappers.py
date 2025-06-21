import json
import time
import asyncio
import base64
import mimetypes

from opentelemetry import context as context_api
from opentelemetry.instrumentation.openai import is_openai_v1
from opentelemetry.instrumentation.openai.shared import (
    _get_openai_base_url,
    metric_shared_attributes,
    model_as_dict,
    _set_span_attribute,
    _set_client_attributes,
    _set_request_attributes,
    set_usage_attributes,
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


@dont_throw
def _handle_image_gen_request(span, kwargs, instance):
    _set_request_attributes(span, kwargs)
    _set_client_attributes(span, instance)

    if should_send_prompts():
        _set_image_gen_prompt(span, kwargs.get("prompt"), kwargs.get("image"))


@dont_throw
async def _ahandle_image_gen_request(span, kwargs, instance):
    _set_request_attributes(span, kwargs)
    _set_client_attributes(span, instance)

    if should_send_prompts():
        await _aset_image_gen_prompt(span, kwargs.get("prompt"), kwargs.get("image"))


def _set_image_gen_prompt(span, prompt, image=None):
    if not span.is_recording():
        return

    if prompt is not None:
        _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.content", prompt)

    if image is not None and Config.upload_base64_image:
        run_async(_process_image_files(span, image))


async def _aset_image_gen_prompt(span, prompt, image=None):
    if not span.is_recording():
        return

    prompts = 0
    if prompt is not None:
        _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.content", prompt)
        prompts += 1

    if image is not None and Config.upload_base64_image:
        if isinstance(image, list):
            for i, img in enumerate(image):
                processed_img = await _process_single_image(img, span, i)
                span.set_attribute(
                    f"{SpanAttributes.LLM_PROMPTS}.{prompts+i}.content",
                    json.dumps(processed_img),
                )

        else:
            processed_img = await _process_single_image(image, span, 0)
            span.set_attribute(
                f"{SpanAttributes.LLM_PROMPTS}.{prompts+1}.content",
                json.dumps(processed_img),
            )


async def _process_image_files(span, image):
    if not span.is_recording():
        return

    if isinstance(image, list):
        processed_images = []
        for i, img in enumerate(image):
            processed_img = await _process_single_image(img, span, i)
            if processed_img:
                processed_images.append(processed_img)

        if processed_images:
            span.set_attribute(
                f"{SpanAttributes.LLM_PROMPTS}.0.images", json.dumps(processed_images)
            )
    else:
        processed_img = await _process_single_image(image, span, 0)
        if processed_img:
            span.set_attribute(
                f"{SpanAttributes.LLM_PROMPTS}.0.images", json.dumps([processed_img])
            )


async def _process_single_image(img, span, index):
    if hasattr(img, "read"):
        try:
            img.seek(0)
            image_data = img.read()
            img.seek(0)

            if hasattr(img, "name"):
                filename = img.name
                mime_type, _ = mimetypes.guess_type(filename)
                if mime_type:
                    image_format = mime_type.split("/")[1]
                else:
                    image_format = "bin"
            else:
                image_format = "bin"
                filename = f"input_image_{index}.{image_format}"

            base64_data = base64.b64encode(image_data).decode("utf-8")

            url = await Config.upload_base64_image(
                span.context.trace_id, span.context.span_id, filename, base64_data
            )

            return {"type": "image_url", "image_url": {"url": url}}
        except Exception:
            return None

    return None


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

    set_usage_attributes(span, response_dict.get("usage", {}))

    if should_send_prompts():
        if Config.upload_base64_image and response.data:
            url = await Config.upload_base64_image(
                span.context.trace_id,
                span.context.span_id,
                "generated_image.png",
                response.data[0].b64_json,
            )
            span.set_attribute(
                f"{SpanAttributes.LLM_COMPLETIONS}.0.content",
                json.dumps([{"type": "image_url", "image_url": {"url": url}}]),
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
    span_name: str,
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
        span_name,
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
    span_name: str,
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
        span_name,
        kind=SpanKind.CLIENT,
    )

    await _ahandle_image_gen_request(span, kwargs, instance)

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
