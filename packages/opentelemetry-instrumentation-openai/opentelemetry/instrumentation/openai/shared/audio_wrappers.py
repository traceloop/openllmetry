import logging
import time

from opentelemetry import context as context_api
from opentelemetry.instrumentation.openai.shared import (
    _set_client_attributes,
    _set_request_attributes,
    _set_response_attributes,
    _set_span_attribute,
    metric_shared_attributes,
    model_as_dict,
)
from opentelemetry.instrumentation.openai.utils import (
    _with_audio_telemetry_wrapper,
    dont_throw,
    is_openai_v1,
    start_as_current_span_async,
)
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.metrics import Counter, Histogram
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)
from opentelemetry.trace import SpanKind, Status, StatusCode

SPAN_NAME = "openai.audio.transcriptions"

logger = logging.getLogger(__name__)


def _get_audio_duration(file):
    """
    Extract audio duration from file object.
    Returns duration in seconds, or None if unable to determine.
    """
    try:
        # Try to get duration from common audio libraries
        # First check if it's a file-like object with a name attribute
        if hasattr(file, "name"):
            file_path = file.name
        elif isinstance(file, (str, bytes)):
            # If it's a path string or bytes
            return None
        else:
            # If it's a file-like object without name, we can't easily determine duration
            return None

        # Try mutagen (supports many formats)
        try:
            from mutagen import File as MutagenFile

            audio = MutagenFile(file_path)
            if audio and hasattr(audio.info, "length"):
                return audio.info.length
        except (ImportError, Exception):
            pass

    except Exception as e:
        logger.debug(f"Unable to extract audio duration: {e}")

    return None


@_with_audio_telemetry_wrapper
def transcription_wrapper(
    tracer,
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

    with tracer.start_as_current_span(
        name=SPAN_NAME,
        kind=SpanKind.CLIENT,
    ) as span:
        _handle_request(span, kwargs, instance)

        try:
            # record time for duration
            start_time = time.time()
            response = wrapped(*args, **kwargs)
            end_time = time.time()
        except Exception as e:  # pylint: disable=broad-except
            end_time = time.time()
            duration = end_time - start_time if "start_time" in locals() else 0
            attributes = {
                "error.type": e.__class__.__name__,
            }

            # if there are legal duration, record it
            if duration > 0 and duration_histogram:
                duration_histogram.record(duration, attributes=attributes)
            if exception_counter:
                exception_counter.add(1, attributes=attributes)

            span.set_attribute(ERROR_TYPE, e.__class__.__name__)
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.end()

            raise

        duration = end_time - start_time

        _handle_response(
            response,
            span,
            instance,
            duration_histogram,
            duration,
        )

        return response


@_with_audio_telemetry_wrapper
async def atranscription_wrapper(
    tracer,
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

    async with start_as_current_span_async(
        tracer=tracer,
        name=SPAN_NAME,
        kind=SpanKind.CLIENT,
    ) as span:
        _handle_request(span, kwargs, instance)

        try:
            # record time for duration
            start_time = time.time()
            response = await wrapped(*args, **kwargs)
            end_time = time.time()
        except Exception as e:  # pylint: disable=broad-except
            end_time = time.time()
            duration = end_time - start_time if "start_time" in locals() else 0
            attributes = {
                "error.type": e.__class__.__name__,
            }

            # if there are legal duration, record it
            if duration > 0 and duration_histogram:
                duration_histogram.record(duration, attributes=attributes)
            if exception_counter:
                exception_counter.add(1, attributes=attributes)

            span.set_attribute(ERROR_TYPE, e.__class__.__name__)
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.end()

            raise

        duration = end_time - start_time

        _handle_response(
            response,
            span,
            instance,
            duration_histogram,
            duration,
        )

        return response


@dont_throw
def _handle_request(span, kwargs, instance):
    _set_request_attributes(span, kwargs, instance)
    _set_client_attributes(span, instance)

    # Extract and set audio duration
    file_param = kwargs.get("file")
    if file_param:
        audio_duration = _get_audio_duration(file_param)
        if audio_duration is not None:
            # _set_span_attribute(
            #     span, SpanAttributes.LLM_OPENAI_AUDIO_INPUT_DURATION_SECONDS, audio_duration
            # )
            # TODO(Ata): come back here later when semconv is published
            _set_span_attribute(
                span, 'gen_ai.openai.audio.input.duration_seconds', audio_duration
            )
        else:
            print("REMOVE ME : ATA-DBG : COULD NOT READ AUDIO FILE WITH MUTAGEN")


@dont_throw
def _handle_response(
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

    # metrics record
    _set_transcription_metrics(
        instance,
        duration_histogram,
        response_dict,
        duration,
    )

    # span attributes
    _set_response_attributes(span, response_dict)


def _set_transcription_metrics(
    instance,
    duration_histogram,
    response_dict,
    duration,
):
    from opentelemetry.instrumentation.openai.shared import _get_openai_base_url

    shared_attributes = metric_shared_attributes(
        response_model=response_dict.get("model") or None,
        operation="audio.transcriptions",
        server_address=_get_openai_base_url(instance),
    )

    # duration metrics
    if duration and isinstance(duration, (float, int)) and duration_histogram:
        duration_histogram.record(duration, attributes=shared_attributes)
