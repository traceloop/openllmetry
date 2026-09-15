from opentelemetry.instrumentation.runpod.utils import (
    dont_throw,
    dump_object,
    get_request_input,
    is_runpod_job,
    should_send_prompts,
    unwrap_input,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.trace.status import Status, StatusCode

# RunPod-specific span attributes (vendor namespace).
#
# These live here rather than in `opentelemetry-semantic-conventions-ai` so the
# instrumentation works against the released version of that package. They are
# candidates for promotion to `SpanAttributes` in a future release of the shared
# package, next to the other vendor attributes (Chroma, Milvus, Qdrant, ...).
RUNPOD_ENDPOINT_ID = "runpod.endpoint_id"
RUNPOD_JOB_ID = "runpod.job_id"

# Operation name constants, matching the instrumented RunPod SDK method names.
OPERATION_RUN = "run"
OPERATION_RUN_SYNC = "run_sync"


def _set_span_attribute(span, name, value):
    if value is not None and value != "":
        span.set_attribute(name, value)
    return


@dont_throw
def set_span_request_attributes(span, to_wrap, instance):
    """
    Set the RunPod specific attributes describing which endpoint was called and
    which serverless operation was used.
    """
    if not span.is_recording():
        return

    _set_span_attribute(span, RUNPOD_ENDPOINT_ID, getattr(instance, "endpoint_id", None))
    _set_span_attribute(span, GenAIAttributes.GEN_AI_OPERATION_NAME, to_wrap.get("operation_name"))


@dont_throw
def set_input_content_attributes(span, args, kwargs):
    """Records the request payload, honoring the TRACELOOP_TRACE_CONTENT flag."""
    if not span.is_recording() or not should_send_prompts():
        return

    request_input = unwrap_input(get_request_input(args, kwargs))
    content = dump_object(request_input)
    if content is None:
        return

    _set_span_attribute(span, f"{GenAIAttributes.GEN_AI_PROMPT}.0.role", "user")
    _set_span_attribute(span, f"{GenAIAttributes.GEN_AI_PROMPT}.0.content", content)


@dont_throw
def set_response_content_attributes(span, response):
    """Records the serverless response, honoring the TRACELOOP_TRACE_CONTENT flag."""
    if not span.is_recording() or not should_send_prompts():
        return

    content = dump_object(response)
    if content is None:
        return

    _set_span_attribute(span, f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role", "assistant")
    _set_span_attribute(span, f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content", content)


@dont_throw
def set_span_sync_response_attributes(span, response):
    """
    Records response metadata for ``Endpoint.run_sync``.

    ``run_sync`` returns the job output - never the job envelope - whenever the job
    reaches a final state within the timeout, and the output of
    ``Job.output(timeout=...)`` otherwise, so in runpod 1.x a ``run_sync`` span does
    not observe the job id. The ``Job`` branch below is a safety net for SDK
    versions that hand the handle back to the caller; the instrumentation never
    issues a request of its own to fill it in.
    """
    if not span.is_recording():
        return

    if is_runpod_job(response):
        job_id = getattr(response, "job_id", None)
        _set_span_attribute(span, RUNPOD_JOB_ID, job_id)
        set_response_content_attributes(span, {"job_id": job_id})
        return

    set_response_content_attributes(span, response)


@dont_throw
def set_span_async_response_attributes(span, response):
    """
    Records metadata for the ``Job`` handle returned by ``AsyncioEndpoint.run``.

    Only the handle is available here: the SDK issues a second, separate request
    when the caller awaits ``Job.output()`` or consumes ``Job.stream()``, and those
    go through the aiohttp session rather than through ``AsyncioEndpoint``. See the
    README section on coverage.
    """
    if not span.is_recording():
        return

    job_id = getattr(response, "job_id", None)
    if job_id is not None:
        _set_span_attribute(span, RUNPOD_JOB_ID, job_id)
        set_response_content_attributes(span, {"job_id": job_id})


@dont_throw
def set_span_status_ok(span):
    if span.is_recording():
        span.set_status(Status(StatusCode.OK))


@dont_throw
def set_span_status_error(span, exception):
    if span.is_recording():
        span.set_status(Status(StatusCode.ERROR, str(exception)))
        span.record_exception(exception)
