from opentelemetry.instrumentation.runpod.utils import (
    dont_throw,
    dump_object,
    is_runpod_job,
    should_emit_events,
    should_send_prompts,
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
def set_input_content_attributes(span, content):
    """
    Records the request payload as legacy attributes, honoring the
    TRACELOOP_TRACE_CONTENT flag.

    ``content`` is the serialized payload, as returned by
    ``opentelemetry.instrumentation.runpod.utils.get_request_content``. Legacy
    attributes are not written when the instrumentor runs with
    ``use_legacy_attributes=False``; that mode carries the same content in a
    ``gen_ai.user.message`` event instead - see the ``event_emitter`` module.
    """
    if not span.is_recording() or should_emit_events() or not should_send_prompts():
        return

    if content is None:
        return

    _set_span_attribute(span, f"{GenAIAttributes.GEN_AI_PROMPT}.0.role", "user")
    _set_span_attribute(span, f"{GenAIAttributes.GEN_AI_PROMPT}.0.content", content)


@dont_throw
def set_response_content_attributes(span, content):
    """
    Records the response content as legacy attributes, honoring the
    TRACELOOP_TRACE_CONTENT flag.

    ``content`` is the serialized response, as returned by the response handlers
    below. As with the request side, nothing is written when the instrumentor runs
    with ``use_legacy_attributes=False``: a ``gen_ai.choice`` event carries the
    content instead.
    """
    if not span.is_recording() or should_emit_events() or not should_send_prompts():
        return

    if content is None:
        return

    _set_span_attribute(span, f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role", "assistant")
    _set_span_attribute(span, f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content", content)


@dont_throw
def set_span_sync_response_attributes(span, response):
    """
    Records response metadata for ``Endpoint.run_sync``, and returns the content that
    the caller records on the span or in the response event.

    ``run_sync`` returns the job output - never the job envelope - whenever the job
    reaches a final state within the timeout, and the output of
    ``Job.output(timeout=...)`` otherwise, so in runpod 1.x a ``run_sync`` span does
    not observe the job id. The ``Job`` branch below is a safety net for SDK
    versions that hand the handle back to the caller; the instrumentation never
    issues a request of its own to fill it in.
    """
    if is_runpod_job(response):
        job_id = getattr(response, "job_id", None)
        _set_span_attribute(span, RUNPOD_JOB_ID, job_id)
        return dump_object({"job_id": job_id})

    return dump_object(response)


@dont_throw
def set_span_job_response_attributes(span, response):
    """
    Records response metadata for the ``Job`` handle returned by ``Endpoint.run`` and
    ``AsyncioEndpoint.run``, and returns the content that the caller records on the
    span or in the response event.

    Only the handle is available here: the SDK issues a second, separate request
    when the caller awaits ``Job.output()`` or consumes ``Job.stream()``, and those
    go through the aiohttp session rather than through the ``Endpoint`` classes. See
    the README section on coverage.
    """
    job_id = getattr(response, "job_id", None)
    if job_id is None:
        return None

    _set_span_attribute(span, RUNPOD_JOB_ID, job_id)
    return dump_object({"job_id": job_id})


@dont_throw
def set_span_status_ok(span):
    if span.is_recording():
        span.set_status(Status(StatusCode.OK))


@dont_throw
def set_span_status_error(span, exception):
    if span.is_recording():
        span.set_status(Status(StatusCode.ERROR, str(exception)))
        span.record_exception(exception)
