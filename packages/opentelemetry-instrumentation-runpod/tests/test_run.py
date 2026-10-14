import json
from types import SimpleNamespace

import pytest
import runpod
from opentelemetry.instrumentation.runpod.span_utils import (
    RUNPOD_ENDPOINT_ID,
    RUNPOD_JOB_ID,
    set_span_job_response_attributes,
    set_span_sync_response_attributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as gen_ai_attributes,
)
from opentelemetry.trace import INVALID_SPAN

ENDPOINT_ID = "kkcxkugqmlv82j"
JOB_ID = "9f4a3d54-4a1f-4d51-9d0b-7d1e8f2a6c31"


@pytest.fixture
def endpoint(fake_client):
    fake_client(
        {
            # `run` only submits the job; the SDK returns a Job handle.
            ("POST", f"{ENDPOINT_ID}/run"): {
                "id": JOB_ID,
                "status": "IN_QUEUE",
            }
        }
    )
    return runpod.Endpoint(ENDPOINT_ID)


def test_run_legacy(instrument_legacy, endpoint, span_exporter):
    job = endpoint.run({"prompt": "tell me a joke"})

    assert job.job_id == JOB_ID

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["runpod.run"]

    runpod_span = spans[0]
    assert runpod_span.attributes.get(gen_ai_attributes.GEN_AI_SYSTEM) == "runpod"
    assert runpod_span.attributes.get(gen_ai_attributes.GEN_AI_OPERATION_NAME) == "run"
    assert runpod_span.attributes.get(RUNPOD_ENDPOINT_ID) == ENDPOINT_ID

    content = runpod_span.attributes.get(f"{gen_ai_attributes.GEN_AI_PROMPT}.0.content")
    assert json.loads(content) == {"input": {"prompt": "tell me a joke"}}
    assert runpod_span.status.status_code.name == "OK"

    # `run` does not wait for the job, so the handle is the only response available
    # on the submission span - the caller collects the output through it.
    assert runpod_span.attributes.get(RUNPOD_JOB_ID) == JOB_ID
    completion = runpod_span.attributes.get(f"{gen_ai_attributes.GEN_AI_COMPLETION}.0.content")
    assert json.loads(completion) == {"job_id": JOB_ID}


def test_run_records_the_request_body_the_sdk_sends(
    instrument_legacy, fake_client, span_exporter
):
    """
    The recorded payload is the body the SDK puts on the wire. `Endpoint.run` wraps a
    payload unless it already carries a truthy `input`, so a falsy one is wrapped a
    second time - and that is what ends up in the span.
    """
    client = fake_client(
        {("POST", f"{ENDPOINT_ID}/run"): {"id": JOB_ID, "status": "IN_QUEUE"}}
    )
    endpoint = runpod.Endpoint(ENDPOINT_ID)

    endpoint.run({"input": ""})

    sent = client.calls[0][2]
    assert sent == {"input": {"input": ""}}

    runpod_span = span_exporter.get_finished_spans()[0]
    assert json.loads(
        runpod_span.attributes.get(f"{gen_ai_attributes.GEN_AI_PROMPT}.0.content")
    ) == sent


def test_response_and_payload_are_not_serialized_when_content_is_off(
    instrument_legacy, fake_client, span_exporter, monkeypatch
):
    """
    With TRACELOOP_TRACE_CONTENT=false the content is dropped whatever happens, so
    the serializer is not run at all - while metadata, which is not content, is
    recorded as usual.
    """
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "False")

    serialized = []

    def _record(value):
        serialized.append(value)
        return "{}"

    monkeypatch.setattr(
        "opentelemetry.instrumentation.runpod.span_utils.dump_object", _record
    )
    monkeypatch.setattr("opentelemetry.instrumentation.runpod.utils.dump_object", _record)

    fake_client({("POST", f"{ENDPOINT_ID}/run"): {"id": JOB_ID, "status": "IN_QUEUE"}})
    endpoint = runpod.Endpoint(ENDPOINT_ID)

    endpoint.run({"prompt": "tell me a joke"})

    assert serialized == []

    runpod_span = span_exporter.get_finished_spans()[0]
    assert runpod_span.attributes.get(RUNPOD_JOB_ID) == JOB_ID
    assert f"{gen_ai_attributes.GEN_AI_PROMPT}.0.content" not in runpod_span.attributes
    assert f"{gen_ai_attributes.GEN_AI_COMPLETION}.0.content" not in runpod_span.attributes


def test_response_handlers_skip_work_when_nothing_can_be_recorded(
    instrument_legacy, tracer_provider
):
    """
    With legacy attributes on a no-op tracer there is nothing to serialize for, so
    the handlers do not serialize the response at all.
    """
    job = SimpleNamespace(job_id=JOB_ID, endpoint_id=ENDPOINT_ID)

    assert set_span_job_response_attributes(INVALID_SPAN, job) is None
    assert set_span_sync_response_attributes(INVALID_SPAN, {"output": "x"}) is None

    span = tracer_provider.get_tracer(__name__).start_span("runpod.run")
    assert set_span_job_response_attributes(span, job) == json.dumps({"job_id": JOB_ID})
    span.end()


def test_run_with_events_instead_of_attributes(
    instrument_with_content, endpoint, span_exporter, log_exporter
):
    """
    With ``use_legacy_attributes=False`` the metadata stays on the span while the
    payload and the response are carried by log events.
    """
    endpoint.run({"prompt": "tell me a joke"})

    runpod_span = span_exporter.get_finished_spans()[0]
    assert runpod_span.attributes.get(RUNPOD_JOB_ID) == JOB_ID
    assert f"{gen_ai_attributes.GEN_AI_PROMPT}.0.content" not in runpod_span.attributes
    assert f"{gen_ai_attributes.GEN_AI_COMPLETION}.0.content" not in runpod_span.attributes

    logs = log_exporter.get_finished_logs()
    assert [record.log_record.event_name for record in logs] == [
        "gen_ai.user.message",
        "gen_ai.choice",
    ]
    assert logs[0].log_record.attributes.get(gen_ai_attributes.GEN_AI_SYSTEM) == "runpod"
    assert logs[0].log_record.body.get("content") == json.dumps(
        {"input": {"prompt": "tell me a joke"}}
    )
    assert logs[1].log_record.body.get("message") == {"content": json.dumps({"job_id": JOB_ID})}


def test_run_sync_falls_back_to_job_when_not_completed(
    instrument_legacy, fake_client, span_exporter
):
    """
    On a non-final ``status`` the SDK returns ``Job.output(timeout=...)``, which is
    the job output itself once the status poll completes.
    """
    fake_client(
        {
            # The job is still running, so `run_sync` falls through to
            # `Job.output(timeout=...)`, i.e. a GET /status poll.
            ("POST", f"{ENDPOINT_ID}/runsync"): {
                "id": JOB_ID,
                "status": "IN_PROGRESS",
                "output": None,
            },
            ("GET", f"{ENDPOINT_ID}/status/{JOB_ID}"): {
                "id": JOB_ID,
                "status": "COMPLETED",
                "output": "42",
            },
        }
    )
    endpoint = runpod.Endpoint(ENDPOINT_ID)

    assert endpoint.run_sync({"prompt": "meaning of life"}, timeout=1) == "42"

    spans = span_exporter.get_finished_spans()
    # Only the submission is traced: the follow-up status polling is performed by
    # the Job object built inside the SDK, which this package does not instrument.
    assert [span.name for span in spans] == ["runpod.run_sync"]
    # A string output is recorded as-is rather than JSON-encoded.
    assert (
        spans[0].attributes.get(f"{gen_ai_attributes.GEN_AI_COMPLETION}.0.content") == "42"
    )


def test_run_sync_returns_job_when_status_is_failed(
    instrument_legacy, fake_client, span_exporter
):
    """
    A ``FAILED`` status counts as completed for the SDK, so the recorded completion
    is the (null) job output rather than a job handle.
    """
    fake_client(
        {
            ("POST", f"{ENDPOINT_ID}/runsync"): {
                "id": JOB_ID,
                "status": "FAILED",
                "output": None,
            }
        }
    )
    endpoint = runpod.Endpoint(ENDPOINT_ID)

    assert endpoint.run_sync({"prompt": "boom"}, timeout=1) is None

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["runpod.run_sync"]
    assert RUNPOD_JOB_ID not in spans[0].attributes


def test_run_sync_positional_argument(instrument_legacy, fake_client, span_exporter):
    """The payload may be passed positionally instead of as a keyword."""
    fake_client(
        {
            ("POST", f"{ENDPOINT_ID}/runsync"): {"id": JOB_ID, "status": "COMPLETED", "output": 7}
        }
    )
    endpoint = runpod.Endpoint(ENDPOINT_ID)

    assert endpoint.run_sync({"prompt": "positional"}) == 7

    runpod_span = span_exporter.get_finished_spans()[0]
    assert json.loads(
        runpod_span.attributes.get(f"{gen_ai_attributes.GEN_AI_PROMPT}.0.content")
    ) == {"input": {"prompt": "positional"}}


def test_health_is_not_instrumented(instrument_legacy, fake_client, span_exporter):
    """`health` is a status probe, not a serverless call, so it produces no span."""
    fake_client({("GET", f"{ENDPOINT_ID}/health"): {"workers": {"idle": 1}}})
    endpoint = runpod.Endpoint(ENDPOINT_ID)

    assert endpoint.health() == {"workers": {"idle": 1}}
    assert span_exporter.get_finished_spans() == ()
