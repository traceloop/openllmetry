import json

import pytest
import runpod
from opentelemetry.instrumentation.runpod.span_utils import (
    RUNPOD_ENDPOINT_ID,
    RUNPOD_JOB_ID,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as gen_ai_attributes,
)

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
    # `run` does not wait for the job, so no output can be recorded on the
    # submission span - the caller collects it through the returned Job handle.
    assert f"{gen_ai_attributes.GEN_AI_COMPLETION}.0.content" not in runpod_span.attributes


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
