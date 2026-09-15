import json

import pytest
import runpod
from opentelemetry.instrumentation.runpod.span_utils import (
    RUNPOD_ENDPOINT_ID,
    RUNPOD_JOB_ID,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

ENDPOINT_ID = "kkcxkugqmlv82j"


@pytest.fixture
def endpoint(fake_client):
    fake_client(
        {
            ("POST", f"{ENDPOINT_ID}/runsync"): {
                "id": "ea1b7b0e-2a43-4b1c-9a5f-1b2c3d4e5f60",
                "status": "COMPLETED",
                "output": {"result": "hello from runpod"},
            }
        }
    )
    return runpod.Endpoint(ENDPOINT_ID)


def test_run_sync_legacy(instrument_legacy, endpoint, span_exporter):
    response = endpoint.run_sync({"prompt": "say hello"})

    assert response == {"result": "hello from runpod"}

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["runpod.run_sync"]

    runpod_span = spans[0]
    assert runpod_span.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "runpod"
    assert runpod_span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "run_sync"
    assert runpod_span.attributes.get(RUNPOD_ENDPOINT_ID) == ENDPOINT_ID
    # `run_sync` returns only the job output when the job completes in time, so the
    # job id is not observable on this path.
    assert RUNPOD_JOB_ID not in runpod_span.attributes

    # The SDK normalizes the payload into {"input": ...} before POSTing it, and
    # that is what the instrumentation records.
    content = runpod_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.0.content")
    assert json.loads(content) == {"input": {"prompt": "say hello"}}
    assert runpod_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.0.role") == "user"

    completion = runpod_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
    assert json.loads(completion) == {"result": "hello from runpod"}
    assert runpod_span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role") == "assistant"

    assert runpod_span.status.status_code.name == "OK"


def test_run_sync_with_no_content(
    instrument_with_no_content, endpoint, span_exporter
):
    endpoint.run_sync({"prompt": "say hello"})

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["runpod.run_sync"]

    runpod_span = spans[0]
    # The call itself is still traced, only the payload is withheld.
    assert runpod_span.attributes.get(RUNPOD_ENDPOINT_ID) == ENDPOINT_ID
    assert f"{GenAIAttributes.GEN_AI_PROMPT}.0.content" not in runpod_span.attributes
    assert f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content" not in runpod_span.attributes


def test_run_sync_error_is_recorded(instrument_legacy, fake_client, span_exporter):
    fake_client({})  # every request raises AssertionError
    endpoint = runpod.Endpoint(ENDPOINT_ID)

    with pytest.raises(AssertionError):
        endpoint.run_sync({"prompt": "say hello"})

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["runpod.run_sync"]

    runpod_span = spans[0]
    assert runpod_span.status.status_code.name == "ERROR"
    assert runpod_span.events[0].name == "exception"
    assert runpod_span.events[0].attributes["exception.type"] == "AssertionError"


def test_run_sync_does_not_span_when_uninstrumented(endpoint, span_exporter):
    endpoint.run_sync({"prompt": "say hello"})

    assert span_exporter.get_finished_spans() == ()
