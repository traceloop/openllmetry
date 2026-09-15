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
JOB_ID = "1c2d3e4f-5a6b-7c8d-9e0f-1a2b3c4d5e6f"


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


class _FakeRequest:
    """Minimal stand-in for the aiohttp request context manager the SDK awaits."""

    def __init__(self, payload):
        self._payload = payload

    async def __aenter__(self):
        return _FakeResponse(self._payload)

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeAiohttpSession:
    """Records requests and replays canned JSON, so tests never touch the network."""

    def __init__(self, post_responses=None, get_responses=None):
        self.post_responses = post_responses or {}
        self.get_responses = get_responses or {}
        self.calls = []

    def post(self, url, headers=None, json=None):  # pylint: disable=redefined-outer-name
        self.calls.append(("POST", url, json))
        return _FakeRequest(self.post_responses[url])

    def get(self, url, headers=None):
        self.calls.append(("GET", url, None))
        return _FakeRequest(self.get_responses[url])


@pytest.fixture
def async_endpoint():
    session = FakeAiohttpSession(
        post_responses={
            f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run": {"id": JOB_ID, "status": "IN_QUEUE"}
        }
    )
    return runpod.AsyncioEndpoint(ENDPOINT_ID, session=session, api_key="test_api_key")


async def test_asyncio_run_legacy(instrument_legacy, async_endpoint, span_exporter):
    job = await async_endpoint.run({"prompt": "tell me a joke"})

    assert job.job_id == JOB_ID

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["runpod.run"]

    runpod_span = spans[0]
    assert runpod_span.attributes.get(gen_ai_attributes.GEN_AI_SYSTEM) == "runpod"
    assert runpod_span.attributes.get(gen_ai_attributes.GEN_AI_OPERATION_NAME) == "run"
    assert runpod_span.attributes.get(RUNPOD_ENDPOINT_ID) == ENDPOINT_ID
    assert runpod_span.attributes.get(RUNPOD_JOB_ID) == JOB_ID

    content = runpod_span.attributes.get(f"{gen_ai_attributes.GEN_AI_PROMPT}.0.content")
    # `AsyncioEndpoint.run` always wraps the payload in an "input" key itself.
    assert json.loads(content) == {"input": {"prompt": "tell me a joke"}}
    assert runpod_span.status.status_code.name == "OK"


async def test_asyncio_run_with_no_content(
    instrument_with_no_content, async_endpoint, span_exporter
):
    await async_endpoint.run({"prompt": "tell me a joke"})

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["runpod.run"]
    assert f"{gen_ai_attributes.GEN_AI_PROMPT}.0.content" not in spans[0].attributes


async def test_asyncio_run_error_is_recorded(instrument_legacy, span_exporter):
    session = FakeAiohttpSession()
    endpoint = runpod.AsyncioEndpoint(ENDPOINT_ID, session=session, api_key="test_api_key")

    with pytest.raises(KeyError):
        await endpoint.run({"prompt": "boom"})

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["runpod.run"]
    assert spans[0].status.status_code.name == "ERROR"
    assert spans[0].events[0].name == "exception"
    assert spans[0].events[0].attributes["exception.type"] == "KeyError"
