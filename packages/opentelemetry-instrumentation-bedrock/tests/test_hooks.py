"""Exercise application hooks through real, offline boto3/aioboto3 clients."""

import copy
import io
import json
from contextlib import asynccontextmanager

import aioboto3
import boto3
import pytest
from botocore.response import StreamingBody
from opentelemetry import context, trace
from opentelemetry.instrumentation.bedrock import BedrockInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
from opentelemetry.trace import StatusCode

from tests.traces.test_aioboto3 import (
    MODEL_ID,
    MESSAGE_LIST,
    _fake_converse_response,
    _fake_converse_stream_response,
    _fake_invoke_model_response,
    _fake_invoke_stream_response,
)


METHODS = [
    ("invoke_model", "InvokeModel"),
    ("invoke_model_with_response_stream", "InvokeModelWithResponseStream"),
    ("converse", "Converse"),
    ("converse_stream", "ConverseStream"),
]
STREAM_METHODS = [METHODS[1], METHODS[3]]


class _EventStream:
    def __init__(self, events, error=None):
        self.events = events
        self.error = error

    def _parse_event(self, event):
        return event

    def __iter__(self):
        for event in self.events:
            yield self._parse_event(event)
        if self.error:
            raise self.error


class _AsyncEventStream(_EventStream):
    async def _parse_event(self, event):
        return event

    async def __aiter__(self):
        for event in self.events:
            yield await self._parse_event(event)
        if self.error:
            raise self.error


def _response(method, is_async, stream_error=None):
    factory = {
        "invoke_model": _fake_invoke_model_response,
        "invoke_model_with_response_stream": _fake_invoke_stream_response,
        "converse": _fake_converse_response,
        "converse_stream": _fake_converse_stream_response,
    }[method]
    response = factory()
    if method in {m for m, _ in STREAM_METHODS}:
        key = "stream" if method == "converse_stream" else "body"
        events = response[key]._events
        if stream_error:
            events = events[:1]
        stream_class = _AsyncEventStream if is_async else _EventStream
        response[key] = stream_class(events, stream_error)
    elif method == "invoke_model" and not is_async:
        raw = response["body"]._raw
        response["body"] = StreamingBody(io.BytesIO(raw), len(raw))
    response["ResponseMetadata"]["RequestId"] = "offline-request"
    return response


def _request(method):
    if method.startswith("invoke_model"):
        return {"modelId": MODEL_ID, "body": json.dumps({"messages": MESSAGE_LIST})}
    return {"modelId": MODEL_ID, "messages": copy.deepcopy(MESSAGE_LIST)}


@asynccontextmanager
async def _client(is_async, response, error=None):
    def call(operation, params):
        if error:
            raise error
        return response

    credentials = {
        "aws_access_key_id": "test",
        "aws_secret_access_key": "test",
        "region_name": "us-east-1",
    }
    if is_async:
        async with aioboto3.Session(**credentials).client("bedrock-runtime") as client:

            async def async_call(operation, params):
                return call(operation, params)

            client._make_api_call = async_call
            yield client
    else:
        client = boto3.client(service_name="bedrock-runtime", **credentials)
        try:
            client._make_api_call = call
            yield client
        finally:
            client.close()


async def _call(client, method, request, is_async):
    result = getattr(client, method)(**request)
    return await result if is_async else result


async def _consume(response, method, is_async):
    key = "stream" if method == "converse_stream" else "body"
    if is_async:
        return [event async for event in response[key]]
    return list(response[key])


@pytest.fixture
def configure_hooks(tracer_provider, meter_provider, logger_provider):
    instrumentor = BedrockInstrumentor()
    instrumentor.uninstrument()

    def configure(use_attributes=True, **hooks):
        BedrockInstrumentor(use_attributes=use_attributes)
        instrumentor.instrument(
            **{
                "tracer_provider": tracer_provider,
                "meter_provider": meter_provider,
                "logger_provider": logger_provider,
                **hooks,
            }
        )

    yield configure
    instrumentor.uninstrument()


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("method,operation", METHODS)
@pytest.mark.parametrize("use_attributes", [True, False], ids=["attributes", "events"])
async def test_hooks_receive_data_and_enrich_active_span(
    configure_hooks, span_exporter, is_async, method, operation, use_attributes
):
    request = _request(method)
    original_request = copy.deepcopy(request)
    response = _response(method, is_async)
    metadata = copy.deepcopy(response["ResponseMetadata"])
    calls = []

    def request_hook(span, service, actual_operation, params):
        assert trace.get_current_span() is span
        assert span.is_recording()
        assert (service, actual_operation) == ("bedrock-runtime", operation)
        assert params == request
        for name, value in request.items():
            assert params[name] is value
        span.set_attribute("custom.request", params["modelId"])
        calls.append("request")
        return {"body": "ignored"}

    def response_hook(span, service, actual_operation, result):
        assert trace.get_current_span() is span
        assert span.is_recording()
        assert (service, actual_operation) == ("bedrock-runtime", operation)
        assert result is response
        span.set_attribute("custom.response", result["ResponseMetadata"]["RequestId"])
        calls.append("response")
        return "ignored"

    configure_hooks(use_attributes=use_attributes, request_hook=request_hook, response_hook=response_hook)
    async with _client(is_async, response) as client:
        returned = await _call(client, method, request, is_async)
        assert returned is response
        if method in {m for m, _ in STREAM_METHODS}:
            assert calls == ["request"]
            assert not span_exporter.get_finished_spans()
            events = await _consume(returned, method, is_async)
            assert events
            # Re-iteration must not fire a second response hook.
            await _consume(returned, method, is_async)
        elif method == "invoke_model":
            raw = returned["body"].read()
            if is_async:
                raw = await raw
            assert json.loads(raw)["output"]["message"]["content"][0]["text"]

    assert calls == ["request", "response"]
    assert request == original_request
    assert response["ResponseMetadata"] == metadata
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["custom.request"] == MODEL_ID
    assert spans[0].attributes["custom.response"] == "offline-request"
    assert trace.get_current_span() is trace.INVALID_SPAN


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("method,operation", METHODS)
@pytest.mark.parametrize("failing_hook", ["request_hook", "response_hook"])
async def test_hook_errors_preserve_response(
    configure_hooks, span_exporter, caplog, is_async, method, operation, failing_hook
):
    calls = []

    def fail(span, service, operation, data):
        calls.append(operation)
        raise ValueError("application hook failed")

    configure_hooks(**{failing_hook: fail})
    response = _response(method, is_async)
    async with _client(is_async, response) as client:
        assert await _call(client, method, _request(method), is_async) is response
        if method in {m for m, _ in STREAM_METHODS}:
            assert await _consume(response, method, is_async)
    assert calls == [operation]
    assert len(span_exporter.get_finished_spans()) == 1
    assert span_exporter.get_finished_spans()[0].status.status_code == StatusCode.UNSET
    assert "application hook failed" in caplog.text


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("method,operation", METHODS)
async def test_request_hook_error_does_not_replace_client_error(configure_hooks, is_async, method, operation):
    responses = []
    requests = []

    def fail(*args):
        requests.append(args)
        raise ValueError("hook error")

    configure_hooks(request_hook=fail, response_hook=lambda *args: responses.append(args))
    error = RuntimeError("client error")
    async with _client(is_async, None, error) as client:
        with pytest.raises(RuntimeError) as raised:
            await _call(client, method, _request(method), is_async)
    assert raised.value is error
    assert len(requests) == 1
    assert responses == []


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("method,operation", STREAM_METHODS)
@pytest.mark.parametrize("ending", ["close", "error"])
async def test_stream_response_hook_runs_once_on_iterator_close_or_error(
    configure_hooks, span_exporter, is_async, method, operation, ending
):
    calls = []

    def response_hook(span, service, actual_operation, result):
        assert trace.get_current_span() is span
        assert span.is_recording()
        assert result is response
        assert actual_operation == operation
        span.set_attribute("custom.partial", True)
        calls.append("response")
        if ending == "error":
            raise ValueError("hook error must not mask stream error")

    configure_hooks(response_hook=response_hook)
    error = RuntimeError("stream error") if ending == "error" else None
    response = _response(method, is_async, error)
    async with _client(is_async, response) as client:
        returned = await _call(client, method, _request(method), is_async)
        assert calls == []
        if error:
            with pytest.raises(RuntimeError) as raised:
                await _consume(returned, method, is_async)
            assert raised.value is error
        else:
            stream = returned["stream" if method == "converse_stream" else "body"]
            iterator = stream.__aiter__() if is_async else iter(stream)
            if is_async:
                await iterator.__anext__()
                await iterator.aclose()
            else:
                next(iterator)
                iterator.close()
    assert calls == ["response"]
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["custom.partial"] is True


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("method,operation", METHODS)
async def test_suppressed_calls_do_not_run_hooks(configure_hooks, span_exporter, is_async, method, operation):
    calls = []
    configure_hooks(request_hook=lambda *a: calls.append(a), response_hook=lambda *a: calls.append(a))
    response = _response(method, is_async)
    token = context.attach(context.set_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True))
    try:
        async with _client(is_async, response) as client:
            assert await _call(client, method, _request(method), is_async) is response
            if method in {m for m, _ in STREAM_METHODS}:
                await _consume(response, method, is_async)
    finally:
        context.detach(token)
    assert calls == []
    assert not span_exporter.get_finished_spans()


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("method,operation", STREAM_METHODS)
async def test_empty_stream_finishes_response_hook_once(configure_hooks, span_exporter, is_async, method, operation):
    calls = []
    configure_hooks(response_hook=lambda *args: calls.append(args))
    response = _response(method, is_async)
    response["stream" if method == "converse_stream" else "body"].events = []
    async with _client(is_async, response) as client:
        await _call(client, method, _request(method), is_async)
        assert calls == []
        assert await _consume(response, method, is_async) == []
        assert await _consume(response, method, is_async) == []
    assert len(calls) == 1
    assert calls[0][2:] == (operation, response)
    assert len(span_exporter.get_finished_spans()) == 1


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("method,operation", METHODS)
async def test_sampled_out_spans_do_not_run_hooks(configure_hooks, is_async, method, operation):
    calls = []
    configure_hooks(
        tracer_provider=TracerProvider(sampler=ALWAYS_OFF),
        request_hook=lambda *args: calls.append(args),
        response_hook=lambda *args: calls.append(args),
    )
    response = _response(method, is_async)
    async with _client(is_async, response) as client:
        assert await _call(client, method, _request(method), is_async) is response
        if method in {m for m, _ in STREAM_METHODS}:
            assert await _consume(response, method, is_async)
    assert calls == []
