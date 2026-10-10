"""Regression tests for https://github.com/traceloop/openllmetry/issues/2845

When the user instruments the underlying HTTP client (e.g. httpx) in addition
to OpenAI, every OpenAI call used to be traced twice: once by this
instrumentation (``openai.chat``) and once by the HTTP instrumentation (a raw
``POST`` span). The OpenAI instrumentation now suppresses HTTP-client
instrumentation around the request, so only the ``openai.chat`` span is emitted.

These tests hit a local HTTP server over a real socket so the httpx
instrumentation actually runs (VCR replay would bypass the transport it wraps).
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from openai import OpenAI
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

_CHAT_RESPONSE = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
}

_STREAM_CHUNKS = [
    {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": "hello"},
             "finish_reason": None}
        ],
    },
    {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    },
]


class _OpenAIStubHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in _STREAM_CHUNKS:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        body = json.dumps(_CHAT_RESPONSE).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence the default stderr logging
        pass


@pytest.fixture
def openai_stub_server():
    server = HTTPServer(("127.0.0.1", 0), _OpenAIStubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}/v1"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.fixture
def instrument_httpx(tracer_provider):
    HTTPXClientInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    HTTPXClientInstrumentor().uninstrument()


def _http_spans(span_exporter):
    return [
        span
        for span in span_exporter.get_finished_spans()
        if span.attributes.get("http.request.method")
        or span.attributes.get("http.method")
    ]


def test_chat_does_not_emit_duplicate_httpx_span(
    instrument_legacy, instrument_httpx, span_exporter, openai_stub_server
):
    # No api_key: the autouse `environment` fixture sets OPENAI_API_KEY, matching
    # the repo's existing local-server client fixtures (e.g. vllm_openai_client).
    client = OpenAI(base_url=openai_stub_server)

    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "say hello only"}],
    )

    span_names = [span.name for span in span_exporter.get_finished_spans()]
    assert span_names == ["openai.chat"]
    assert _http_spans(span_exporter) == []


def test_chat_streaming_does_not_emit_duplicate_httpx_span(
    instrument_legacy, instrument_httpx, span_exporter, openai_stub_server
):
    # No api_key: the autouse `environment` fixture sets OPENAI_API_KEY, matching
    # the repo's existing local-server client fixtures (e.g. vllm_openai_client).
    client = OpenAI(base_url=openai_stub_server)

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "say hello only"}],
        stream=True,
    )
    for _ in stream:
        pass

    span_names = [span.name for span in span_exporter.get_finished_spans()]
    assert span_names == ["openai.chat"]
    assert _http_spans(span_exporter) == []
