"""Hermetic unit tests for the httpx instrumentation wiring (#2283)."""

import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from traceloop.sdk.tracing.tracing import init_httpx_instrumentor

URL_KEYS = ("url.full", "url", "http.url")
STATUS_KEYS = ("http.response.status_code", "http.status_code", "status_code")


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):
        pass


@pytest.fixture()
def local_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    yield f"http://127.0.0.1:{port}/hello"
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def test_init_httpx_instrumentor_returns_true():
    try:
        assert init_httpx_instrumentor() is True
    finally:
        HTTPXClientInstrumentor().uninstrument()


def test_httpx_excluded_url_is_not_traced(exporter, local_server):
    try:
        os.environ["OTEL_PYTHON_HTTPX_EXCLUDED_URLS"] = "127.0.0.1"
        assert init_httpx_instrumentor() is True
        with httpx.Client() as client:
            response = client.get(local_server, timeout=5)
        assert response.status_code == 200

        spans = exporter.get_finished_spans()
        assert not any(
            str(span.attributes.get(key, "")).startswith("http://127.0.0.1")
            for span in spans
            for key in URL_KEYS
        ), "excluded URL should not be traced"
    finally:
        HTTPXClientInstrumentor().uninstrument()
        os.environ.pop("OTEL_PYTHON_HTTPX_EXCLUDED_URLS", None)


def test_httpx_request_produces_a_span(exporter, local_server):
    try:
        assert init_httpx_instrumentor() is True
        with httpx.Client() as client:
            response = client.get(local_server, timeout=5)
        assert response.status_code == 200

        attrs = None
        for span in exporter.get_finished_spans():
            span_attrs = span.attributes or {}
            if any(
                str(span_attrs.get(key, "")).startswith("http://127.0.0.1")
                for key in URL_KEYS
            ):
                attrs = span_attrs
                break

        assert attrs is not None, (
            "no span captured for the httpx request to the local server"
        )
        assert any(attrs.get(key) == 200 for key in STATUS_KEYS), (
            f"httpx span missing a 200 status attribute: {attrs}"
        )
    finally:
        HTTPXClientInstrumentor().uninstrument()
