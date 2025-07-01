from threading import Thread
from typing import Generator

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from opentelemetry.instrumentation.mcp import McpInstrumentor
from tests.trace_collector import OTLPServer, Telemetry


@pytest.fixture(scope="session")
def telemetry() -> Telemetry:
    return Telemetry()


@pytest.fixture(scope="session")
def otlp_collector(telemetry: Telemetry) -> Generator[OTLPServer, None, None]:
    server = OTLPServer(("localhost", 0), telemetry)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join()


@pytest.fixture(scope="session")
def tracer_provider(
    otlp_collector: OTLPServer,
) -> Generator[trace_api.TracerProvider, None, None]:
    span_exporter = OTLPSpanExporter(
        f"http://localhost:{otlp_collector.server_port}/v1/traces"
    )
    tracer_provider = trace_sdk.TracerProvider()
    span_processor = SimpleSpanProcessor(span_exporter)
    tracer_provider.add_span_processor(span_processor)
    try:
        yield tracer_provider
    finally:
        tracer_provider.shutdown()


@pytest.fixture(scope="session")
def tracer(tracer_provider: trace_api.TracerProvider) -> trace_api.Tracer:
    return tracer_provider.get_tracer("mcp-test-client")


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: trace_api.TracerProvider, telemetry: Telemetry
) -> Generator[None, None, None]:
    instrumenter = McpInstrumentor()
    instrumenter.instrument(tracer_provider=tracer_provider)
    try:
        yield
    finally:
        instrumenter.uninstrument()
        telemetry.clear()
