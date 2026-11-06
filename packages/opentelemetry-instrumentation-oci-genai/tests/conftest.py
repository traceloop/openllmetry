"""Unit tests configuration module.

Recording cassettes requires OCI credentials::

    OCI_CONFIG_PROFILE=<profile> OCI_COMPARTMENT_ID=<compartment ocid> uv run pytest tests/ --record-mode=once

Both API key and session token (``security_token_file``) profiles are supported. Cassettes are scrubbed: the
``authorization`` signature (which embeds the session token), ``opc-request-id`` headers, request ids echoed in
response bodies and every OCID (including ``compartmentId``) are replaced before anything is written to disk.
"""

import os
import re

import oci
import oci.base_client
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from oci.generative_ai_inference import GenerativeAiInferenceClient
from opentelemetry.instrumentation.oci_genai import OCIGenAIInstrumentor
from opentelemetry.instrumentation.oci_genai.utils import TRACELOOP_TRACE_CONTENT
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.metrics import Counter, Histogram, MeterProvider
from opentelemetry.sdk.metrics.export import (
    AggregationTemporality,
    InMemoryMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from vcr.stubs.urllib3_stubs import VCRRequestsHTTPSConnection

REDACTED_COMPARTMENT_ID = "ocid1.compartment.oc1..redacted"
REDACTED_REQUEST_ID = "REDACTED/REDACTED/REDACTED"

_OCID_RE = re.compile(r"ocid1\.(?P<type>[a-z0-9_-]+)\.oc[0-9]+\.[a-z0-9-]*\.[a-zA-Z0-9]+")
_REQUEST_ID_RE = re.compile(r"[0-9A-F]{32}/[0-9A-F]{32}/[0-9A-F]{32}")
_SCRUBBED_RESPONSE_HEADERS = ("opc-request-id",)


def _scrub_text(text: str) -> str:
    text = _OCID_RE.sub(lambda match: f"ocid1.{match.group('type')}.oc1..redacted", text)
    return _REQUEST_ID_RE.sub(REDACTED_REQUEST_ID, text)


def _scrub_body(body):
    if body is None:
        return body
    if isinstance(body, bytes):
        return _scrub_text(body.decode("utf-8", errors="replace")).encode("utf-8")
    if isinstance(body, str):
        return _scrub_text(body)
    return body


def _scrub_request(request):
    request.body = _scrub_body(request.body)
    request.uri = _scrub_text(request.uri)
    return request


def _scrub_response(response):
    headers = response.get("headers") or {}
    for header in list(headers):
        if header.lower() in _SCRUBBED_RESPONSE_HEADERS:
            del headers[header]
    body = response.get("body") or {}
    if "string" in body:
        body["string"] = _scrub_body(body["string"])
    return response


def _playback_signer():
    """Signer used for cassette playback: a throwaway key and a dummy session token.

    The resulting ``authorization`` header is filtered out of the cassettes and ignored when matching requests.
    """
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return oci.auth.signers.SecurityTokenSigner("playback-token", private_key)


def _build_client(region):
    # No retries (deterministic cassettes) and a generous read timeout for cold model endpoints.
    client_kwargs = {"retry_strategy": oci.retry.NoneRetryStrategy(), "timeout": (10, 240)}
    profile = os.environ.get("OCI_CONFIG_PROFILE")
    if not profile:
        return GenerativeAiInferenceClient(config={"region": region}, signer=_playback_signer(), **client_kwargs)

    config = oci.config.from_file(profile_name=profile)
    if config.get("security_token_file"):
        with open(os.path.expanduser(config["security_token_file"])) as token_file:
            token = token_file.read()
        private_key = oci.signer.load_private_key_from_file(
            os.path.expanduser(config["key_file"]), config.get("pass_phrase")
        )
        signer = oci.auth.signers.SecurityTokenSigner(token, private_key)
        return GenerativeAiInferenceClient(config={"region": region}, signer=signer, **client_kwargs)
    config["region"] = region
    return GenerativeAiInferenceClient(config, **client_kwargs)


@pytest.fixture(scope="function", name="span_exporter")
def fixture_span_exporter():
    exporter = InMemorySpanExporter()
    yield exporter


@pytest.fixture(scope="function", name="tracer_provider")
def fixture_tracer_provider(span_exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture(scope="function", name="log_exporter")
def fixture_log_exporter():
    exporter = InMemoryLogExporter()
    yield exporter


@pytest.fixture(scope="function", name="logger_provider")
def fixture_logger_provider(log_exporter):
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))
    return provider


@pytest.fixture(scope="function", name="reader")
def fixture_reader():
    reader = InMemoryMetricReader({Counter: AggregationTemporality.DELTA, Histogram: AggregationTemporality.DELTA})
    return reader


@pytest.fixture(scope="function", name="meter_provider")
def fixture_meter_provider(reader):
    resource = Resource.create()
    meter_provider = MeterProvider(metric_readers=[reader], resource=resource)

    return meter_provider


@pytest.fixture
def compartment_id():
    return os.environ.get("OCI_COMPARTMENT_ID", REDACTED_COMPARTMENT_ID)


@pytest.fixture
def oci_client():
    return _build_client(os.environ.get("OCI_TEST_REGION", "us-chicago-1"))


@pytest.fixture(scope="function")
def instrument_legacy(reader, tracer_provider, meter_provider):
    instrumentor = OCIGenAIInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )

    yield instrumentor

    instrumentor.uninstrument()


@pytest.fixture(scope="function")
def instrument_with_content(reader, tracer_provider, logger_provider, meter_provider, monkeypatch):
    monkeypatch.setenv(TRACELOOP_TRACE_CONTENT, "True")

    instrumentor = OCIGenAIInstrumentor(use_attributes=False)
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    yield instrumentor

    instrumentor.uninstrument()


@pytest.fixture(scope="function")
def instrument_with_no_content(reader, tracer_provider, logger_provider, meter_provider, monkeypatch):
    monkeypatch.setenv(TRACELOOP_TRACE_CONTENT, "False")

    instrumentor = OCIGenAIInstrumentor(use_attributes=False)
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    yield instrumentor

    instrumentor.uninstrument()


@pytest.fixture(scope="module")
def vcr_config():
    return {
        # The OCI SDK routes HTTPS through its own ``OCIConnectionPool`` (``ConnectionCls = OCIConnection``),
        # which bypasses VCR's stock urllib3 patch; patch that pool's connection class explicitly.
        "custom_patches": ((oci.base_client.OCIConnectionPool, "ConnectionCls", VCRRequestsHTTPSConnection),),
        "filter_headers": [
            "authorization",
            "opc-request-id",
            "opc-client-info",
            "x-content-sha256",
            "date",
            "user-agent",
        ],
        "before_record_request": _scrub_request,
        "before_record_response": _scrub_response,
        "decode_compressed_response": True,
    }
