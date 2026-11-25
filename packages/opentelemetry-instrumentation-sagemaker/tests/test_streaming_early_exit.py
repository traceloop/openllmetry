import binascii
import struct
from unittest.mock import patch

import pytest
from botocore.awsrequest import AWSResponse
from botocore.exceptions import EventStreamError
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace.status import StatusCode


def _header(name, value):
    name, value = name.encode(), value.encode()
    return struct.pack("!B", len(name)) + name + b"\x07" + struct.pack("!H", len(value)) + value


def _message(payload, event_type="PayloadPart", message_type="event"):
    if message_type == "exception":
        headers = _header(":exception-type", event_type) + _header(":message-type", "exception")
    else:
        headers = _header(":event-type", event_type) + _header(":message-type", "event")
    headers += _header(":content-type", "application/octet-stream")
    prelude = struct.pack("!II", 12 + len(headers) + len(payload) + 4, len(headers))
    prelude += struct.pack("!I", binascii.crc32(prelude) & 0xFFFFFFFF)
    message = prelude + headers + payload
    return message + struct.pack("!I", binascii.crc32(message) & 0xFFFFFFFF)


class _RawStream:
    def __init__(self, data):
        self._data = data

    def stream(self, *args, **kwargs):
        yield self._data

    def close(self):
        pass

    def release_conn(self):
        pass


def _invoke_stream(smrt, *messages):
    response = AWSResponse(
        "https://runtime.sagemaker.us-east-1.amazonaws.com",
        200,
        {"Content-Type": "application/vnd.amazon.eventstream"},
        _RawStream(b"".join(messages)),
    )
    with patch("botocore.endpoint.URLLib3Session.send", return_value=response):
        return smrt.invoke_endpoint_with_response_stream(
            EndpointName="my-test-endpoint",
            Body=b'{"inputs": "Tell me a joke"}',
            ContentType="application/json",
        )


def test_stream_span_ends_when_caller_breaks(instrument_legacy, smrt, span_exporter):
    response = _invoke_stream(smrt, _message(b'{"token": "a"}'), _message(b'{"token": "b"}'))
    for _ in response["Body"]:
        break

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.UNSET
    assert spans[0].attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] == '"{\\"token\\": \\"a\\"}"'


def test_stream_span_ends_when_caller_closes(instrument_legacy, smrt, span_exporter):
    response = _invoke_stream(smrt, _message(b'{"token": "a"}'))
    response["Body"].close()

    assert len(span_exporter.get_finished_spans()) == 1


def test_stream_span_ends_once_when_drained(instrument_legacy, smrt, span_exporter):
    response = _invoke_stream(smrt, _message(b'{"token": "a"}'), _message(b'{"token": "b"}'))
    with patch("opentelemetry.sdk.trace.logger") as sdk_logger:
        list(response["Body"])
        response["Body"].close()

    assert len(span_exporter.get_finished_spans()) == 1
    sdk_logger.warning.assert_not_called()


def test_stream_error_sets_span_status(instrument_legacy, smrt, span_exporter):
    response = _invoke_stream(
        smrt,
        _message(b'{"token": "a"}'),
        _message(b'{"Message": "boom"}', event_type="ModelStreamError", message_type="exception"),
    )
    with pytest.raises(EventStreamError):
        list(response["Body"])

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR
    assert spans[0].attributes.get("error.type") == "EventStreamError"
