"""Tests for InstrumentedStreamWriter.send() result handling (issue #4038).

send() is wrapped in @dont_throw, so anything that raises while inspecting
request.result is swallowed and the wrapped send() is never awaited -- the MCP
message is dropped. These tests pin both halves: the message is always
forwarded, and the ERROR status is still set for isError results.
"""
import pytest
from opentelemetry.trace.status import StatusCode

from opentelemetry.instrumentation.mcp.instrumentation import InstrumentedStreamWriter


class FakeRoot:
    """Stands in for JSONRPCMessage.root; only .result and .id are read here."""

    def __init__(self, result):
        self.result = result
        self.id = 1


class FakeItem:
    def __init__(self, result):
        self.root = FakeRoot(result)


class FakeStream:
    def __init__(self):
        self.sent = []

    async def send(self, item):
        self.sent.append(item)
        return "sent"


class ObjectBlock:
    type = "text"
    text = "object-shaped failure"


class ObjectResult:
    """An MCP result exposing fields as attributes rather than dict keys."""

    isError = True
    content = [ObjectBlock()]


@pytest.mark.parametrize(
    "result,expected_description",
    [
        # first content block is an image, so there is no "text" key
        (
            {
                "isError": True,
                "content": [
                    {"type": "image", "data": "iVBOR", "mimeType": "image/png"}
                ],
            },
            "",
        ),
        # error result carrying no content blocks at all
        ({"isError": True, "content": []}, ""),
        # error result with no "content" key
        ({"isError": True}, ""),
        # object-shaped result: `"isError" in result` would raise TypeError
        (ObjectResult(), "object-shaped failure"),
        # the already-working shape, to pin that it is unchanged
        (
            {"isError": True, "content": [{"type": "text", "text": "tool blew up"}]},
            "tool blew up",
        ),
    ],
)
async def test_error_results_are_forwarded_and_marked(
    result, expected_description, span_exporter, tracer_provider
):
    stream = FakeStream()
    writer = InstrumentedStreamWriter(stream, tracer_provider.get_tracer(__name__))

    returned = await writer.send(FakeItem(result))

    assert returned == "sent", "send() swallowed an error and dropped the message"
    assert len(stream.sent) == 1

    span = span_exporter.get_finished_spans()[-1]
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description == expected_description


async def test_non_error_result_leaves_status_unset(span_exporter, tracer_provider):
    stream = FakeStream()
    writer = InstrumentedStreamWriter(stream, tracer_provider.get_tracer(__name__))

    returned = await writer.send(
        FakeItem({"content": [{"type": "text", "text": "fine"}]})
    )

    assert returned == "sent"
    span = span_exporter.get_finished_spans()[-1]
    assert span.status.status_code == StatusCode.UNSET
