"""Metadata values must be plain data, never a stringified object.

Association properties are set from the caller's ``config={"metadata": ...}`` and
are copied onto every descendant span. Before this test, any non-primitive value
was passed through ``str()``, so an object's repr landed on the whole trace. A
model, client or config object renders its constructor state, which routinely
includes an API key, so one such value exported a credential to the trace
backend. Metadata is also not gated by TRACELOOP_TRACE_CONTENT, so turning
content capture off did not suppress it.

These are unit tests over the sanitizer: they need no network and no cassette.
"""

import datetime
import json
import uuid
from decimal import Decimal
from enum import Enum
from pathlib import PurePosixPath

from opentelemetry.semconv_ai import SpanAttributes

from opentelemetry.instrumentation.langchain.callback_handler import (
    _sanitize_metadata_value,
)

MARKER = "metadata-object-marker-9f3a"


class _ClientLikeObject:
    """Stands in for a model/client object whose repr renders its config."""

    def __init__(self, api_key: str) -> None:
        """Store the marker the way a real client stores its credential."""
        self.api_key = api_key

    def __repr__(self) -> str:
        """Render the credential, as a real client's repr does."""
        return f"_ClientLikeObject(api_key='{self.api_key}')"


def test_primitives_are_preserved():
    """The documented use case - string and numeric labels - keeps working."""
    assert _sanitize_metadata_value("12345") == "12345"
    assert _sanitize_metadata_value(42) == 42
    assert _sanitize_metadata_value(1.5) == 1.5
    assert _sanitize_metadata_value(True) is True
    assert _sanitize_metadata_value(b"bytes") == b"bytes"


def test_falsy_primitives_are_preserved_not_dropped():
    """0, False and "" are values, not absences."""
    assert _sanitize_metadata_value(0) == 0
    assert _sanitize_metadata_value(False) is False
    assert _sanitize_metadata_value("") == ""


class _Tier(Enum):
    """A caller-defined enum, the kind that shows up as a metadata label."""

    GOLD = "gold"


def test_stdlib_scalars_are_stringified_not_dropped():
    """A UUID session_id, a timestamp, a Decimal, an Enum carry no credential."""
    session_id = uuid.uuid4()
    assert _sanitize_metadata_value(session_id) == str(session_id)
    assert _sanitize_metadata_value(datetime.datetime(2026, 9, 16, 12, 0)) == (
        "2026-09-16 12:00:00"
    )
    assert _sanitize_metadata_value(Decimal("3.14")) == "3.14"
    assert _sanitize_metadata_value(_Tier.GOLD) == str(_Tier.GOLD)
    assert _sanitize_metadata_value(PurePosixPath("/tmp/x")) == "/tmp/x"


def test_object_is_dropped_not_stringified():
    """The leak: an object's repr must never become the attribute value."""
    assert _sanitize_metadata_value(_ClientLikeObject(MARKER)) is None


def test_object_inside_a_list_is_dropped():
    """A sequence keeps its scalar elements and loses its object elements."""
    session_id = uuid.uuid4()
    value = _sanitize_metadata_value(["ok", session_id, _ClientLikeObject(MARKER)])
    assert value == ["ok", str(session_id)]
    assert MARKER not in str(value)


def test_plain_dict_is_kept_as_json():
    """A mapping of plain data stays, encoded as JSON rather than a Python repr."""
    value = _sanitize_metadata_value({"tenant": "acme", "retries": 2})
    assert value == '{"tenant": "acme", "retries": 2}'


def test_dict_loses_only_its_object_keys():
    """One bad key must not discard its siblings, as the list branch doesn't."""
    session_id = uuid.uuid4()
    value = json.loads(
        _sanitize_metadata_value(
            {"tenant": "acme", "sid": session_id, "client": _ClientLikeObject(MARKER)}
        )
    )
    assert value == {"tenant": "acme", "sid": str(session_id)}


def test_dict_of_only_objects_is_dropped():
    """Nothing left to record means no attribute, not an empty one."""
    assert _sanitize_metadata_value({"client": _ClientLikeObject(MARKER)}) is None


def test_object_never_reaches_association_properties(instrument_legacy, span_exporter):
    """End to end through the callback handler: the object key never appears.

    Only association properties are checked. The caller's metadata is also
    dumped onto ``traceloop.entity.input``, marker and all, but that path is
    gated by TRACELOOP_TRACE_CONTENT and dumps the whole input wholesale --
    content capture working as documented, not the ungated trace-wide leak
    this fix is about.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda

    chain = ChatPromptTemplate.from_messages(
        [("user", "{question}")]
    ) | RunnableLambda(lambda prompt: "stubbed")

    chain.invoke(
        {"question": "hi"},
        config={
            "metadata": {
                "user_id": "12345",
                "client": _ClientLikeObject(MARKER),
            }
        },
    )

    spans = span_exporter.get_finished_spans()
    assert spans, "expected the chain invocation to be traced"

    prefix = f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}."
    properties = {
        key: value
        for span in spans
        for key, value in (span.attributes or {}).items()
        if key.startswith(prefix)
    }

    # The legitimate label still propagates to every span...
    assert properties.get(f"{prefix}user_id") == "12345"

    # ...while the object is dropped rather than recorded as its repr.
    assert f"{prefix}client" not in properties
    for key, value in properties.items():
        assert MARKER not in str(value), f"marker leaked into {key}"
