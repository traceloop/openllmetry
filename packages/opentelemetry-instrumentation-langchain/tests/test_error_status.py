"""Regression tests for span status handling when a LangChain callback errors.

Covers a subtle bug in `TraceloopCallbackHandler._handle_error` where the
error message was passed as the second positional argument to
`Span.set_status(...)` alongside a `Status` object. The OTel SDK contract
silently drops the second argument in that case AND emits a warning,
leaving the span without its human-readable error description.

Correct call: ``span.set_status(Status(StatusCode.ERROR, str(error)))``.
"""

import logging

from langchain_core.runnables import RunnableLambda
from opentelemetry.trace import StatusCode


def _explode(_):
    raise ValueError("boom: something went wrong")


def test_chain_error_sets_status_description(instrument_legacy, span_exporter, caplog):
    """When a chain raises, the span should be ERROR with the message as description.

    Pre-fix, ``span.set_status(Status(StatusCode.ERROR), str(error))`` made the
    SDK log a ``"Description ... ignored. Use either Status or
    (StatusCode, Description)"`` warning AND drop the description. This test
    pins the post-fix behaviour: status description is set, no warning emitted.
    """
    # `span_exporter` is session-scoped (see conftest.py), so spans from earlier
    # tests accumulate. Clear before invoking the chain so the assertions below
    # only see spans produced by this test.
    span_exporter.clear()

    chain = RunnableLambda(_explode)

    with caplog.at_level(logging.WARNING, logger="opentelemetry.sdk.trace"):
        try:
            chain.invoke("anything")
        except ValueError:
            pass  # expected

    spans = span_exporter.get_finished_spans()
    assert spans, "expected at least one span from the failing chain"

    error_spans = [s for s in spans if s.status.status_code == StatusCode.ERROR]
    assert error_spans, (
        "expected at least one span marked ERROR; got statuses "
        f"{[s.status.status_code for s in spans]}"
    )

    # The bug: pre-fix, status.description was None because the SDK dropped it.
    for span in error_spans:
        assert span.status.description == "boom: something went wrong", (
            f"span {span.name!r} has description={span.status.description!r}; "
            "expected the exception message to be preserved on the status."
        )

    # The bug also emits an SDK warning. After the fix, no such warning fires.
    bad_warnings = [
        rec
        for rec in caplog.records
        if rec.name == "opentelemetry.sdk.trace"
        and "ignored" in rec.getMessage()
        and "Use either" in rec.getMessage()
    ]
    assert not bad_warnings, (
        "OTel SDK emitted a 'Description ... ignored' warning, meaning "
        "set_status() was called with both a Status object and a separate "
        "description argument. Messages: "
        f"{[r.getMessage() for r in bad_warnings]}"
    )
