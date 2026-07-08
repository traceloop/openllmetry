"""
Unit tests for TraceloopCallbackHandler OTel context-token lifecycle.

These tests do NOT use VCR cassettes; they exercise context-management
logic in isolation by calling internal methods directly.

No real HTTP is performed — the Tracer is backed by an InMemorySpanExporter.
"""
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from opentelemetry import context as context_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    LLMRequestTypeValues,
)

from opentelemetry.instrumentation.langchain.callback_handler import (
    TraceloopCallbackHandler,
)


@pytest.fixture
def handler():
    """Create a callback handler backed by an in-memory tracer for lifecycle tests."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    return TraceloopCallbackHandler(
        tracer=tracer,
        duration_histogram=MagicMock(),
        token_histogram=MagicMock(),
    )


@pytest.fixture(autouse=True)
def restore_otel_context():
    """
    Snapshot the OTel context before each test and restore it afterwards.

    context_api.attach(ctx) stores the *current* context in the returned token.
    context_api.detach(token) calls ContextVar.reset(token), which unconditionally
    resets the ContextVar to the snapshot saved in the token — regardless of any
    intermediate attaches that happened while the test ran.  This ensures leaked
    tokens from a failing test cannot pollute subsequent tests.
    """
    restore_token = context_api.attach(context_api.get_current())
    yield
    context_api.detach(restore_token)


def _suppression_active() -> bool:
    """Return whether language-model suppression is currently active in OTel context."""
    return context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY) is True


def _association_properties() -> dict:
    """Return the current association properties context payload, if any."""
    return context_api.get_value("association_properties") or {}


# ---------------------------------------------------------------------------
# Ordering fix (commit 2) — normal single-call lifecycle
# ---------------------------------------------------------------------------

def test_suppression_active_after_create_llm_span(handler):
    """After a normal _create_llm_span call the suppression flag must be set."""
    run_id = uuid4()
    assert not _suppression_active(), "precondition: no suppression before call"

    handler._create_llm_span(run_id, None, "gpt-4", LLMRequestTypeValues.CHAT)

    assert _suppression_active(), (
        "Suppression must be active so downstream OpenAI/Bedrock instrumentation "
        "is skipped for this LLM call."
    )

    span = handler.spans[run_id].span
    handler._end_span(span, run_id)


def test_suppression_cleared_after_end_span(handler):
    """After _end_span the suppression flag must be gone (normal single-call lifecycle)."""
    run_id = uuid4()
    handler._create_llm_span(run_id, None, "gpt-4", LLMRequestTypeValues.CHAT)

    span = handler.spans[run_id].span
    handler._end_span(span, run_id)

    assert not _suppression_active(), (
        "Suppression must be cleared when the LLM span ends; otherwise every "
        "subsequent call in this thread/task is permanently suppressed."
    )


def test_association_properties_cleared_after_end_span(handler):
    """Metadata-specific association properties must not outlive the span."""
    run_id = uuid4()
    assert _association_properties() == {}, "precondition: no association properties"

    span = handler._create_span(
        run_id,
        None,
        "test-span",
        metadata={"user_id": "12345"},
    )

    assert _association_properties().get("user_id") == "12345"

    handler._end_span(span, run_id)

    assert _association_properties() == {}, (
        "association_properties must be detached when the span ends; otherwise "
        "later spans can inherit stale metadata."
    )


# ---------------------------------------------------------------------------
# P2 — duplicate run_id leaks supp_token_1 (issue #3957)
# ---------------------------------------------------------------------------

def test_duplicate_run_id_leaks_suppression_token(handler):
    """
    Regression test for issue #3957.

    When _create_llm_span is called a second time with the same run_id,
    _create_span (called first inside _create_llm_span) unconditionally
    overwrites self.spans[run_id] before _create_llm_span can read and
    detach the old SpanHolder's suppression token.

    Context-stack trace (each arrow = context_api.attach / detach)
    ---------------------------------------------------------------
    ctx_0            baseline (no suppression)
      ↓ _create_span (1st call): attach span_ctx_1
    span_ctx_1       {SPAN_KEY: span_1}          span_token_1 remembers ctx_0
      ↓ detach span_token_1  (commit-2 ordering fix)
    ctx_0
      ↓ attach suppression
    supp_ctx_1       {SUPPRESS: True}             supp_token_1 remembers ctx_0
                     ← stored in SpanHolder; 1st _create_llm_span returns

      ↓ _create_span (2nd call, same run_id!): attach span_ctx_2
    span_ctx_2       {SUPPRESS:True, SPAN_KEY: span_2}  span_token_2 remembers supp_ctx_1
                     _create_span writes self.spans[run_id] = SpanHolder(span_2, span_token_2)
                     supp_token_1 is now GONE from self.spans — unreachable forever
      ↓ _create_llm_span reads existing_holder → finds SpanHolder(span_2, span_token_2)
      ↓ detach span_token_2
    supp_ctx_1       {SUPPRESS: True}             NOT ctx_0 — 1st suppression lives on
      ↓ attach new suppression
    supp_ctx_2       {SUPPRESS: True}             supp_token_2 remembers supp_ctx_1
                     stored in SpanHolder; 2nd _create_llm_span returns

      ↓ _end_span: detach supp_token_2
    supp_ctx_1       {SUPPRESS: True}             ← BUG: supp_token_1 never detached

    Expected after _end_span: ctx_0 (no suppression)
    Actual   after _end_span: supp_ctx_1 ({SUPPRESS: True})

    Root cause
    ----------
    The fix must capture `existing_holder = self.spans.get(run_id)` BEFORE
    calling `_create_span`, because _create_span unconditionally overwrites
    self.spans[run_id] at line 334, destroying the reference to supp_token_1.

    Correct fix in _create_llm_span (before the _create_span call):

        old_holder = self.spans.get(run_id)
        if old_holder is not None and old_holder.token is not None:
            self._safe_detach_context(old_holder.token)

        span = self._create_span(run_id, ...)   # now safe to overwrite
    """
    run_id = uuid4()
    assert not _suppression_active(), "precondition: no suppression before test"

    # First call — normal; establishes supp_token_1
    handler._create_llm_span(run_id, None, "gpt-4", LLMRequestTypeValues.CHAT)
    assert _suppression_active(), "suppression active after 1st call (sanity)"

    # Second call with the SAME run_id — triggers issue #3957
    handler._create_llm_span(run_id, None, "gpt-4", LLMRequestTypeValues.CHAT)
    assert _suppression_active(), "suppression active after 2nd call (sanity)"

    # End the one surviving span (from the 2nd call)
    span = handler.spans[run_id].span
    handler._end_span(span, run_id)

    # After all spans are done the suppression MUST be cleared.
    # P2 bug present → still True  (supp_token_1 leaked, never detached)
    # P2 bug fixed   → None        (both tokens properly cleaned up)
    assert not _suppression_active(), (
        "Issue #3957 not fixed: suppression flag is still set after _end_span. "
        "supp_token_1 from the first _create_llm_span call was never detached because "
        "_create_span overwrites self.spans[run_id] before the old holder can be read. "
        "Fix: move `existing_holder = self.spans.get(run_id)` (and its detach) to "
        "BEFORE the `_create_span(...)` call in _create_llm_span."
    )


def test_duplicate_run_id_replaces_association_properties(handler):
    """
    Replacing a SpanHolder for the same run_id must clean up the old metadata context.
    """
    run_id = uuid4()

    first_span = handler._create_span(
        run_id,
        None,
        "first-span",
        metadata={"user_id": "12345"},
    )
    assert _association_properties().get("user_id") == "12345"

    second_span = handler._create_span(
        run_id,
        None,
        "second-span",
        metadata={"request_id": "req-1"},
    )

    assert first_span.end_time is None, "sanity: old span stays open until caller ends it"
    assert _association_properties().get("user_id") is None, (
        "The previous association_properties context should be detached before "
        "creating a replacement holder for the same run_id."
    )
    assert _association_properties().get("request_id") == "req-1"

    handler._end_span(second_span, run_id)

    assert _association_properties() == {}, (
        "association_properties from the replacement span should also be cleaned up "
        "when the surviving holder is ended."
    )


# ---------------------------------------------------------------------------
# Issue #3526 — orphaned context_api.attach() in on_chain_end corrupts context stack
# ---------------------------------------------------------------------------

def test_on_chain_end_does_not_leak_context_frame(handler):
    """
    Regression test for issue #3526.

    Before the fix, on_chain_end() contained an orphaned context_api.attach() call
    for root chains (parent_run_id=None):

        context_api.attach(context_api.set_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, False))

    No token was saved, so the frame was *never detached*.  The OTel context stack
    grew by one entry on every root chain completion — permanently polluting the
    context for the lifetime of the thread/task.

    Observable effect: after on_chain_end the key is False (explicitly set) rather
    than None (absent/unset), and it stays False for all subsequent work in the
    same execution context.

    The fix removes the orphaned attach entirely — _end_span() already detaches the
    suppression token stored in the SpanHolder, which restores the pre-chain context.
    """
    chain_run_id = uuid4()
    llm_run_id = uuid4()

    assert context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY) is None, (
        "precondition: suppression key must be absent before test"
    )

    # Simulate on_chain_start for a root chain
    handler.on_chain_start(
        serialized={"name": "TestChain"},
        inputs={"input": "hello"},
        run_id=chain_run_id,
        parent_run_id=None,
    )

    # Simulate an LLM call inside the chain (sets suppression)
    handler._create_llm_span(llm_run_id, chain_run_id, "gpt-4", LLMRequestTypeValues.CHAT)

    assert _suppression_active(), "sanity: suppression must be active during LLM span"

    # End the LLM span
    llm_span = handler.spans[llm_run_id].span
    handler._end_span(llm_span, llm_run_id)

    # End the root chain — this is where the orphaned attach fires in the buggy code
    handler.on_chain_end(
        outputs={"output": "result"},
        run_id=chain_run_id,
        parent_run_id=None,
    )

    # The suppression key must be truly absent (None), not just falsy (False).
    # Bug present  → value is False (orphaned attach pushed False onto the stack and never popped)
    # Bug fixed    → value is None  (key is absent; _end_span restored the baseline context)
    raw_value = context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY)
    assert raw_value is None, (
        f"Issue #3526 not fixed: SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY is "
        f"{raw_value!r} after on_chain_end, expected None. "
        "The orphaned context_api.attach() in on_chain_end pushed False onto the context "
        "stack without saving a token, so it was never detached. "
        "Fix: remove the orphaned attach — _end_span() already restores the context."
    )


def test_on_chain_end_context_stack_does_not_accumulate(handler):
    """
    Complementary check for issue #3526: running N root chains must not grow the
    context stack.  Detected by checking the suppression key stays None after each
    completion and never becomes False from a previous chain's leaked frame.
    """
    for i in range(3):
        chain_run_id = uuid4()

        handler.on_chain_start(
            serialized={"name": f"Chain{i}"},
            inputs={"input": "x"},
            run_id=chain_run_id,
            parent_run_id=None,
        )
        handler.on_chain_end(
            outputs={"output": "y"},
            run_id=chain_run_id,
            parent_run_id=None,
        )

        raw_value = context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY)
        assert raw_value is None, (
            f"Issue #3526: after chain #{i + 1}, SUPPRESS key is {raw_value!r} "
            f"(expected None). Each root on_chain_end leaked a context frame."
        )


def test_duplicate_llm_run_id_replaces_association_properties(handler):
    """
    Replacing an LLM span holder must clear the prior metadata context and suppression.
    """
    run_id = uuid4()

    first_span = handler._create_llm_span(
        run_id,
        None,
        "gpt-4",
        LLMRequestTypeValues.CHAT,
        metadata={"user_id": "12345"},
    )
    assert _association_properties().get("user_id") == "12345"
    assert _suppression_active(), "suppression active after 1st call (sanity)"

    second_span = handler._create_llm_span(
        run_id,
        None,
        "gpt-4",
        LLMRequestTypeValues.CHAT,
        metadata={"request_id": "req-1"},
    )

    # The first span is intentionally not ended because the replacement path is under test.
    assert first_span.end_time is None, "sanity: old span stays open until caller ends it"
    assert _association_properties().get("user_id") is None, (
        "The previous association_properties context should be detached before "
        "creating a replacement LLM holder for the same run_id."
    )
    assert _association_properties().get("request_id") == "req-1"
    assert _suppression_active(), "suppression active after 2nd call (sanity)"

    handler._end_span(second_span, run_id)

    assert _association_properties() == {}, (
        "association_properties from the replacement LLM span should be cleaned up "
        "when the surviving holder is ended."
    )
    assert not _suppression_active(), (
        "Suppression must be cleared after ending the replacement LLM span, matching "
        "the behavior verified in test_duplicate_run_id_leaks_suppression_token."
    )
