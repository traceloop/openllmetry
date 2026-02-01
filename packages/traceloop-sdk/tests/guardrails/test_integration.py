"""Integration tests for guardrails with VCR cassettes.

These tests record and playback evaluator API requests and SSE responses
using VCR cassettes.

To record cassettes (requires TRACELOOP_API_KEY):
    cd packages/traceloop-sdk
    TRACELOOP_API_KEY="your-key" uv run pytest tests/guardrails/test_integration.py --record-mode=all -v

To playback without API key:
    uv run pytest tests/guardrails/test_integration.py -v
"""

import os
import pytest
from traceloop.sdk import Traceloop
from traceloop.sdk.guardrail import Guardrails, OnFailure, Guards
from traceloop.sdk.generated.evaluators.request import (
    PIIDetectorInput,
    ToxicityDetectorInput,
    AnswerRelevancyInput,
)


@pytest.fixture
def traceloop_client(async_http_client):
    """Initialize Traceloop client for tests.

    Uses environment variables for recording, fake values for VCR playback.
    """
    # Save the original client to restore later (Traceloop is a singleton)
    original_client = getattr(Traceloop, '_Traceloop__client', None)

    # Reset the client to allow re-initialization
    Traceloop._Traceloop__client = None

    api_key = os.environ.get("TRACELOOP_API_KEY", "fake-key-for-vcr-playback")
    base_url = os.environ.get(
        "TRACELOOP_BASE_URL",
        "https://api.traceloop.dev"
    )

    client = Traceloop.init(
        app_name="guardrail-integration-tests",
        api_key=api_key,
        api_endpoint=base_url,
        disable_batch=True,
    )

    yield client

    # Restore the original client
    Traceloop._Traceloop__client = original_client


@pytest.fixture
def guardrails(async_http_client):
    """Create guardrails instance with async HTTP client."""
    return Guardrails(async_http_client)


class TestPIIDetectorGuard:
    """Tests for PII detector evaluator as guard."""

    @pytest.mark.vcr
    @pytest.mark.anyio
    async def test_pii_detector_guard_passes_clean_text(
        self, guardrails, traceloop_client
    ):
        """PII detector guard passes when text has no PII."""
        guard = Guards.pii_detector()

        g = guardrails.create(
            guards=[guard],
            on_failure=OnFailure.raise_exception("PII detected"),
            name="pii-check",
        )

        # Clean text without PII
        passed = await g.validate([
            PIIDetectorInput(text="Hello, this is a simple message without any personal information.")
        ])

        assert passed is True

    @pytest.mark.vcr
    @pytest.mark.anyio
    async def test_pii_detector_guard_fails_with_email(
        self, guardrails, traceloop_client
    ):
        """PII detector guard fails when text contains email."""
        guard = Guards.pii_detector()

        g = guardrails.create(
            guards=[guard],
            on_failure=OnFailure.log(),
            name="pii-check",
        )

        # Text with PII (email)
        passed = await g.validate([
            PIIDetectorInput(text="Contact me at john.doe@example.com for more details.")
        ])

        assert passed is False


class TestToxicityDetectorGuard:
    """Tests for toxicity detector evaluator as guard."""

    @pytest.mark.vcr
    @pytest.mark.anyio
    async def test_toxicity_detector_guard_passes_friendly_text(
        self, guardrails, traceloop_client
    ):
        """Toxicity detector guard passes for friendly text."""
        guard = Guards.toxicity_detector()

        g = guardrails.create(
            guards=[guard],
            on_failure=OnFailure.raise_exception("Toxic content"),
            name="toxicity-check",
        )

        # Friendly text
        passed = await g.validate([
            ToxicityDetectorInput(text="Thank you for helping me today! I really appreciate your assistance.")
        ])

        assert passed is True

    @pytest.mark.vcr
    @pytest.mark.anyio
    async def test_toxicity_detector_guard_fails_toxic_text(
        self, guardrails, traceloop_client
    ):
        """Toxicity detector guard fails for toxic text."""
        guard = Guards.toxicity_detector()

        g = guardrails.create(
            guards=[guard],
            on_failure=OnFailure.log(),
            name="toxicity-check",
        )

        # Toxic text
        passed = await g.validate([
            ToxicityDetectorInput(text="You are stupid and worthless!")
        ])

        assert passed is False


class TestAnswerRelevancyGuard:
    """Tests for answer relevancy evaluator as guard."""

    @pytest.mark.vcr
    @pytest.mark.anyio
    async def test_answer_relevancy_passes_relevant_answer(
        self, guardrails, traceloop_client
    ):
        """Answer relevancy guard passes for relevant answer."""
        guard = Guards.answer_relevancy()

        g = guardrails.create(
            guards=[guard],
            on_failure=OnFailure.raise_exception("Answer not relevant"),
            name="relevancy-check",
        )

        # Relevant answer
        passed = await g.validate([
            AnswerRelevancyInput(
                question="What is the capital of France?",
                answer="The capital of France is Paris."
            )
        ])

        assert passed is True

    @pytest.mark.vcr
    @pytest.mark.anyio
    async def test_answer_relevancy_fails_irrelevant_answer(
        self, guardrails, traceloop_client
    ):
        """Answer relevancy guard fails for irrelevant answer."""
        guard = Guards.answer_relevancy()

        g = guardrails.create(
            guards=[guard],
            on_failure=OnFailure.log(),
            name="relevancy-check",
        )

        # Irrelevant answer
        passed = await g.validate([
            AnswerRelevancyInput(
                question="What is the capital of France?",
                answer="I like eating pizza on weekends."
            )
        ])

        assert passed is False


class TestMultipleGuardsValidation:
    """Tests for Guardrails.validate() with multiple guards."""

    @pytest.mark.vcr
    @pytest.mark.anyio
    async def test_validate_multiple_guards_all_pass(
        self, guardrails, traceloop_client
    ):
        """Multiple guards all pass validation."""
        pii_guard = Guards.pii_detector()
        toxicity_guard = Guards.toxicity_detector()

        g = guardrails.create(
            guards=[pii_guard, toxicity_guard],
            on_failure=OnFailure.raise_exception("Guard failed"),
            name="content-safety",
            parallel=True,
        )

        # Clean, friendly text
        passed = await g.validate([
            PIIDetectorInput(text="Welcome to our service! We hope you enjoy your experience."),
            ToxicityDetectorInput(text="Welcome to our service! We hope you enjoy your experience."),
        ])

        assert passed is True

    @pytest.mark.vcr
    @pytest.mark.anyio
    async def test_validate_multiple_guards_one_fails(
        self, guardrails, traceloop_client
    ):
        """Multiple guards where one fails validation."""
        pii_guard = Guards.pii_detector()
        toxicity_guard = Guards.toxicity_detector()

        g = guardrails.create(
            guards=[pii_guard, toxicity_guard],
            on_failure=OnFailure.log(),
            name="content-safety",
            parallel=True,
        )

        # Text with PII but not toxic
        passed = await g.validate([
            PIIDetectorInput(text="Contact john.smith@company.com for help."),
            ToxicityDetectorInput(text="Contact john.smith@company.com for help."),
        ])

        assert passed is False
