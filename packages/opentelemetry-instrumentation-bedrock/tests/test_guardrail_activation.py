from opentelemetry.instrumentation.bedrock.guardrail import is_guardrail_activated


def test_response_without_guardrail_key_is_not_an_activation():
    # Bedrock omits amazon-bedrock-guardrailAction when no guardrail is configured.
    assert is_guardrail_activated({"stopReason": "end_turn"}) is False
    assert is_guardrail_activated({}) is False


def test_configured_guardrail_that_did_not_fire_is_not_an_activation():
    assert is_guardrail_activated({"amazon-bedrock-guardrailAction": "NONE"}) is False


def test_activations_are_detected():
    assert is_guardrail_activated({"amazon-bedrock-guardrailAction": "INTERVENED"}) is True
    assert is_guardrail_activated({"stopReason": "guardrail_intervened"}) is True
    assert (
        is_guardrail_activated({"results": [{"completionReason": "CONTENT_FILTERED"}]})
        is True
    )
