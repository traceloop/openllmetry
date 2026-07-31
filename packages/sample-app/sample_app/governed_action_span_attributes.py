"""
Example: attach governed agent action metadata to OpenTelemetry spans.

These attributes are illustrative custom span attributes. They can help connect
an agent's proposed action to the governance decision, approval state, proof
record, and an external verifier reference without requiring a vendor-specific
collector or backend.
"""

from opentelemetry import trace
from traceloop.sdk import Traceloop


ACTION_ATTRIBUTES = {
    "gen_ai.agent.action.ref": "tool:crm.update_customer",
    "gen_ai.agent.action.hash": (
        "sha256:6f1f2d8a3c8c9e7a0b4d5e6f7890abcd1234567890abcdef1234567890abcdef"
    ),
    "governance.verdict": "allow",
    "governance.approval.status": "approved",
    "governance.proof.url": "https://governance.example/proofs/run-9b7c/action-42",
    "governance.external_verifier.ref": "verifier://policy-engine/prod/decision-42",
}


def main():
    Traceloop.init(
        app_name="governed-action-span-attributes",
        disable_batch=True,
        instruments={},
    )

    tracer = trace.get_tracer("sample_app.governed_action_span_attributes")

    with tracer.start_as_current_span("agent.action") as span:
        for key, value in ACTION_ATTRIBUTES.items():
            span.set_attribute(key, value)

        # Run the approved action here. OSuite or another governance emitter can
        # provide these values before the action is executed.
        print("Recorded governed action metadata on the current span.")


if __name__ == "__main__":
    main()
