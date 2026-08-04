"""Attributes OpenLLMetry emits in the `gen_ai.*` namespace that the contract
does not define.

Emitting an undeclared attribute under the official namespace means downstream
consumers read non-standard data as if it were conventional. Every entry here is
therefore a deliberate, documented decision — not a licence to invent more.

Adding an entry requires a rationale and, where a migration is intended, an
issue link. `test_extensions.py` fails if upstream later defines one of these,
which forces removal rather than indefinite divergence.

Prefer a non-`gen_ai.*` namespace (`traceloop.*`) for anything genuinely
OpenLLMetry-specific; the harness ignores other namespaces entirely.

Known limitation: matching is exact, so an entry here covers only that literal
attribute name. The legacy content attributes are emitted in indexed form
(`gen_ai.prompt.0.content`, `gen_ai.completion.0.role`, ...), which these entries
do NOT cover — those still surface as `undeclared_gen_ai`. Until the harness
supports prefix matching, a content-capturing package cannot reach enforcing mode
through this file alone.
"""

EXTENSION_RATIONALE = {
    "gen_ai.usage.total_tokens": (
        "Sum of input and output tokens. Upstream models the two separately and "
        "leaves the sum to consumers. Long-standing OpenLLMetry attribute; removing "
        "it is a breaking change for dashboards."
    ),
    "gen_ai.user": (
        "End-user identifier passed through from provider SDKs. No upstream "
        "equivalent in the GenAI registry at the pinned ref."
    ),
    "gen_ai.headers": (
        "Request headers captured for debugging. Opt-in only. No upstream equivalent."
    ),
    "gen_ai.is_streaming": (
        "Whether the request used streaming. Upstream defines the equivalent as "
        "`gen_ai.request.stream`, so this is a rename, not a genuine gap: migrating "
        "to the contract name is the intended path. Retained meanwhile because "
        "existing dashboards query it."
    ),
    "gen_ai.prompt": (
        "Legacy prompt content attribute, the input-side counterpart to "
        "`gen_ai.completion`, predating upstream's `gen_ai.input.messages`. Retained "
        "for backwards compatibility; migration to `gen_ai.input.messages` is the "
        "intended path."
    ),
    "gen_ai.completion": (
        "Legacy prompt/completion content attribute, predating upstream's "
        "gen_ai.output.messages. Retained for backwards compatibility; migration "
        "to gen_ai.output.messages is the intended path."
    ),
}

EXTENSIONS = frozenset(EXTENSION_RATIONALE)

__all__ = ["EXTENSIONS", "EXTENSION_RATIONALE"]
