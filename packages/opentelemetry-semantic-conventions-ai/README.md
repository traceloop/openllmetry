# OpenTelemetry Semantic Conventions extensions for gen-AI applications

<a href="https://pypi.org/project/opentelemetry-semantic-conventions-ai/">
    <img src="https://badge.fury.io/py/opentelemetry-semantic-conventions-ai.svg">
</a>

This is an extension of the standard [OpenTelemetry Semantic Conventions](https://github.com/open-telemetry/semantic-conventions) for gen AI applications. It defines additional attributes for spans that are useful for debugging and monitoring prompts, completions, token usage, etc.

## Delegation / authority on agent spans (opt-in)

Multi-agent apps act on someone's behalf. `opentelemetry.semconv_ai.delegation` lets the
application declare *on whose behalf* and *under which grant* an agent run happens; the
OpenAI Agents and LangChain/LangGraph instrumentations then record it on agent and handoff
spans. Nothing is recorded unless you opt in, and only identifiers and hashes are accepted:
values that look like tokens are rejected and the scope itself is never stored.

```python
from opentelemetry.semconv_ai.delegation import delegation, hash_chain, hash_scope

with delegation(
    root_principal="user:42",                # who the chain acts for (an id, never a token)
    chain_id="dlg_7f3a",                     # reference to the verified delegation chain
    scope_hash=hash_scope(["crm:read", "email:send"]),
    depth=1,                                 # optional: delegation hops so far
    actor="agent:triage",                    # optional: the agent acting right now
    chain_hash=hash_chain(verified_hops),    # optional: digest of the verified chain
    credential_ref="txn:9c1e",               # optional: id of the credential that carried it
):
    Runner.run_sync(triage_agent, "Refund order 1234")
```

Each hop can declare its own scope. A nested block with a `scope_hash` also records the
enclosing block's `scope_hash` as `parent_scope_hash`, so a verifier that holds the scopes
can check that authority only narrowed from hop to hop:

```python
with delegation("user:42", scope_hash=hash_scope(["crm:read", "email:send"])):
    # handoff to the billing agent with a narrower grant
    with delegation(
        "user:42",
        actor="agent:billing",
        depth=2,
        scope_hash=hash_scope(["crm:read"]),
        credential_ref="spiffe://example.org/agent/billing",
    ):
        ...  # spans here carry scope_hash and parent_scope_hash
```

To record a pre-tool authorization decision, call `record_authorization` from your own
tool guard before the tool runs. It annotates the current span (or the one you pass):

```python
from opentelemetry.semconv_ai.delegation import record_authorization

if not policy.allows(tool_name, scope):
    record_authorization("deny", reason="scope_exceeded")
    raise PermissionError(tool_name)
record_authorization("allow")
```

| Attribute | Type | Description |
|---|---|---|
| `traceloop.delegation.root_principal` | string | Principal at the root of the delegation chain |
| `traceloop.delegation.chain_id` | string | Reference to the delegation chain, for out-of-band verification |
| `traceloop.delegation.scope_hash` | string | `<algorithm>:<hex digest>` of the granted scope |
| `traceloop.delegation.depth` | int | Number of delegation hops |
| `traceloop.delegation.actor` | string | Agent / service acting right now |
| `traceloop.delegation.chain_hash` | string | `<algorithm>:<hex digest>` of the verified delegation chain (order-sensitive) |
| `traceloop.delegation.parent_scope_hash` | string | `scope_hash` of the enclosing hop, recorded next to this hop's own `scope_hash` |
| `traceloop.delegation.credential_ref` | string | Opaque reference to the credential that carried the authority (e.g. a token exchange id, VAID or SPIFFE ID); never the credential itself |
| `traceloop.authz.decision` | string | `allow` or `deny`, recorded by the app's tool guard before the tool runs |
| `traceloop.authz.reason` | string | Short machine-readable code for a denial, e.g. `scope_exceeded` |

These live under the `traceloop.*` namespace until the OpenTelemetry GenAI semantic
conventions define an equivalent. The value is kept in the in-process OpenTelemetry
`Context` (not Baggage), so it is never sent to other services in request headers.
