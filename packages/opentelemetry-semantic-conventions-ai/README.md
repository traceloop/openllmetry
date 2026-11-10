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
from opentelemetry.semconv_ai.delegation import delegation, hash_scope

with delegation(
    root_principal="user:42",                # who the chain acts for (an id, never a token)
    chain_id="dlg_7f3a",                     # reference to the verified delegation chain
    scope_hash=hash_scope(["crm:read", "email:send"]),
    depth=1,                                 # optional: delegation hops so far
):
    Runner.run_sync(triage_agent, "Refund order 1234")
```

| Attribute | Type | Description |
|---|---|---|
| `traceloop.delegation.root_principal` | string | Principal at the root of the delegation chain |
| `traceloop.delegation.chain_id` | string | Reference to the delegation chain, for out-of-band verification |
| `traceloop.delegation.scope_hash` | string | `<algorithm>:<hex digest>` of the granted scope |
| `traceloop.delegation.depth` | int | Number of delegation hops |

These live under the `traceloop.*` namespace until the OpenTelemetry GenAI semantic
conventions define an equivalent. The value is kept in the in-process OpenTelemetry
`Context` (not Baggage), so it is never sent to other services in request headers.
