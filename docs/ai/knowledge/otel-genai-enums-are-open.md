---
type: Convention Behaviour
title: OTel GenAI enums are open, not closed
description: An attribute value outside an enum's declared members is legal; treating enums as closed traps most providers.
tags: [semconv, conformance, gen_ai]
generated:
  by: claude-opus/5
  at: 2026-08-03T00:00:00Z
sources:
  - author: human:gal
    note: Confirmed while reviewing the spec-1 conformance harness.
status: stable
stale_after: 2027-02-01
---

# OTel GenAI enums are open, not closed

`gen_ai.provider.name` is `required` and declares 16 members: `openai`, `gcp.gen_ai`,
`gcp.vertex_ai`, `gcp.gemini`, `anthropic`, `cohere`, `azure.ai.inference`,
`azure.ai.openai`, `ibm.watsonx.ai`, `aws.bedrock`, `perplexity`, `x_ai`, `deepseek`,
`groq`, `mistral_ai`, `moonshot_ai`.

Many providers we instrument — ollama, together, replicate, writer, voyageai, alephalpha —
have **no member at all**. Treating the enum as closed traps them: emitting their own name
is a violation, and omitting the attribute is also a violation, with no legal option.

OTel semantic-convention enums are extensible by design. Evidence from the registry
itself: `error.type` declares exactly one member (`_OTHER`) while its own `examples`
are `timeout`, `java.net.UnknownHostException`, `500`.

**Consequence for the conformance harness:** an unlisted value produces
`unknown_enum_value`, which is deliberately excluded from `BLOCKING_KINDS`. It is
reported so a genuine typo is still visible, but it never fails a build.

**Caveat:** weaver's resolved JSON carries no extensibility metadata — the `type` dict
contains only `members` — so openness cannot be read from the registry. It is applied
uniformly. If upstream ever adds an extensibility flag, this should be revisited.
