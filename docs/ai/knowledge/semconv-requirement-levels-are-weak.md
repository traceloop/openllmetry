---
type: Convention Behaviour
title: Contract requirement levels alone cannot catch a dropped attribute
description: Only ~9% of contract attributes are required, so packages must declare their own expected set.
tags: [semconv, conformance, testing]
generated:
  by: claude-opus/5
  at: 2026-08-03T00:00:00Z
sources:
  - author: human:gal
    note: Measured against the pinned registry during spec-1 review.
status: stable
stale_after: 2027-02-01
---

# Contract requirement levels alone cannot catch a dropped attribute

Measured against the pinned registry (18 span groups, 372 attribute entries):

| Level | Entries | Presence enforced? |
|---|---|---|
| `required` | 33 (8.9%) | yes |
| `conditionally_required` | 118 (31.7%) | no — the condition is prose the harness cannot evaluate |
| `recommended` | 167 (44.9%) | no |
| `opt_in` | 54 (14.5%) | no |

Only 7 of 80 unique attribute names are ever presence-checked by the contract alone:
`aws.bedrock.guardrail.id`, `gen_ai.operation.name`, `gen_ai.provider.name`,
`gen_ai.request.model`, `gen_ai.response.id`, `gen_ai.tool.name`, `mcp.method.name`.
A package emitting nothing but `gen_ai.operation.name` and `gen_ai.provider.name` passes
as fully conformant.

Critically, `gen_ai.response.finish_reasons` is `recommended` in **all 9** groups that
declare it. Issue #4362 — streaming dropping `finish_reasons` — is therefore invisible to
a contract-levels-only check.

**Consequence:** each package declares an `EXPECTED` frozenset in its
`test_conformance.py` (at `tests/test_conformance.py`, or `tests/traces/test_conformance.py`
for the batch-3 packages) naming the attributes it promises to emit. A missing promise
is a blocking `missing_expected` violation regardless of the registry's level. The
contract supplies the vocabulary; the package supplies the commitment.

A name in `EXPECTED` that the contract does not declare raises `ValueError` — a typo
must fail loudly rather than silently check nothing.
