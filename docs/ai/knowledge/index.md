---
okf_version: "0.2"
type: Knowledge Bundle
title: OpenLLMetry agent knowledge
description: What we have learned about instrumenting these SDKs and maintaining this repo.
---

# OpenLLMetry agent knowledge

An [OKF](https://github.com/GoogleCloudPlatform/knowledge-catalog) bundle: one concept
per markdown file, each with frontmatter carrying provenance and expiry.

This holds **empirical** knowledge — things observed to be true that could stop being
true. Imperative instructions belong in [AGENTS.md](../../../AGENTS.md); procedures
belong in [../procedures/](../procedures/README.md).

## Reading this

Check `stale_after` before relying on a concept. A concept past its date describes how
something behaved at the time of writing and may no longer hold — verify before acting
on it.

Check `verified`. A concept with no `verified` entry was written by an agent and not yet
confirmed by a human. Trust tiers: unverified → machine-confirmed → human-reviewed.

## Concepts

| Concept | About |
|---|---|
| [otel-genai-enums-are-open.md](otel-genai-enums-are-open.md) | Why an unlisted enum value is not a violation |
| [semconv-requirement-levels-are-weak.md](semconv-requirement-levels-are-weak.md) | Why the contract alone cannot catch a dropped attribute |
| [vcr-cassette-reuse-ignores-body.md](vcr-cassette-reuse-ignores-body.md) | How a borrowed cassette silently goes stale |

## Adding a concept

One concept per file. Frontmatter must include a non-empty `type`. Include `generated`
(who wrote it, when) and `stale_after` where the fact can rot. Record the addition in
[log.md](log.md). Add a row above.

Write what was **observed and non-obvious**. Do not restate what the code, README, or
git history already says.
