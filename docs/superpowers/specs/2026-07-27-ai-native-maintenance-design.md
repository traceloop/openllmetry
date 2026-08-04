# AI-Native Maintenance System — Design

**Date:** 2026-07-27
**Status:** Approved, pending implementation plans
**Repo:** `traceloop/openllmetry`

## Goal

Make the repository maintainable primarily by AI agents: agents take issues, produce
verified fixes, and iterate on review feedback. Humans approve PRs and make product
decisions. Nothing else is delegated away from humans.

## Repository baseline (measured 2026-07-27)

| Fact | Value |
|---|---|
| Python packages under Nx + uv | 35 (32 instrumentations) |
| Open issues | 110 (22 opened in last 90d, 39 closed) |
| Open PRs | 499 (33 merged in last 90d) |
| Packages with any semconv-compliance test | 5 of 32 |
| Packages dual-emitting `gen_ai.*` and legacy `llm.*`/`traceloop.*` | 21 |
| References to upstream weaver / `semantic-conventions-genai` | 0 |
| Existing agent config | one flat `CLAUDE.md`; no skills, agents, or AI workflows |

Two findings drive the design:

**Issue inflow is not the constraint.** ~7 issues/month arrive; the PR queue is 499 deep
and drains at ~11/month. Autonomous issue-solving adds supply to the flooded side of the
pipeline. The system must therefore be verification-heavy, throughput-limited, and must
not enable auto-merge.

**The semconv contract is enforced in 15% of the places it applies.** Open bugs #4362
(anthropic drops `finish_reasons`), #4192 (ollama missing inference metadata), #4234
(bedrock wrong vendor attribute) and #4378 (agent name leaking) are not four bugs. They
are one bug — an unenforced contract — reported four times. Fixing them individually is
32x the work of making the contract executable.

## Decisions

| Decision | Choice |
|---|---|
| First target | Knowledge substrate + autonomous issue→PR authoring |
| Runtime | GitHub Actions, in-repo, versioned |
| Cassette policy | Agent records live, behind a fail-closed scrubbing gate |
| Substrate form | Executable contract + layered docs/skills + accumulated lessons |
| Autonomy gate | Triage agent tiers issues; only tier 1 is auto-attempted |
| Contract source | `github.com/open-telemetry/semantic-conventions-genai`, pinned SHA |
| Fix methodology | Mandatory TDD, mechanically enforced RED→GREEN |
| Knowledge portability | Agent-agnostic: `AGENTS.md` canonical, OKF v0.2 for empirical knowledge |

## Architecture

Three layers. Governing principle: **knowledge that can fail CI beats knowledge written
as prose.**

### Layer 1 — Knowledge substrate

```
.semconv/
  versions.env              SEMCONV_GENAI_REF=<pinned sha>   # deliberate bumps only
  registry/                 vendored model/ from semantic-conventions-genai
packages/opentelemetry-semantic-conventions-ai/
  .../_generated/contract.py    weaver-generated: required/recommended attrs per
                                span kind, enum values, payload JSON Schemas
  .../conformance.py            hand-written harness consuming contract.py
AGENTS.md                   canonical instructions; CLAUDE.md is a symlink to it
packages/*/AGENTS.md        per-package: layout, commands, conventions
docs/ai/procedures/*.md     plain-markdown procedures, readable by any agent:
                            fix-instrumentation-bug, add-instrumentation,
                            record-cassette, triage-issue, semconv-conformance
.claude/skills/*/SKILL.md   THIN WRAPPERS pointing at docs/ai/procedures/.
                            No content of their own.
docs/ai/knowledge/          OKF v0.2 bundle: index.md, log.md, one concept per file.
                            Accumulated lessons + empirical SDK behaviour, with
                            provenance, trust tier, and stale_after.
```

The substrate is deliberately portable across agent implementations — see
[Agent portability](#agent-portability) below. No knowledge lives in a Claude-Code-specific
file; `.claude/` contains only pointers.

### Layer 2 — Agent roles (`.claude/agents/`)

| Agent | Responsibility |
|---|---|
| `triage` | Classify tier 1/2/3, dedupe, write repro hypothesis, label package |
| `fixer` | Failing test → fix → self-verify → draft PR |
| `recorder` | **Isolated.** Holds provider keys. Never reads issue or PR text. |
| `reviewer` | Adversarial pass attempting to refute the fix, before a human sees it |
| `responder` | Applies maintainer review feedback; writes the OKF lesson concept |

`recorder` is a separate agent by architectural necessity, not convenience. It is the only
component holding live provider keys and must be structurally incapable of reading
attacker-controlled issue text. `fixer` declares the interaction it needs via a committed
`.recording-manifest.yaml`; `recorder` reads only that manifest. Collapsing the two agents
reopens the exfiltration path.

### Layer 3 — Orchestration (`.github/workflows/`)

| Workflow | Trigger |
|---|---|
| `ai-triage.yml` | `issues[opened, reopened]` |
| `ai-fix.yml` | label `ai:tier1` |
| `ai-record.yml` | label `ai:needs-recording` — **GitHub Environment, human-approved** |
| `ai-review.yml` | PR ready / fixer push |
| `ai-review-response.yml` | `pull_request_review[submitted]` with `changes_requested` |
| `ai-conformance-sweep.yml` | `schedule` |

## The contract

The conformance suite is **generated from upstream, not written by hand.**

Upstream `semantic-conventions-genai` is a Weaver registry (`WEAVER_VERSION=v0.25.0`)
containing `registry.yaml`, `spans.yaml`, `metrics.yaml`, `events.yaml`, and JSON Schemas
for payload shapes (`gen-ai-input-messages.json`, `gen-ai-output-messages.json`,
`gen-ai-tool-definitions.json`, …). It also ships per-provider docs (`anthropic.md`,
`openai.md`, `aws-bedrock.md`, `mcp.md`, `azure-ai-inference.md`) mapping nearly 1:1 onto
openllmetry packages. It declares itself the canonical home for GenAI conventions,
superseding the gen-ai directories in the main semconv repo.

`_generated/contract.py` is **committed to the repo**, not generated at test time. CI
regenerates and fails on diff. This makes every contract change a reviewable line-level
diff rather than an invisible behavioural shift, and keeps test runs offline.

Each instrumentation package gets one test importing the shared harness, parametrized over
the spans it emits. The harness validates emitted attributes against the generated
contract and validates message payloads against upstream's JSON Schemas directly.

Why generation:

- Four reported bugs plus the ~28 unreported instances collapse into one red test.
- Upstream drift becomes a reviewable PR: bump pin → regenerate → CI diff names exactly
  which packages fell out of compliance.
- The `gen_ai.*` vs `llm.*` dual-emit migration gets an end condition: assert-required on
  the contract namespace, warn-only on legacy.

Two hard constraints:

- Upstream is `stability: development`, `schema_url: .../gen-ai-dev/1.42.0-dev`, with **no
  tags**. Pin a SHA and bump deliberately. Tracking `main` would red-CI all 32 packages on
  a third party's merge.
- The harness lands **warn-only**, flipping to enforcing per-package as each is fixed.
  Enabling enforcement repo-wide on day one red-CIs the entire repository. Enforcement
  state is declared per package in its `project.json` under
  `tags: ["semconv:enforcing"]` vs `tags: ["semconv:warn"]`, so the migration frontier is
  visible in one `grep` and flipping a package is a one-line reviewable change.

## Agent portability

Requirement: the knowledge base must be usable by agents other than Claude Code. No
knowledge may be locked in a vendor-specific file.

Mechanisms, layer by layer:

| Layer | Portability mechanism |
|---|---|
| Contract | A pytest suite. Universally runnable; needs no agent at all. |
| Instructions | `AGENTS.md` is canonical. `CLAUDE.md` is a **symlink** (git mode `120000`). |
| Procedures | Plain markdown in `docs/ai/procedures/`. `.claude/skills/*/SKILL.md` are thin wrappers that point at them and hold no content. |
| Empirical knowledge | OKF v0.2 bundle — a vendor-neutral spec, and plain markdown regardless. |

The symlink approach is not invented here: upstream `semantic-conventions-genai` already
ships `AGENTS.md` as a regular file with `CLAUDE.md` as a 9-byte symlink to it. Adopting
the same layout keeps us consistent with the contract repo.

### Why OKF, and where it does not apply

OKF (Open Knowledge Format, Google, published 2026-06-12, Apache-2.0, spec v0.2) is a
directory of markdown files with YAML frontmatter, one concept per file. It is used for
**empirical** knowledge only — accumulated lessons and observed SDK behaviour. Imperative
content ("run `nx run <pkg>:test`") stays in `AGENTS.md`; declarative content ("Anthropic
returns an empty content block on tool-only turns") becomes an OKF concept.

The split is *imperative → `AGENTS.md`, empirical → OKF.*

OKF fields carry real weight for this use case rather than being ceremony:

| Field | Use here |
|---|---|
| `generated: {by, at}`, `verified: []` | Trust tiers *unverified → machine-confirmed → human-reviewed*, matching the rule that a `responder`-written lesson is untrusted until a human approves its PR |
| `stale_after` | SDK-behaviour facts rot on SDK bumps; mechanical expiry replaces a rotting doc pile |
| `sources` | Links a lesson back to the review comment that produced it |
| `status: draft\|stable\|deprecated` | Lesson lifecycle |
| Actor convention `<producer>/<version>`, `human:<id>` | Attribution once multiple agent types write to the bundle |

Risks, accepted knowingly: the spec is ~6 weeks old at v0.2 and will churn, and no agent
has native OKF support — pointing agents at the bundle is required. Both are acceptable
because the failure mode is benign: an abandoned OKF is still a directory of markdown with
sensible frontmatter. That asymmetry is the reason to adopt it.

### Runner indirection

Workflows invoke agents through a single thin composite action rather than calling a
vendor action inline at six call sites. Swapping or adding an agent implementation is then
one file, not a rewrite. The knowledge base being portable is worthless if the
orchestration hard-codes one vendor in every workflow.

## The TDD gate

"Write a failing test first" as a prompt instruction is unverifiable — the agent will claim
compliance and CI cannot tell. It is enforced structurally instead.

`fixer` must produce exactly two commits:

```
test: reproduce #4362 — streaming drops finish_reasons      # test files only
fix(anthropic): emit finish_reasons on empty-content turns  # source files only
```

CI enforces:

| Step | Check | On violation |
|---|---|---|
| 1 | Checkout **test commit only**, run the new test | must **FAIL** — else block: test reproduces nothing |
| 2 | Capture failure output verbatim | posted into PR body as RED evidence |
| 3 | Run new test at HEAD | must **PASS** |
| 4 | Run full affected suite at HEAD | must **PASS** — no collateral breakage |
| 5 | Commit 1 touches no `packages/*/opentelemetry/**` | else block: fix smuggled into test commit |

Step 1 carries the weight. It catches the three ways autonomous agents fake completion: a
test that asserts nothing, a test that passes trivially, and a "fix" for a bug that was
never real. Human review then begins from proof the bug existed rather than the agent's
claim that it did.

## Flows

### Flow A — issue to merged PR

1. `issues[opened]` → `triage`: dedupe against open and closed-in-90d; classify.
   Tier 1 (mechanical: a defect or enhancement whose correct behaviour is determined by
   the semconv contract, an existing test, or explicit upstream SDK behaviour) gets
   `ai:tier1` + `pkg:<name>` + a repro hypothesis. Tier 2
   (needs maintainer decision) gets the options drafted and the maintainer @-mentioned —
   **no PR**. Tier 3 is answered or closed.
2. `ai:tier1` → `fixer`: loads root `AGENTS.md` → package `AGENTS.md` → the relevant
   procedure in `docs/ai/procedures/` → `docs/ai/knowledge/index.md`, retrieving concepts
   tagged for the affected package and ignoring any whose `stale_after` has passed.
   Writes commit 1 (failing test). If a new cassette is required, writes
   `packages/<pkg>/tests/.recording-manifest.yaml` — declaring the target module, call,
   and arguments to record, and nothing free-form — applies `ai:needs-recording`, and
   stops at draft.
   Otherwise writes commit 2 and opens a draft PR.
3. `ai:needs-recording` → `recorder` in a human-approved GitHub Environment: reads only the
   manifest, records with live keys, runs the deterministic scrubber, secret-scans
   (fail closed), commits the cassette, re-triggers `fixer`.
4. TDD gate + normal CI.
5. `reviewer` attempts to refute the fix. A hole found means comment and re-trigger
   `fixer`. Clean means the PR goes ready-for-review and the maintainer is requested.
6. Maintainer approves and merges, or requests changes.

### Flow B — review feedback to durable lesson

`changes_requested` → `responder` applies the change, preserving test-first discipline for
any new behavior, then classifies the comment as one-off or generalizable. Generalizable
feedback becomes an OKF concept under `docs/ai/knowledge/` **in the same PR**, written with
`generated: {by: <agent>/<version>, at: <ts>}` and no `verified` entry. Merging the PR is
what adds the `verified: [{by: human:<maintainer>, at: ...}]` entry, promoting the concept
to the human-reviewed trust tier. `log.md` records the addition.

So knowledge enters the substrate only through normal human review, and its trust level is
recorded in the file rather than assumed. Nothing self-modifies unreviewed. This loop is
what stops the same correction being needed thirty times.

### Flow C — scheduled conformance sweep

Bump pin → regenerate → run harness across all 32 packages → file one pre-diagnosed
`ai:tier1` issue per gap, carrying the exact failing assertion. **Rate-limited to 10 issues
per run, weekly.** First enable will find well over a hundred gaps; unthrottled it buries
the queue.

The sweep is the only actor permitted to modify `.semconv/`, and it does so in a dedicated
PR that changes nothing else. This is deliberately not an exception to the `fixer` path
guard below — it is a different workflow with a different, narrower write scope.

## Guardrails

| Risk | Control |
|---|---|
| Robot-authored 499 backlog | Max 5 in-flight AI PRs, enforced pre-flight |
| Infinite retry burn | 2 attempts per issue, then `ai:needs-human` and stop |
| Blast radius | CI path guard, per agent role: `fixer` may not touch `.github/**`, `.claude/**`, `.semconv/**`, `docs/ai/**`, release config, or version fields. `responder` may additionally write `docs/ai/knowledge/**` and nothing else new. Only the sweep may touch `.semconv/**`. |
| Prompt injection | Untrusted text passed as delimited data with explicit preamble; `recorder` never receives it |
| Secret leak via cassette | gitleaks + provider-key patterns, fail closed |
| Runaway spend | Per-run token cap; provider keys used by `recorder` carry their own monthly spend limit set at the provider, not in CI |
| Systemic failure | Repo variable `AI_MAINTAINER_ENABLED=false`, checked at top of every workflow |
| Trust laundering | Dedicated bot identity, `ai-authored` label, **auto-merge never enabled** |

Guardrails are enforced in CI, not by prompt instruction. A prompt asking an agent not to
touch `.github/` is a request; a CI path check is a control.

## Build order

Four specs, each independently valuable.

1. **Contract** — no agents. Vendor registry, weaver codegen, conformance harness,
   warn-only rollout, land the 5 packages that already have compliance tests. Ships real
   value even if work stops here.
2. **Substrate** — root `AGENTS.md` plus `CLAUDE.md` symlink, 32 per-package `AGENTS.md`
   (agent-bootstrapped, human-reviewed), procedures in `docs/ai/procedures/` with thin
   `.claude/skills/` wrappers, OKF bundle scaffold (`index.md`, `log.md`, `okf_version`),
   and a CI check asserting no knowledge content lives under `.claude/`.
3. **Fix loop** — TDD gate CI, `fixer` + `recorder`, path guard, scrub gate. Piloted
   manually on 3 hand-picked issues before any trigger is wired.
4. **Autonomy** — `triage`, `reviewer`, `responder`, sweep, concurrency and budget guardrails.

Spec 3's verification story rests entirely on spec 1 existing. Spec 4 earns trust only
because spec 3 has a track record. Building spec 4 first is the standard failure mode for
this class of project.

**Sequencing constraint:** Flow C's sweep stays disabled until spec 3 has merged ~10 PRs
successfully. A sweep filing 100+ correct issues into an unproven fix loop produces 100
bad PRs.

## Explicitly out of scope

- The existing 499-PR backlog. Acknowledged as the larger constraint; deferred by decision.
- Auto-merge, at any tier.
- Agent authority over releases, versioning, or `.github` configuration.
- Native OKF retrieval tooling. Agents are pointed at the bundle; no indexing service.

## References

- [open-telemetry/semantic-conventions-genai](https://github.com/open-telemetry/semantic-conventions-genai)
  — the contract. Weaver registry, `stability: development`, no tags.
- [OKF SPEC.md](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md)
  — Open Knowledge Format v0.2, Apache-2.0.
- [How the Open Knowledge Format can improve data sharing](https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing)
  — Google Cloud announcement, 2026-06-12.
- [OKF: Redefining Knowledge Bases for AI Agents](https://www.analyticsvidhya.com/blog/2026/07/open-knowledge-format/)
  — independent overview.
