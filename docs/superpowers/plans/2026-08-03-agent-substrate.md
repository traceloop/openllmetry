# Agent Substrate Implementation Plan (Spec 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give agents of any vendor a portable knowledge base for this repo — imperative instructions in `AGENTS.md`, procedures in plain markdown, and empirical knowledge in an OKF bundle with provenance and expiry.

**Architecture:** `AGENTS.md` becomes the canonical instruction file with `CLAUDE.md` as a git symlink to it, matching what the upstream contract repo already does. Procedures live in `docs/ai/procedures/` as plain markdown; `.claude/skills/*/SKILL.md` are content-free pointers to them. Empirical knowledge — accumulated lessons and observed SDK behaviour — becomes an OKF v0.2 bundle at `docs/ai/knowledge/`. A CI guard asserts no knowledge content ever lands in a vendor-specific directory.

**Tech Stack:** Markdown, YAML frontmatter, Python 3.10+ (for the guard test), pytest, Nx, GitHub Actions.

## Global Constraints

- **No vendor lock-in.** No knowledge may live in a Claude-specific file. `AGENTS.md` is canonical; `CLAUDE.md` is a symlink (git mode `120000`). `.claude/` holds only pointers.
- **Split rule:** *imperative → `AGENTS.md`, empirical → OKF.* "Run `nx run <pkg>:test`" is imperative. "Anthropic returns an empty content block on tool-only turns" is empirical.
- **OKF v0.2** (`github.com/GoogleCloudPlatform/knowledge-catalog`, Apache-2.0). Bundle root `index.md` declares `okf_version: "0.2"`. Every concept file carries YAML frontmatter with a non-empty `type`.
- **No blanket per-package files.** All 35 packages share identical nx targets, so a per-package `AGENTS.md` would be boilerplate. Create one only where a package genuinely deviates from the standard workflow.
- Actor convention: `<producer>/<version>` for agents, `human:<id>` for people.
- This spec adds documentation and one guard test. It changes no instrumentation source and no package behaviour.

## Baseline (verified 2026-08-03)

- Root `CLAUDE.md` exists, 89 lines, is a regular file, and is the only agent instruction file in the repo.
- `.claude/` contains only `settings.local.json`.
- `docs/ai/` contains only `semconv-rollout.md` (from spec 1).
- 35 packages, all with a `README.md`, all with identical `project.json` targets (`install`, `lint`, `test`, `build`, `build-release`, `lock`, `add`, `update`, `remove`).
- 16 packages carry a `semconv:warn` tag from spec 1.

## File Structure

| Path | Responsibility |
|---|---|
| `AGENTS.md` | Canonical instructions. Content of today's `CLAUDE.md` plus a router section pointing at procedures and the knowledge bundle. |
| `CLAUDE.md` | Symlink → `AGENTS.md`. Git mode `120000`. |
| `docs/ai/procedures/README.md` | Index of procedures. |
| `docs/ai/procedures/fix-instrumentation-bug.md` | The TDD-first bug procedure. |
| `docs/ai/procedures/add-instrumentation.md` | Adding a new instrumentation package. |
| `docs/ai/procedures/record-cassette.md` | Recording/scrubbing a VCR cassette. |
| `docs/ai/procedures/semconv-conformance.md` | Working with the spec-1 contract. |
| `.claude/skills/*/SKILL.md` | Content-free pointers to the above. |
| `docs/ai/knowledge/index.md` | OKF bundle root. Declares `okf_version`. |
| `docs/ai/knowledge/log.md` | OKF change history, newest first. |
| `docs/ai/knowledge/*.md` | One concept per file. |
| `tools/agent_substrate/test_substrate.py` | The guard: symlink intact, no knowledge under `.claude/`, OKF frontmatter valid. |
| `tools/agent_substrate/project.json` | Nx wiring so the guard runs in CI. |

---

### Task 1: AGENTS.md canonical, CLAUDE.md as symlink

**Files:**
- Create: `AGENTS.md` (from existing `CLAUDE.md` content)
- Replace: `CLAUDE.md` (regular file → symlink)

**Interfaces:**
- Consumes: nothing.
- Produces: `AGENTS.md` at repo root as the canonical instruction file; `CLAUDE.md` as a symlink to it. Task 5's guard test asserts both.

- [ ] **Step 1: Move the file, preserving content**

```bash
cd "$(git rev-parse --show-toplevel)"
git mv CLAUDE.md AGENTS.md
```

- [ ] **Step 2: Append the router section to AGENTS.md**

Add at the end of `AGENTS.md`:

```markdown
## For agents working in this repo

This file is canonical and vendor-neutral. `CLAUDE.md` is a symlink to it — never
edit `CLAUDE.md` directly, and never put knowledge in `.claude/`, which holds only
pointers.

**Procedures** — how to carry out recurring tasks — live in `docs/ai/procedures/`:

| Task | Procedure |
|---|---|
| Fix a reported bug | [fix-instrumentation-bug.md](docs/ai/procedures/fix-instrumentation-bug.md) |
| Add a new instrumentation package | [add-instrumentation.md](docs/ai/procedures/add-instrumentation.md) |
| Record or re-record a VCR cassette | [record-cassette.md](docs/ai/procedures/record-cassette.md) |
| Work with the semconv contract | [semconv-conformance.md](docs/ai/procedures/semconv-conformance.md) |

**Knowledge** — what we have learned about these SDKs and this codebase — lives in
`docs/ai/knowledge/`, an [OKF](https://github.com/GoogleCloudPlatform/knowledge-catalog)
bundle. Start at [index.md](docs/ai/knowledge/index.md). Each concept carries
provenance and a `stale_after` date; ignore concepts whose date has passed.

The split is deliberate: **imperative content belongs here, empirical content belongs
in the knowledge bundle.** "Run `nx run <pkg>:test`" is imperative. "Anthropic returns
an empty content block on tool-only turns" is empirical — it rots when the SDK bumps,
so it needs the expiry and provenance the bundle provides.

There are deliberately no per-package `AGENTS.md` files. All 35 packages share the
same nx targets, so per-package instruction files would be boilerplate. Package-specific
findings go in the knowledge bundle instead. Create a per-package `AGENTS.md` only if a
package genuinely deviates from the standard workflow.
```

- [ ] **Step 3: Replace CLAUDE.md with a symlink**

```bash
cd "$(git rev-parse --show-toplevel)"
ln -s AGENTS.md CLAUDE.md
git add AGENTS.md CLAUDE.md
```

- [ ] **Step 4: Verify git recorded a symlink, not a copy**

```bash
git ls-files -s CLAUDE.md AGENTS.md
```

Expected: `CLAUDE.md` has mode `120000`; `AGENTS.md` has mode `100644`. If `CLAUDE.md`
shows `100644`, git stored a regular file — remove it, re-create the symlink, and re-add.

- [ ] **Step 5: Verify the symlink resolves**

```bash
head -1 CLAUDE.md
```

Expected: `# OpenLLMetry Repository Guide` — the first line of `AGENTS.md`.

- [ ] **Step 6: Commit**

```bash
git commit -m "docs: make AGENTS.md canonical with CLAUDE.md as a symlink"
```

---

### Task 2: Procedures in plain markdown

**Files:**
- Create: `docs/ai/procedures/README.md`
- Create: `docs/ai/procedures/fix-instrumentation-bug.md`
- Create: `docs/ai/procedures/add-instrumentation.md`
- Create: `docs/ai/procedures/record-cassette.md`
- Create: `docs/ai/procedures/semconv-conformance.md`

**Interfaces:**
- Consumes: the router table in `AGENTS.md` (Task 1) links to these exact filenames.
- Produces: five procedure files that Task 3's skill wrappers point at, and that Task 5's guard test checks for existence.

- [ ] **Step 1: Write the procedures index**

Create `docs/ai/procedures/README.md`:

```markdown
# Procedures

How to carry out recurring work in this repo. Plain markdown, readable by any agent
or human — no vendor-specific format.

| Procedure | Use when |
|---|---|
| [fix-instrumentation-bug.md](fix-instrumentation-bug.md) | A defect is reported against an instrumentation package |
| [add-instrumentation.md](add-instrumentation.md) | Adding support for a new AI library |
| [record-cassette.md](record-cassette.md) | A test needs a new or refreshed VCR cassette |
| [semconv-conformance.md](semconv-conformance.md) | Working with the OTel GenAI contract |

These describe *how to do things*. Things we have *learned* live in
[../knowledge/](../knowledge/index.md) instead.
```

- [ ] **Step 2: Write the bug-fixing procedure**

Create `docs/ai/procedures/fix-instrumentation-bug.md`:

```markdown
# Fixing an instrumentation bug

**The rule: write a failing test that reproduces the bug before you change any source.**
A fix without a test that failed first is not accepted, because nothing proves the bug
existed or that the change addresses it.

## 1. Reproduce

Identify the package and read its existing tests to learn the fixture names and the
call shape. Do not assume fixture names — they differ per package.

Write a test that fails for the reason the issue describes. Run it and read the failure
output. If it passes, you have not reproduced the bug: either the report is wrong, or
your test does not exercise the path. Resolve that before continuing.

Prefer an existing cassette. Recording a new one needs API keys — see
[record-cassette.md](record-cassette.md).

## 2. Commit the failing test on its own

    git add packages/<pkg>/tests/test_<area>.py
    git commit -m "test: reproduce #<issue> — <one line>"

Test files only in this commit. No source changes.

## 3. Fix

Change the source until the new test passes.

    cd packages/<pkg> && uv run pytest tests/test_<area>.py -v

## 4. Prove nothing else broke

    cd packages/<pkg> && uv run pytest tests/ -q
    cd packages/<pkg> && uv run ruff check .

## 5. Commit the fix separately

    git add packages/<pkg>/opentelemetry
    git commit -m "fix(<pkg>): <one line>"

Two commits, in this order, is the required shape: the first must fail without the
second. That is what makes the fix reviewable.

## Attribute changes

If the bug concerns span attributes, read [semconv-conformance.md](semconv-conformance.md)
first. The attribute name you want may already be defined by the contract, and inventing
a `gen_ai.*` name that upstream does not define is itself a defect.
```

- [ ] **Step 3: Write the add-instrumentation procedure**

Create `docs/ai/procedures/add-instrumentation.md`:

```markdown
# Adding a new instrumentation package

## Layout

Copy the structure of an existing small package — `opentelemetry-instrumentation-together`
is a good template.

    packages/opentelemetry-instrumentation-<name>/
      pyproject.toml
      project.json                 # nx targets — copy verbatim, change only the name/paths
      README.md
      opentelemetry/instrumentation/<name>/
        __init__.py                # the Instrumentor
        config.py
        span_utils.py              # attribute setting
        version.py
      tests/
        conftest.py
        cassettes/

## Wiring

`project.json` targets are identical across all packages. Copy from the template and
change only `name`, `sourceRoot`, and the `cwd` values. Add `"instrumentation"` to `tags`.

If the package depends on the local semconv package, add to `pyproject.toml`:

    [tool.uv.sources]
    opentelemetry-semantic-conventions-ai = { path = "../opentelemetry-semantic-conventions-ai", editable = true }

Without this, uv resolves a published wheel instead of local source.
`scripts/build-release.sh` strips this block before building, so it does not leak into
the published package.

## Attributes

Emit the attributes the OTel GenAI contract defines. Read
[semconv-conformance.md](semconv-conformance.md) before naming anything. Do not invent
new `gen_ai.*` names.

## Conformance

Add `tests/test_conformance.py` following the pattern in
`packages/opentelemetry-instrumentation-ollama/tests/test_conformance.py`, and add
`"semconv:warn"` to the package's `project.json` tags.

## Verify

    npx nx run opentelemetry-instrumentation-<name>:install
    npx nx run opentelemetry-instrumentation-<name>:lint
    npx nx run opentelemetry-instrumentation-<name>:test
```

- [ ] **Step 4: Write the cassette procedure**

Create `docs/ai/procedures/record-cassette.md`:

```markdown
# Recording a VCR cassette

Tests replay recorded HTTP interactions so they run without network or API keys.

## Prefer reuse

Recording requires live API keys and spends real money. Before recording, check whether
an existing cassette covers the call. Reuse one from another test module with:

    @pytest.mark.vcr
    @pytest.mark.default_cassette("test_messages/test_some_existing_test")

plus a module-scoped `vcr_cassette_dir` fixture rooted at `tests/cassettes`. See
`packages/opentelemetry-instrumentation-anthropic/tests/test_conformance.py`.

**Trap:** VCR's default match is `(method, scheme, host, port, path, query)` — the request
body is NOT compared. Every call to a given provider hits the same path, so if the test
whose cassette you borrowed later changes its request, yours silently replays a stale
interaction. Mirror any such change, or record your own.

## Recording

Requires valid provider API keys in the environment. Ask the repo owner; never hardcode them.

    cd packages/<pkg>
    uv run pytest tests/test_x.py --record-mode=once      # only if missing
    uv run pytest tests/test_x.py --record-mode=all       # re-record everything
    uv run pytest tests/ --record-mode=none               # fail if a cassette is missing

## Before committing a cassette

Cassettes are committed to a public repository. Verify:

1. No `Authorization`, `x-api-key`, or `api-key` header values remain. The package's
   `vcr_config` fixture should filter them — confirm it actually did.
2. No PII in request or response bodies.
3. No account identifiers, org IDs, or billing data.

Read the recorded YAML before committing it. A leaked key in git history is not
recoverable by deleting the file.
```

- [ ] **Step 5: Write the conformance procedure**

Create `docs/ai/procedures/semconv-conformance.md`:

```markdown
# Working with the semconv contract

The OpenTelemetry GenAI semantic conventions are vendored and executable here. The
contract is the authority on attribute names — do not invent `gen_ai.*` names.

## Layout

    .semconv/versions.env      pinned upstream SHA, weaver version
    .semconv/registry/         vendored upstream registry (committed)
    packages/opentelemetry-semantic-conventions-ai/
      .../\_contract/generated.py    committed, generated — never hand-edit
      .../\_contract/extensions.py   non-standard gen_ai.* we knowingly emit
      .../conformance.py            the checker

## Commands

    make -C .semconv verify-vendor   # registry still matches the pinned SHA
    make -C .semconv check           # generated.py still matches the registry
    make -C .semconv vendor          # refetch at the pinned SHA
    make -C .semconv generate        # regenerate the committed artifact

Both `verify-vendor` and `check` run in CI. Regenerating requires Docker.

## Checking a package

Each covered package has `tests/test_conformance.py` declaring:

- `CONTRACT_GROUP` — which contract group its spans map to
- `EXPECTED` — attributes the package **promises** to emit
- `ENFORCING` — `False` while violations remain

`EXPECTED` matters more than it looks. Only ~9% of contract attributes are `required`,
and `gen_ai.response.finish_reasons` is merely `recommended`, so registry levels alone
will not notice a dropped attribute. The package's own promise is what catches it.

## Violation kinds

| Kind | Blocking | Meaning |
|---|---|---|
| `missing_required` | yes | Contract marks it required; span lacks it |
| `missing_expected` | yes | Package promised it; span lacks it |
| `undeclared_gen_ai` | yes | A `gen_ai.*` attribute the contract does not define |
| `unknown_enum_value` | no | Value outside the declared members — OTel enums are open |

## Adding an extension

If a package must emit a `gen_ai.*` attribute the contract does not define, add it to
`_contract/extensions.py` with a real rationale. Each entry is a permanent hole in the
check, so prefer a `traceloop.*` name — the harness ignores other namespaces entirely.

**Known limitation:** matching is exact. An entry covers only that literal name, so the
indexed legacy forms (`gen_ai.prompt.0.content`) are not covered by declaring
`gen_ai.prompt`.

## Bumping the pin

Change `SEMCONV_GENAI_REF` in `.semconv/versions.env`, then `make -C .semconv vendor`
and `make -C .semconv generate`, and commit. The diff on `generated.py` shows exactly what
changed in the contract. Never track `main` — upstream is `stability: development` with
no tags.

## Current coverage

See [../semconv-rollout.md](../semconv-rollout.md).
```

- [ ] **Step 6: Verify every link in the router resolves**

```bash
cd "$(git rev-parse --show-toplevel)"
for f in fix-instrumentation-bug add-instrumentation record-cassette semconv-conformance; do
  test -f "docs/ai/procedures/$f.md" && echo "ok $f" || echo "MISSING $f"
done
```

Expected: four `ok` lines.

- [ ] **Step 7: Commit**

```bash
git add docs/ai/procedures
git commit -m "docs: add agent procedures in plain markdown"
```

---

### Task 3: Thin skill wrappers

**Files:**
- Create: `.claude/skills/fix-instrumentation-bug/SKILL.md`
- Create: `.claude/skills/add-instrumentation/SKILL.md`
- Create: `.claude/skills/record-cassette/SKILL.md`
- Create: `.claude/skills/semconv-conformance/SKILL.md`

**Interfaces:**
- Consumes: the four procedure files from Task 2.
- Produces: four `SKILL.md` files, each under 15 lines and containing no procedural content of its own. Task 5's guard test enforces both properties.

- [ ] **Step 1: Write the four wrappers**

Each is a pointer, not a copy. Duplicating procedure text here would create a
Claude-only fork of the knowledge that silently drifts — which is exactly what this
spec exists to prevent.

Create `.claude/skills/fix-instrumentation-bug/SKILL.md`:

```markdown
---
name: fix-instrumentation-bug
description: Use when fixing a reported defect in an instrumentation package - enforces writing a failing reproducing test before any source change
---

Read and follow `docs/ai/procedures/fix-instrumentation-bug.md`.

That file is the procedure. This wrapper exists only so the procedure is reachable as a
skill; it deliberately holds no content of its own, because duplicating it here would
create a vendor-specific copy that drifts.
```

Create `.claude/skills/add-instrumentation/SKILL.md`:

```markdown
---
name: add-instrumentation
description: Use when adding a new instrumentation package for an AI library - covers layout, nx wiring, semconv attributes, and conformance
---

Read and follow `docs/ai/procedures/add-instrumentation.md`.

That file is the procedure. This wrapper exists only so the procedure is reachable as a
skill; it deliberately holds no content of its own, because duplicating it here would
create a vendor-specific copy that drifts.
```

Create `.claude/skills/record-cassette/SKILL.md`:

```markdown
---
name: record-cassette
description: Use when a test needs a new or refreshed VCR cassette - covers reuse, recording with API keys, and the pre-commit secret checks
---

Read and follow `docs/ai/procedures/record-cassette.md`.

That file is the procedure. This wrapper exists only so the procedure is reachable as a
skill; it deliberately holds no content of its own, because duplicating it here would
create a vendor-specific copy that drifts.
```

Create `.claude/skills/semconv-conformance/SKILL.md`:

```markdown
---
name: semconv-conformance
description: Use when working with span attributes or the OTel GenAI contract - covers the vendored registry, violation kinds, extensions, and bumping the pin
---

Read and follow `docs/ai/procedures/semconv-conformance.md`.

That file is the procedure. This wrapper exists only so the procedure is reachable as a
skill; it deliberately holds no content of its own, because duplicating it here would
create a vendor-specific copy that drifts.
```

- [ ] **Step 2: Verify each wrapper is genuinely thin**

```bash
cd "$(git rev-parse --show-toplevel)"
wc -l .claude/skills/*/SKILL.md
```

Expected: each under 15 lines. A wrapper that grew past that is accumulating content
that belongs in `docs/ai/procedures/`.

- [ ] **Step 3: Commit**

```bash
git add .claude/skills
git commit -m "docs: add thin skill wrappers pointing at the procedures"
```

---

### Task 4: OKF knowledge bundle

**Files:**
- Create: `docs/ai/knowledge/index.md`
- Create: `docs/ai/knowledge/log.md`
- Create: `docs/ai/knowledge/otel-genai-enums-are-open.md`
- Create: `docs/ai/knowledge/vcr-cassette-reuse-ignores-body.md`
- Create: `docs/ai/knowledge/semconv-requirement-levels-are-weak.md`

**Interfaces:**
- Consumes: nothing.
- Produces: an OKF v0.2 bundle. `index.md` declares `okf_version: "0.2"` in its frontmatter — the only file permitted to. Every other `.md` in the directory is a concept with frontmatter carrying a non-empty `type`. Task 5's guard test parses all of them.

The three seed concepts are real findings from spec 1, not filler. A bundle that ships
empty teaches nothing and signals that nobody intends to fill it.

- [ ] **Step 1: Write the bundle root**

Create `docs/ai/knowledge/index.md`:

```markdown
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
```

- [ ] **Step 2: Write the log**

Create `docs/ai/knowledge/log.md`:

```markdown
---
type: Change Log
title: Knowledge bundle change log
description: Dated history of concept additions and revisions, newest first.
---

# Change log

## 2026-08-03

- Created the bundle.
- Added `otel-genai-enums-are-open` — from the spec-1 conformance harness review.
- Added `semconv-requirement-levels-are-weak` — from the spec-1 conformance harness review.
- Added `vcr-cassette-reuse-ignores-body` — found while wiring conformance tests.
```

- [ ] **Step 3: Write the open-enums concept**

Create `docs/ai/knowledge/otel-genai-enums-are-open.md`:

```markdown
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
```

- [ ] **Step 4: Write the requirement-levels concept**

Create `docs/ai/knowledge/semconv-requirement-levels-are-weak.md`:

```markdown
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

Only 5 of 80 unique attribute names are ever presence-checked by the contract alone.
A package emitting nothing but `gen_ai.operation.name` and `gen_ai.provider.name` passes
as fully conformant.

Critically, `gen_ai.response.finish_reasons` is `recommended` in **all 9** groups that
declare it. Issue #4362 — streaming dropping `finish_reasons` — is therefore invisible to
a contract-levels-only check.

**Consequence:** each package declares an `EXPECTED` frozenset in its
`tests/test_conformance.py` naming the attributes it promises to emit. A missing promise
is a blocking `missing_expected` violation regardless of the registry's level. The
contract supplies the vocabulary; the package supplies the commitment.

A name in `EXPECTED` that the contract does not declare raises `ValueError` — a typo
must fail loudly rather than silently check nothing.
```

- [ ] **Step 5: Write the cassette concept**

Create `docs/ai/knowledge/vcr-cassette-reuse-ignores-body.md`:

```markdown
---
type: Testing Pitfall
title: A borrowed VCR cassette goes stale silently
description: VCR's default match ignores the request body, so reusing another test's cassette can replay the wrong interaction.
tags: [testing, vcr, cassettes]
generated:
  by: claude-opus/5
  at: 2026-08-03T00:00:00Z
sources:
  - author: human:gal
    note: Found while wiring conformance tests that reuse existing cassettes.
status: stable
stale_after: 2027-02-01
---

# A borrowed VCR cassette goes stale silently

Conformance tests reuse cassettes recorded by other test modules, because recording new
ones needs API keys. The package `vcr_config` fixtures set only `filter_headers`, so VCR
falls back to its default match: `(method, scheme, host, port, path, query)`. **The
request body is never compared.**

Every call to a given provider posts to the same path — `/v1/messages` for Anthropic.
So if the source test later changes its model, prompt, or parameters, the borrowing test
keeps replaying the old cassette without complaint, silently validating an interaction
that no longer corresponds to what it asked for.

**Mitigation attempted and rejected:** adding `body` to `match_on`. It broke on
Anthropic's structured-output cassette because the installed SDK rewrites `output_format`
into `output_config.format` on the wire — SDK drift, not test drift. Strict body matching
therefore produces false failures on cassettes recorded with an older SDK.

**Current state:** unguarded. If you change a test whose cassette another module borrows
via `@pytest.mark.default_cassette`, grep for that cassette name and mirror the change.
```

- [ ] **Step 6: Verify the frontmatter parses**

```bash
cd "$(git rev-parse --show-toplevel)"
python3 -c "
import glob, sys
ok = True
for p in sorted(glob.glob('docs/ai/knowledge/*.md')):
    text = open(p).read()
    if not text.startswith('---'):
        print('NO FRONTMATTER', p); ok = False; continue
    fm = text.split('---', 2)[1]
    if 'type:' not in fm:
        print('NO TYPE', p); ok = False
    print('ok', p)
sys.exit(0 if ok else 1)
"
```

Expected: an `ok` line per file, exit 0.

- [ ] **Step 7: Commit**

```bash
git add docs/ai/knowledge
git commit -m "docs: add OKF knowledge bundle with seed concepts"
```

---

### Task 5: The substrate guard

**Files:**
- Create: `tools/agent_substrate/test_substrate.py`
- Create: `tools/agent_substrate/project.json`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: everything Tasks 1–4 created.
- Produces: an nx project named `agent-substrate` with a `test` target, wired into CI.

This is what makes the substrate's invariants enforceable rather than aspirational. Every
rule stated in `AGENTS.md` that a person could quietly violate gets a test here.

- [ ] **Step 1: Write the failing test**

Create `tools/agent_substrate/test_substrate.py`:

```python
"""Guards the agent substrate's portability invariants.

The substrate is only vendor-neutral if nothing drifts back into a vendor-specific
file. These assertions are the enforcement; the prose in AGENTS.md is not.
"""

import glob
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(
    subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()
)

PROCEDURES = REPO_ROOT / "docs" / "ai" / "procedures"
KNOWLEDGE = REPO_ROOT / "docs" / "ai" / "knowledge"
SKILLS = REPO_ROOT / ".claude" / "skills"

MAX_WRAPPER_LINES = 15


class TestCanonicalInstructions:
    def test_agents_md_is_the_canonical_regular_file(self):
        path = REPO_ROOT / "AGENTS.md"
        assert path.is_file(), "AGENTS.md must exist at the repo root"
        assert not path.is_symlink(), "AGENTS.md is canonical; it must not be a symlink"

    def test_claude_md_is_a_symlink_to_agents_md(self):
        """CLAUDE.md must be a pointer, never a second copy that can drift."""
        path = REPO_ROOT / "CLAUDE.md"
        assert path.is_symlink(), (
            "CLAUDE.md must be a symlink to AGENTS.md, not a regular file — "
            "two copies drift apart and the knowledge becomes vendor-specific"
        )
        assert os.readlink(path) == "AGENTS.md"

    def test_git_records_claude_md_as_a_symlink(self):
        """A symlink on disk is not enough; git must have stored mode 120000."""
        out = subprocess.check_output(
            ["git", "ls-files", "-s", "CLAUDE.md"], cwd=REPO_ROOT, text=True
        )
        assert out.split()[0] == "120000", f"expected git mode 120000, got: {out!r}"


class TestNoKnowledgeInVendorDirs:
    def test_claude_dir_contains_only_pointers(self):
        """Any substantial file under .claude/ is knowledge that belongs in docs/ai/."""
        offenders = []
        for path in SKILLS.rglob("*.md"):
            lines = path.read_text().splitlines()
            if len(lines) > MAX_WRAPPER_LINES:
                offenders.append(f"{path.relative_to(REPO_ROOT)} ({len(lines)} lines)")
        assert not offenders, (
            "these files under .claude/ exceed a pointer's size, so they hold content "
            f"that belongs in docs/ai/procedures/: {offenders}"
        )

    def test_every_skill_wrapper_points_at_a_real_procedure(self):
        wrappers = sorted(SKILLS.glob("*/SKILL.md"))
        assert wrappers, "expected skill wrappers under .claude/skills/*/SKILL.md"
        for wrapper in wrappers:
            text = wrapper.read_text()
            assert "docs/ai/procedures/" in text, (
                f"{wrapper.relative_to(REPO_ROOT)} does not reference a procedure file"
            )
            referenced = [
                tok.strip("`.,)")
                for tok in text.split()
                if tok.strip("`.,)").startswith("docs/ai/procedures/")
            ]
            for ref in referenced:
                assert (REPO_ROOT / ref).is_file(), (
                    f"{wrapper.relative_to(REPO_ROOT)} points at {ref}, which does not exist"
                )


class TestProcedures:
    @pytest.mark.parametrize(
        "name",
        [
            "fix-instrumentation-bug",
            "add-instrumentation",
            "record-cassette",
            "semconv-conformance",
        ],
    )
    def test_procedure_exists(self, name):
        assert (PROCEDURES / f"{name}.md").is_file()

    def test_agents_md_links_every_procedure(self):
        """A procedure nobody can find from the entry point is not discoverable."""
        agents = (REPO_ROOT / "AGENTS.md").read_text()
        for path in PROCEDURES.glob("*.md"):
            if path.name == "README.md":
                continue
            assert path.name in agents, (
                f"{path.name} is not linked from AGENTS.md, so an agent starting at the "
                "canonical entry point will never find it"
            )


class TestOkfBundle:
    def _concepts(self):
        return [
            Path(p)
            for p in sorted(glob.glob(str(KNOWLEDGE / "*.md")))
            if Path(p).name not in ("index.md", "log.md")
        ]

    def test_bundle_root_declares_okf_version(self):
        text = (KNOWLEDGE / "index.md").read_text()
        assert 'okf_version: "0.2"' in text

    def test_only_the_root_declares_okf_version(self):
        """OKF permits okf_version in the bundle root's frontmatter only."""
        for path in KNOWLEDGE.glob("*.md"):
            if path.name == "index.md":
                continue
            assert "okf_version:" not in path.read_text(), (
                f"{path.name} declares okf_version; only index.md may"
            )

    def test_log_exists(self):
        assert (KNOWLEDGE / "log.md").is_file()

    def test_bundle_has_concepts(self):
        assert self._concepts(), (
            "the bundle has no concepts — an empty knowledge base teaches nothing"
        )

    def test_every_file_has_parseable_frontmatter_with_a_type(self):
        for path in KNOWLEDGE.glob("*.md"):
            text = path.read_text()
            assert text.startswith("---\n"), f"{path.name} has no frontmatter block"
            frontmatter = text.split("---", 2)[1]
            assert "type:" in frontmatter, f"{path.name} frontmatter has no type field"
            type_line = next(
                line for line in frontmatter.splitlines() if line.startswith("type:")
            )
            assert type_line.split(":", 1)[1].strip(), f"{path.name} has an empty type"

    def test_every_concept_is_listed_in_the_index(self):
        index = (KNOWLEDGE / "index.md").read_text()
        for path in self._concepts():
            assert path.name in index, (
                f"{path.name} is not listed in index.md, so retrieval by an agent "
                "reading the index will miss it"
            )

    def test_concepts_carry_provenance(self):
        """OKF trust tiers depend on `generated`; a concept without it is anonymous."""
        for path in self._concepts():
            assert "generated:" in path.read_text(), (
                f"{path.name} has no `generated:` block, so its trust tier is unknowable"
            )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd tools/agent_substrate && python3 -m pytest test_substrate.py -v`

Expected: FAIL. Before Tasks 1–4 land, `AGENTS.md` and the bundle do not exist. If Tasks
1–4 are already complete, expect PASS — in that case deliberately break one invariant
(e.g. `rm CLAUDE.md && cp AGENTS.md CLAUDE.md`) and confirm
`test_claude_md_is_a_symlink_to_agents_md` fails, then restore with
`rm CLAUDE.md && ln -s AGENTS.md CLAUDE.md`. A guard that cannot fail is not a guard.

- [ ] **Step 3: Wire it into nx**

Create `tools/agent_substrate/project.json`:

```json
{
  "name": "agent-substrate",
  "$schema": "../../node_modules/nx/schemas/project-schema.json",
  "projectType": "library",
  "targets": {
    "test": {
      "executor": "nx:run-commands",
      "options": {
        "command": "python3 -m pytest test_substrate.py -v",
        "cwd": "tools/agent_substrate"
      }
    }
  },
  "tags": ["tooling"]
}
```

- [ ] **Step 4: Verify nx sees the project**

```bash
cd "$(git rev-parse --show-toplevel)"
npx nx show projects --projects agent-substrate
```

Expected: `agent-substrate`

- [ ] **Step 5: Run through nx**

```bash
npx nx run agent-substrate:test
```

Expected: all tests pass.

- [ ] **Step 6: Add the CI job**

Add to `.github/workflows/ci.yml` under `jobs:`, matching the surrounding indentation:

```yaml
  agent-substrate:
    name: Agent Substrate
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.sha }}
          persist-credentials: false

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: 3.11

      - run: pip install pytest

      # Enforces the substrate's portability invariants: CLAUDE.md stays a symlink,
      # no knowledge accumulates under .claude/, and the OKF bundle stays valid.
      - name: Verify agent substrate
        run: python3 -m pytest tools/agent_substrate/test_substrate.py -v
```

Note the checkout has no `fetch-depth: 0` — the test only needs the working tree and
`git ls-files`, not history.

- [ ] **Step 7: Verify the symlink survives a fresh clone**

Git stores symlinks, but a checkout on a filesystem without symlink support materialises
them as regular files, which would make the CI job fail confusingly. Confirm on a clean
clone:

```bash
cd "$(mktemp -d)"
git clone -q "$(git -C "$OLDPWD" rev-parse --show-toplevel)" repo
cd repo && git checkout -q "$(git -C "$OLDPWD" branch --show-current)" 2>/dev/null || true
python3 -c "
import os
print('is symlink:', os.path.islink('CLAUDE.md'))
print('target:', os.readlink('CLAUDE.md') if os.path.islink('CLAUDE.md') else 'N/A')
"
```

Expected: `is symlink: True`, `target: AGENTS.md`. Then remove the temp clone.

- [ ] **Step 8: Commit**

```bash
git add tools/agent_substrate .github/workflows/ci.yml
git commit -m "test: enforce agent substrate portability invariants"
```

---

## Self-Review

**Spec coverage.** Every spec-2 requirement maps to a task: `AGENTS.md` canonical with
`CLAUDE.md` symlink (T1), procedures as plain markdown (T2), content-free `.claude/skills`
wrappers (T3), OKF v0.2 bundle with `index.md`/`log.md`/`okf_version` (T4), and the CI
check asserting no knowledge content lives under `.claude/` (T5).

**Deliberate deviation from the design, approved by the repo owner.** The design specified
32 per-package `AGENTS.md` files. Those are not built here. All 35 packages share identical
nx targets, so per-package instruction files would be boilerplate; and per the substrate's
own *imperative → `AGENTS.md`, empirical → OKF* rule, genuine package-specific findings are
empirical and belong in the knowledge bundle where they get provenance and `stale_after`.
`AGENTS.md` states this explicitly so the absence reads as a decision rather than an
oversight. A per-package file is still permitted where a package genuinely deviates.

**Placeholder scan.** No TBDs. Every file's full content is given. Task 5 Step 2's
"deliberately break an invariant" branch is a real instruction with the exact commands,
not a gap.

**Type consistency.** Procedure filenames are identical across T2 (creation), T1's router
table, T3's wrappers, and T5's parametrized test: `fix-instrumentation-bug`,
`add-instrumentation`, `record-cassette`, `semconv-conformance`. `MAX_WRAPPER_LINES = 15`
in T5 matches the "under 15 lines" check in T3 Step 2. The nx project name
`agent-substrate` is consistent across T5's `project.json`, the verification command, and
the CI job.

**Non-vacuousness.** T5 Step 2 requires seeing the guard fail before trusting it, and T5
Step 7 checks the symlink survives a fresh clone — the failure mode that would make the
whole vendor-neutrality mechanism silently degrade into two diverging copies.
