# Semconv Contract Implementation Plan (Spec 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the OpenTelemetry GenAI semantic conventions into an executable contract that every instrumentation package is tested against, so contract violations fail CI instead of being reported as individual bugs.

**Architecture:** Vendor the upstream Weaver registry at a pinned SHA into `.semconv/`. A Makefile resolves it to JSON via the pinned `otel/weaver` container, and a generator script converts that into a committed Python module (`_contract/generated.py`). A pure-Python conformance harness checks emitted span attributes against that contract and is wired into each package's test suite, warn-only at first, flipping to enforcing per package.

**Tech Stack:** Python 3.10–3.12, pytest, uv, Nx, Docker (for `otel/weaver:v0.25.0`), GNU Make.

## Global Constraints

- Upstream contract repo: `https://github.com/open-telemetry/semantic-conventions-genai`, pinned to SHA `8484f22ff8069267f37cb1be54bcebbf1972b682`. **Never track `main`** — upstream is `stability: development` with no tags, and tracking `main` red-CIs all 32 packages on a third party's merge.
- Weaver version: `v0.25.0`, run via the `otel/weaver:v0.25.0` container. Do not add a local weaver install as a contributor prerequisite.
- Upstream `semantic-conventions` dependency version: `v1.43.0` (from upstream's `versions.env`).
- `_contract/generated.py` is **committed**. CI regenerates and fails on diff. Tests must never invoke Docker or the network.
- The harness lands **warn-only**. A package flips to enforcing only via an explicit `project.json` tag change.
- Python floor is 3.10 (matches the CI matrix), so `X | None` unions in annotations are fine but `match` statements are the only 3.10+ syntax to avoid gratuitously.
- No task in this plan touches instrumentation source. This spec adds tests and infrastructure only; fixing the violations it uncovers is spec 3's job.

## Baseline Facts (verified 2026-08-02)

These correct the design document's estimates. Verify before relying on them; they age.

- **Zero** of 32 instrumentation packages verify emitted spans against the contract. The 5 packages with `test_semconv*.py` import `opentelemetry.semconv_ai._testing`, whose 441 lines assert only that constants equal string literals (`assert SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS == "gen_ai.usage.total_tokens"`). No span is emitted or inspected, so these cannot catch issues like #4362.
- OpenLLMetry emits attributes in the official `gen_ai.*` namespace that **do not exist upstream**: `gen_ai.usage.total_tokens`, `gen_ai.user`, `gen_ai.headers`, `gen_ai.is_streaming`, `gen_ai.completion`. These need an explicit extension declaration (Task 5), not silent tolerance.
- Resolved registry: 43 groups, 18 of type `span`. Span group ids include `anthropic.inference.client`, `openai.inference.client`, `aws.bedrock.inference.client`, `span.gen_ai.inference.client`, `span.gen_ai.execute_tool.internal`, `span.mcp.client`.
- Resolved group keys: `id`, `type`, `brief`, `note`, `stability`, `attributes`, `span_kind`, `name`, `lineage`.
- Resolved attribute `requirement_level` is either a plain string (`"required"`, `"recommended"`, `"opt_in"`) or a single-key dict (`{"conditionally_required": "<reason>"}`, `{"recommended": "<reason>"}`).
- Resolved attribute `type` is either a string (`"string"`, `"boolean"`, `"int"`, `"any"`) or a dict with `members: [{id, value, ...}]` for enums.

## File Structure

| Path | Responsibility |
|---|---|
| `.semconv/versions.env` | The three pins: contract SHA, weaver version, upstream semconv version |
| `.semconv/Makefile` | `vendor`, `resolve`, `generate`, `check` targets; owns all Docker invocation |
| `.semconv/registry/` | Vendored copy of upstream `model/` at the pinned SHA |
| `.semconv/.build/` | Gitignored scratch: cloned upstream dep, `resolved.json` |
| `.../semconv_ai/_contract/_generator.py` | Pure transform: `resolved.json` → `generated.py`. No network, no Docker. Lives inside the semconv package so `nx run <pkg>:test` covers it. |
| `.../semconv_ai/_contract/__init__.py` | Public types: `Level`, `AttributeSpec`, `SpanSpec` |
| `.../semconv_ai/_contract/generated.py` | **Committed generated artifact.** Never hand-edited. |
| `.../semconv_ai/_contract/extensions.py` | Declared non-standard `gen_ai.*` attributes, with rationale |
| `.../semconv_ai/conformance.py` | The harness: pure checker + pytest adapter |
| `.../semconv_ai/_testing_conformance.py` | Shared per-package test helper, mirroring the existing `_testing.py` precedent |
| `packages/*/tests/test_conformance.py` | One per package: a few lines wiring the shared helper to that package's fixtures |
| `.github/workflows/ci.yml` | New `semconv-contract` job |

The checker is deliberately pure — it takes a `Mapping[str, Any]` of attributes, not an OTel object — so it is testable without spinning up a tracer. A thin adapter converts `ReadableSpan`.

---

### Task 1: Vendoring infrastructure and pins

**Files:**
- Create: `.semconv/versions.env`
- Create: `.semconv/Makefile`
- Create: `.semconv/README.md`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: nothing.
- Produces: `make -C .semconv vendor` populates `.semconv/registry/`. `make -C .semconv resolve` produces `.semconv/.build/resolved.json`.

- [ ] **Step 1: Create the pins file**

Create `.semconv/versions.env`:

```sh
# Pinned upstream GenAI semantic conventions. This is the contract.
# Upstream is `stability: development` and publishes no tags, so we pin a SHA
# and bump it deliberately via the conformance sweep. Never track main.
SEMCONV_GENAI_REPO=https://github.com/open-telemetry/semantic-conventions-genai.git
SEMCONV_GENAI_REF=8484f22ff8069267f37cb1be54bcebbf1972b682

# Weaver, run via container so contributors need no local install.
WEAVER_VERSION=v0.25.0

# Upstream general semantic-conventions, required by the GenAI registry manifest
# for shared attributes (server.*, error.type, ...). Must match the SEMCONV_VERSION
# in the GenAI repo's own versions.env at SEMCONV_GENAI_REF.
SEMCONV_VERSION=v1.43.0
```

- [ ] **Step 2: Create the Makefile**

Create `.semconv/Makefile`. Note `--format json` on `registry resolve`: the command prints a deprecation notice but works correctly on the pinned weaver version, and the pin makes the deprecation inert. Migration to `registry generate` requires authoring a Jinja template and is deferred.

```make
# Vendors the upstream GenAI semantic conventions and resolves them to JSON.
# All Docker invocation lives here. Nothing else in the repo runs weaver.
include versions.env

SELF_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
BUILD    := .build
REGISTRY := registry
UPSTREAM := $(BUILD)/sc-upstream-$(SEMCONV_VERSION)
FILTERED := $(BUILD)/sc-upstream-filtered

WEAVER := docker run --rm -u $(shell id -u):$(shell id -g) \
	-v "$(SELF_DIR):/w" -w /w -e HOME=/tmp otel/weaver:$(WEAVER_VERSION)

.PHONY: vendor resolve generate check clean

## Fetch the pinned registry into registry/ .
vendor:
	rm -rf $(REGISTRY) $(BUILD)
	mkdir -p $(BUILD)
	git clone -q $(SEMCONV_GENAI_REPO) $(BUILD)/genai
	cd $(BUILD)/genai && git checkout -q $(SEMCONV_GENAI_REF)
	cp -r $(BUILD)/genai/model $(REGISTRY)
	rm -rf $(BUILD)/genai
	@echo "vendored $(SEMCONV_GENAI_REF) -> $(REGISTRY)/"

## Build the filtered upstream dependency the registry manifest points at.
$(FILTERED):
	mkdir -p $(BUILD)
	git clone -q --depth 1 --branch $(SEMCONV_VERSION) \
		https://github.com/open-telemetry/semantic-conventions.git $(UPSTREAM)
	mkdir -p $(dir $(FILTERED))
	cp -r $(UPSTREAM)/model $(FILTERED)
	cd $(FILTERED) && rm -rf gen-ai mcp openai

## Resolve the registry to a single JSON document.
resolve: $(FILTERED)
	$(WEAVER) registry resolve -r $(REGISTRY) --format json -o /w/$(BUILD)/resolved.json
	@echo "resolved -> $(BUILD)/resolved.json"

## Regenerate the committed contract module.
generate: resolve
	cd ../packages/opentelemetry-semantic-conventions-ai && uv run python -m \
		opentelemetry.semconv_ai._contract._generator \
		--resolved ../../.semconv/$(BUILD)/resolved.json \
		--out opentelemetry/semconv_ai/_contract/generated.py

## Fail if the committed artifact is stale.
check: generate
	cd .. && git diff --exit-code -- \
		packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/_contract/generated.py

clean:
	rm -rf $(BUILD) $(REGISTRY)/.build
```

- [ ] **Step 3: Gitignore build scratch**

Append to `.gitignore`:

```gitignore
# semconv contract build scratch (vendored registry itself IS committed)
.semconv/.build/
.semconv/registry/.build/
```

- [ ] **Step 4: Run the vendor and resolve targets**

Run: `make -C .semconv vendor && make -C .semconv resolve`

Expected: `.semconv/registry/manifest.yaml` exists, and `.semconv/.build/resolved.json` is roughly 900KB. Weaver prints two warnings (`span.gen_ai.fetch_response.client does not set requirement_level`, and `File format definition/2 is not yet stable`) and a deprecation notice. All three are expected; exit code is 0.

- [ ] **Step 5: Verify the resolved shape**

Run:

```bash
python3 -c "
import json; d=json.load(open('.semconv/.build/resolved.json'))
spans=[g for g in d['groups'] if g.get('type')=='span']
print('groups', len(d['groups']), 'spans', len(spans))
assert len(spans) >= 18, spans
assert any(s['id']=='anthropic.inference.client' for s in spans)
print('OK')
"
```

Expected: `groups 43 spans 18` then `OK`.

- [ ] **Step 6: Write the README**

Create `.semconv/README.md`:

```markdown
# Vendored GenAI semantic conventions

This directory vendors https://github.com/open-telemetry/semantic-conventions-genai
at the SHA pinned in `versions.env`. It is the contract every instrumentation
package is tested against.

`registry/` is committed. `.build/` is scratch and gitignored.

## Commands

    make -C .semconv vendor     # refetch the registry at the pinned SHA
    make -C .semconv resolve    # -> .build/resolved.json (needs Docker)
    make -C .semconv generate   # regenerate the committed contract module
    make -C .semconv check      # fail if the committed module is stale

## Bumping the pin

Change `SEMCONV_GENAI_REF`, then run `vendor` and `generate`, and commit the
result. The diff on `_contract/generated.py` shows exactly what changed in the
contract, and the conformance suite shows which packages fell out of compliance.
Bump deliberately and never track `main`: upstream is `stability: development`
with no tags.
```

- [ ] **Step 7: Commit**

```bash
git add .semconv .gitignore
git commit -m "feat(semconv): vendor pinned GenAI semantic conventions registry"
```

---

### Task 2: Contract data model

**Files:**
- Create: `packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/_contract/__init__.py`
- Test: `packages/opentelemetry-semantic-conventions-ai/tests/test_contract_model.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Level` (str enum with members `REQUIRED`, `CONDITIONALLY_REQUIRED`, `RECOMMENDED`, `OPT_IN`), `AttributeSpec(name: str, level: Level, condition: str | None, enum_members: tuple[str, ...] | None)`, `SpanSpec(id: str, span_kind: str | None, attributes: tuple[AttributeSpec, ...])` with method `required() -> tuple[AttributeSpec, ...]`, and `parse_requirement_level(raw) -> tuple[Level, str | None]`. Task 3 imports all of these; Task 4 consumes `SpanSpec`.

- [ ] **Step 1: Write the failing test**

Create `packages/opentelemetry-semantic-conventions-ai/tests/test_contract_model.py`:

```python
import pytest

from opentelemetry.semconv_ai._contract import (
    AttributeSpec,
    Level,
    SpanSpec,
    parse_requirement_level,
)


class TestParseRequirementLevel:
    def test_plain_string_required(self):
        assert parse_requirement_level("required") == (Level.REQUIRED, None)

    def test_plain_string_opt_in(self):
        assert parse_requirement_level("opt_in") == (Level.OPT_IN, None)

    def test_dict_conditionally_required_carries_condition(self):
        raw = {"conditionally_required": "If the operation ended in an error."}
        assert parse_requirement_level(raw) == (
            Level.CONDITIONALLY_REQUIRED,
            "If the operation ended in an error.",
        )

    def test_dict_recommended_carries_condition(self):
        raw = {"recommended": "when available"}
        assert parse_requirement_level(raw) == (Level.RECOMMENDED, "when available")

    def test_unknown_level_raises(self):
        with pytest.raises(ValueError, match="unknown requirement_level"):
            parse_requirement_level("sometimes_maybe")


class TestSpanSpec:
    def test_required_filters_to_required_only(self):
        spec = SpanSpec(
            id="span.gen_ai.inference.client",
            span_kind="client",
            attributes=(
                AttributeSpec("gen_ai.operation.name", Level.REQUIRED, None, None),
                AttributeSpec("gen_ai.input.messages", Level.OPT_IN, None, None),
                AttributeSpec("error.type", Level.CONDITIONALLY_REQUIRED, "on error", None),
            ),
        )
        assert [a.name for a in spec.required()] == ["gen_ai.operation.name"]

    def test_is_hashable_and_frozen(self):
        spec = AttributeSpec("gen_ai.operation.name", Level.REQUIRED, None, None)
        hash(spec)
        with pytest.raises(Exception):
            spec.name = "other"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/opentelemetry-semantic-conventions-ai && uv run pytest tests/test_contract_model.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'opentelemetry.semconv_ai._contract'`

- [ ] **Step 3: Write the implementation**

Create `packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/_contract/__init__.py`:

```python
"""Data model for the OpenTelemetry GenAI semantic-convention contract.

The concrete contract lives in the generated sibling module `generated.py`,
which is produced by `make -C .semconv generate` and committed. Nothing here
reads it; these are plain value types so the conformance harness can be tested
without the generated artifact.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Tuple, Union


class Level(str, Enum):
    """Requirement level of an attribute on a span, per the semconv spec."""

    REQUIRED = "required"
    CONDITIONALLY_REQUIRED = "conditionally_required"
    RECOMMENDED = "recommended"
    OPT_IN = "opt_in"


def parse_requirement_level(
    raw: Union[str, Mapping[str, str]],
) -> Tuple[Level, Optional[str]]:
    """Normalise weaver's two encodings of requirement_level.

    Weaver emits either a bare string ("required", "opt_in") or a single-key
    mapping carrying the condition ({"conditionally_required": "If ..."}).
    """
    if isinstance(raw, str):
        try:
            return Level(raw), None
        except ValueError:
            raise ValueError(f"unknown requirement_level: {raw!r}") from None

    if isinstance(raw, Mapping) and len(raw) == 1:
        key, condition = next(iter(raw.items()))
        try:
            return Level(key), condition
        except ValueError:
            raise ValueError(f"unknown requirement_level: {key!r}") from None

    raise ValueError(f"unknown requirement_level: {raw!r}")


@dataclass(frozen=True)
class AttributeSpec:
    """One attribute as the contract defines it for a given span."""

    name: str
    level: Level
    condition: Optional[str] = None
    enum_members: Optional[Tuple[str, ...]] = None


@dataclass(frozen=True)
class SpanSpec:
    """One span group from the contract."""

    id: str
    span_kind: Optional[str]
    attributes: Tuple[AttributeSpec, ...]

    def required(self) -> Tuple[AttributeSpec, ...]:
        return tuple(a for a in self.attributes if a.level is Level.REQUIRED)

    def by_name(self, name: str) -> Optional[AttributeSpec]:
        for a in self.attributes:
            if a.name == name:
                return a
        return None


def enum_members_of(attr_type: Any) -> Optional[Tuple[str, ...]]:
    """Extract enum values from weaver's attribute `type` field, if it is an enum."""
    if isinstance(attr_type, Mapping) and "members" in attr_type:
        return tuple(str(m["value"]) for m in attr_type["members"])
    return None


__all__ = [
    "AttributeSpec",
    "Level",
    "SpanSpec",
    "enum_members_of",
    "parse_requirement_level",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/opentelemetry-semantic-conventions-ai && uv run pytest tests/test_contract_model.py -v`

Expected: PASS, 7 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/_contract/__init__.py \
        packages/opentelemetry-semantic-conventions-ai/tests/test_contract_model.py
git commit -m "feat(semconv): add contract data model"
```

---

### Task 3: Contract generator

**Files:**
- Create: `packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/_contract/_generator.py`
- Create: `packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/_contract/generated.py` (via the generator, not by hand)
- Test: `packages/opentelemetry-semantic-conventions-ai/tests/test_generator.py`

**Placement note:** the generator lives inside the semconv package rather than a
repo-root `scripts/` directory. The repo has no root-level Python project — no root
`pyproject.toml`, no root `tests/`, and Nx orchestrates everything per-package — so a
root-level test would never run in CI. As a package module it is covered by the existing
`nx run opentelemetry-semantic-conventions-ai:test` target with no new CI job.

**Interfaces:**
- Consumes: `Level`, `AttributeSpec`, `SpanSpec`, `enum_members_of`, `parse_requirement_level` from Task 2.
- Produces: `build_specs(resolved: dict) -> dict[str, SpanSpec]` and `render(specs, ref: str) -> str` in `opentelemetry.semconv_ai._contract._generator`, runnable as `python -m opentelemetry.semconv_ai._contract._generator`. The generated module exposes `SPANS: dict[str, SpanSpec]` and `CONTRACT_REF: str`. Task 4 imports `SPANS`.

- [ ] **Step 1: Write the failing test**

Create `packages/opentelemetry-semantic-conventions-ai/tests/test_generator.py`:

```python
import importlib.util
import sys

import pytest

from opentelemetry.semconv_ai._contract import Level
from opentelemetry.semconv_ai._contract._generator import build_specs, render


RESOLVED_FIXTURE = {
    "registry_url": "registry",
    "groups": [
        {
            "id": "span.gen_ai.inference.client",
            "type": "span",
            "span_kind": "client",
            "attributes": [
                {"name": "gen_ai.operation.name", "requirement_level": "required",
                 "type": {"members": [{"id": "chat", "value": "chat"},
                                      {"id": "embeddings", "value": "embeddings"}]}},
                {"name": "gen_ai.request.model",
                 "requirement_level": {"conditionally_required": "If available."},
                 "type": "string"},
                {"name": "gen_ai.input.messages", "requirement_level": "opt_in",
                 "type": "any"},
            ],
        },
        {
            "id": "registry.gen_ai",
            "type": "attribute_group",
            "attributes": [{"name": "ignored", "requirement_level": "required"}],
        },
    ],
}


class TestBuildSpecs:
    def test_keeps_only_span_groups(self):
        specs = build_specs(RESOLVED_FIXTURE)
        assert set(specs) == {"span.gen_ai.inference.client"}

    def test_captures_span_kind(self):
        specs = build_specs(RESOLVED_FIXTURE)
        assert specs["span.gen_ai.inference.client"].span_kind == "client"

    def test_normalises_requirement_levels(self):
        spec = build_specs(RESOLVED_FIXTURE)["span.gen_ai.inference.client"]
        assert spec.by_name("gen_ai.operation.name").level is Level.REQUIRED
        assert spec.by_name("gen_ai.request.model").level is Level.CONDITIONALLY_REQUIRED
        assert spec.by_name("gen_ai.request.model").condition == "If available."
        assert spec.by_name("gen_ai.input.messages").level is Level.OPT_IN

    def test_captures_enum_members(self):
        spec = build_specs(RESOLVED_FIXTURE)["span.gen_ai.inference.client"]
        assert spec.by_name("gen_ai.operation.name").enum_members == ("chat", "embeddings")
        assert spec.by_name("gen_ai.request.model").enum_members is None

    def test_attributes_are_sorted_for_stable_diffs(self):
        spec = build_specs(RESOLVED_FIXTURE)["span.gen_ai.inference.client"]
        names = [a.name for a in spec.attributes]
        assert names == sorted(names)

    def test_group_without_requirement_level_defaults_to_recommended(self):
        resolved = {"groups": [{
            "id": "span.x", "type": "span", "span_kind": "client",
            "attributes": [{"name": "a.b", "type": "string"}],
        }]}
        spec = build_specs(resolved)["span.x"]
        assert spec.by_name("a.b").level is Level.RECOMMENDED


class TestRender:
    def test_output_is_importable_and_round_trips(self, tmp_path):
        specs = build_specs(RESOLVED_FIXTURE)
        out = tmp_path / "generated.py"
        out.write_text(render(specs, ref="deadbeef"))

        sys.path.insert(0, str(tmp_path))
        spec_mod = importlib.util.spec_from_file_location("generated", out)
        mod = importlib.util.module_from_spec(spec_mod)
        spec_mod.loader.exec_module(mod)

        assert mod.CONTRACT_REF == "deadbeef"
        assert set(mod.SPANS) == {"span.gen_ai.inference.client"}
        got = mod.SPANS["span.gen_ai.inference.client"]
        assert got.by_name("gen_ai.operation.name").level is Level.REQUIRED
        assert got.by_name("gen_ai.operation.name").enum_members == ("chat", "embeddings")

    def test_is_deterministic(self):
        specs = build_specs(RESOLVED_FIXTURE)
        assert render(specs, ref="x") == render(specs, ref="x")

    def test_carries_do_not_edit_banner(self):
        out = render(build_specs(RESOLVED_FIXTURE), ref="x")
        assert "DO NOT EDIT" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/opentelemetry-semantic-conventions-ai && uv run pytest tests/test_generator.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'opentelemetry.semconv_ai._contract._generator'`

- [ ] **Step 3: Write the generator**

Create `packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/_contract/_generator.py`:

```python
"""Generate the committed semconv contract module from weaver's resolved JSON.

Pure transform: reads a JSON file, writes a Python file. No network, no Docker.
Invoked by `make -C .semconv generate` as:

    python -m opentelemetry.semconv_ai._contract._generator --resolved ... --out ...
"""

import argparse
import json
from pathlib import Path
from typing import Dict

from opentelemetry.semconv_ai._contract import (
    AttributeSpec,
    Level,
    SpanSpec,
    enum_members_of,
    parse_requirement_level,
)

BANNER = '''"""The OpenTelemetry GenAI semantic-convention contract, as Python.

DO NOT EDIT. Generated by opentelemetry.semconv_ai._contract._generator from the
registry vendored in .semconv/, pinned at the SHA below. Regenerate with:

    make -C .semconv generate
"""

from opentelemetry.semconv_ai._contract import AttributeSpec, Level, SpanSpec

CONTRACT_REF = "{ref}"

SPANS = {{
'''


def build_specs(resolved: Dict) -> Dict[str, SpanSpec]:
    """Convert weaver's resolved registry into SpanSpec objects, keyed by group id."""
    specs: Dict[str, SpanSpec] = {}

    for group in resolved.get("groups", []):
        if group.get("type") != "span":
            continue

        attributes = []
        for attr in group.get("attributes", []):
            # Weaver back-fills requirement_level during resolution, so every
            # resolved attribute carries one. A missing key means upstream changed
            # shape — fail loudly rather than silently downgrading a required
            # attribute to non-blocking in the conformance harness.
            try:
                raw_level = attr["requirement_level"]
            except KeyError:
                raise SystemExit(
                    f"attribute {attr.get('name')!r} in group {group['id']!r} has no "
                    "requirement_level; the resolved registry changed shape"
                ) from None
            level, condition = parse_requirement_level(raw_level)
            attributes.append(
                AttributeSpec(
                    name=attr["name"],
                    level=level,
                    condition=condition,
                    enum_members=enum_members_of(attr.get("type")),
                )
            )

        # Sorted so regeneration produces a stable, reviewable diff.
        attributes.sort(key=lambda a: a.name)

        specs[group["id"]] = SpanSpec(
            id=group["id"],
            span_kind=group.get("span_kind"),
            attributes=tuple(attributes),
        )

    return specs


def _render_attribute(attr: AttributeSpec) -> str:
    members = (
        "None"
        if attr.enum_members is None
        else "(" + "".join(f"{m!r}, " for m in attr.enum_members) + ")"
    )
    return (
        f"            AttributeSpec({attr.name!r}, Level.{attr.level.name}, "
        f"{attr.condition!r}, {members}),\n"
    )


def render(specs: Dict[str, SpanSpec], ref: str) -> str:
    """Render the specs as an importable Python module. Deterministic."""
    out = [BANNER.format(ref=ref)]

    for span_id in sorted(specs):
        spec = specs[span_id]
        out.append(f"    {span_id!r}: SpanSpec(\n")
        out.append(f"        id={spec.id!r},\n")
        out.append(f"        span_kind={spec.span_kind!r},\n")
        out.append("        attributes=(\n")
        for attr in spec.attributes:
            out.append(_render_attribute(attr))
        out.append("        ),\n")
        out.append("    ),\n")

    out.append("}\n\n__all__ = [\"CONTRACT_REF\", \"SPANS\"]\n")
    return "".join(out)


def _pinned_ref() -> str:
    # .../packages/<pkg>/opentelemetry/semconv_ai/_contract/_generator.py -> repo root
    repo_root = Path(__file__).resolve().parents[5]
    versions = repo_root / ".semconv" / "versions.env"
    for line in versions.read_text().splitlines():
        if line.startswith("SEMCONV_GENAI_REF="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("SEMCONV_GENAI_REF not found in .semconv/versions.env")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    resolved = json.loads(args.resolved.read_text())
    specs = build_specs(resolved)
    if not specs:
        raise SystemExit("no span groups found — is the resolved registry correct?")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(specs, ref=_pinned_ref()))
    print(f"wrote {args.out} ({len(specs)} span groups)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/opentelemetry-semantic-conventions-ai && uv run pytest tests/test_generator.py -v`

Expected: PASS, 9 passed.

- [ ] **Step 5: Generate the real contract**

Run: `make -C .semconv generate`

Expected: `wrote packages/.../\_contract/generated.py (18 span groups)`

- [ ] **Step 6: Sanity-check the generated artifact**

Run:

```bash
cd packages/opentelemetry-semantic-conventions-ai && uv run python -c "
from opentelemetry.semconv_ai._contract.generated import SPANS, CONTRACT_REF
print('ref', CONTRACT_REF)
print('spans', len(SPANS))
s = SPANS['anthropic.inference.client']
print('required:', [a.name for a in s.required()])
"
```

Expected: `ref 8484f22f...`, `spans 18`, and `required: ['gen_ai.operation.name', ...]`.

- [ ] **Step 7: Verify `make check` passes on a clean tree**

Run: `make -C .semconv check`

Expected: exit 0, no diff output.

- [ ] **Step 8: Commit**

```bash
git add packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/_contract/_generator.py \
        packages/opentelemetry-semantic-conventions-ai/tests/test_generator.py \
        packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/_contract/generated.py
git commit -m "feat(semconv): generate contract module from vendored registry"
```

---

### Task 4: Conformance harness

> **AMENDED AFTER IMPLEMENTATION (owner-approved).** The code block below is the
> ORIGINAL design and is superseded. Two Critical flaws were found in review and
> fixed: (1) only 33/372 contract attributes are `required` and
> `gen_ai.response.finish_reasons` is `recommended` everywhere, so #4362 was
> undetectable — a per-package `expected` frozenset was added, and a missing
> promised attribute is a blocking `missing_expected` violation; (2) OTel enums are
> open, and `gen_ai.provider.name` is `required` + closed to 16 values with no member
> for ollama/together/replicate — `bad_enum_value` became non-blocking
> `unknown_enum_value`, excluded from `BLOCKING_KINDS`. Read the committed
> `conformance.py` for the authoritative implementation, not the block below.

**Files:**
- Create: `packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/conformance.py`
- Test: `packages/opentelemetry-semantic-conventions-ai/tests/test_conformance.py`

**Interfaces:**
- Consumes: `Level`, `AttributeSpec`, `SpanSpec` (Task 2); `SPANS` (Task 3).
- Produces: `Violation(kind, attribute, detail)`; `check_attributes(attributes, spec, extensions=frozenset(), expected=frozenset()) -> list[Violation]`; `ConformanceWarning`; `assert_conforms(span, group_id, *, enforcing, extensions=frozenset(), expected=frozenset(), _spans=None) -> list[Violation]`. Tasks 5–7 consume `assert_conforms`, `extensions`, and `expected`.

Violation `kind` is one of exactly four strings: `"missing_required"`, `"missing_expected"`, `"undeclared_gen_ai"` (all blocking), and `"unknown_enum_value"` (non-blocking). `BLOCKING_KINDS` excludes `"unknown_enum_value"`.

- [ ] **Step 1: Write the failing test**

Create `packages/opentelemetry-semantic-conventions-ai/tests/test_conformance.py`:

```python
import pytest

from opentelemetry.semconv_ai._contract import AttributeSpec, Level, SpanSpec
from opentelemetry.semconv_ai.conformance import (
    ConformanceWarning,
    Violation,
    assert_conforms,
    check_attributes,
)

SPEC = SpanSpec(
    id="span.gen_ai.inference.client",
    span_kind="client",
    attributes=(
        AttributeSpec("gen_ai.operation.name", Level.REQUIRED, None, ("chat", "embeddings")),
        AttributeSpec("gen_ai.request.model", Level.CONDITIONALLY_REQUIRED, "If available.", None),
        AttributeSpec("gen_ai.usage.input_tokens", Level.RECOMMENDED, None, None),
        AttributeSpec("gen_ai.input.messages", Level.OPT_IN, None, None),
    ),
)


class TestCheckAttributes:
    def test_conforming_span_yields_no_violations(self):
        attrs = {"gen_ai.operation.name": "chat", "gen_ai.request.model": "claude-opus-4"}
        assert check_attributes(attrs, SPEC) == []

    def test_missing_required_is_a_violation(self):
        violations = check_attributes({"gen_ai.request.model": "x"}, SPEC)
        assert [v.kind for v in violations] == ["missing_required"]
        assert violations[0].attribute == "gen_ai.operation.name"

    def test_missing_conditionally_required_is_not_a_violation(self):
        # The condition is prose the harness cannot evaluate, so absence is tolerated.
        violations = check_attributes({"gen_ai.operation.name": "chat"}, SPEC)
        assert violations == []

    def test_missing_recommended_and_opt_in_are_not_violations(self):
        violations = check_attributes({"gen_ai.operation.name": "chat"}, SPEC)
        assert violations == []

    def test_undeclared_gen_ai_attribute_is_a_violation(self):
        attrs = {"gen_ai.operation.name": "chat", "gen_ai.is_streaming": True}
        violations = check_attributes(attrs, SPEC)
        assert [v.kind for v in violations] == ["undeclared_gen_ai"]
        assert violations[0].attribute == "gen_ai.is_streaming"

    def test_declared_extension_is_tolerated(self):
        attrs = {"gen_ai.operation.name": "chat", "gen_ai.is_streaming": True}
        violations = check_attributes(attrs, SPEC, extensions=frozenset({"gen_ai.is_streaming"}))
        assert violations == []

    def test_non_gen_ai_namespace_is_ignored(self):
        # traceloop.*, llm.*, db.* are outside the contract's scope entirely.
        attrs = {"gen_ai.operation.name": "chat", "traceloop.workflow.name": "w", "llm.vendor": "x"}
        assert check_attributes(attrs, SPEC) == []

    def test_bad_enum_value_is_a_violation(self):
        attrs = {"gen_ai.operation.name": "definitely-not-a-real-operation"}
        violations = check_attributes(attrs, SPEC)
        assert [v.kind for v in violations] == ["bad_enum_value"]

    def test_violations_are_sorted_by_attribute_for_stable_output(self):
        attrs = {"gen_ai.zzz": 1, "gen_ai.aaa": 1}
        violations = check_attributes(attrs, SPEC)
        names = [v.attribute for v in violations]
        assert names == sorted(names)


class FakeSpan:
    def __init__(self, attributes):
        self.attributes = attributes
        self.name = "fake"


class TestAssertConforms:
    def test_enforcing_mode_raises_on_violation(self):
        with pytest.raises(AssertionError, match="missing_required"):
            assert_conforms(
                FakeSpan({"gen_ai.request.model": "x"}),
                "span.gen_ai.inference.client",
                enforcing=True,
                _spans={"span.gen_ai.inference.client": SPEC},
            )

    def test_warn_mode_warns_and_returns_violations(self):
        with pytest.warns(ConformanceWarning, match="gen_ai.operation.name"):
            violations = assert_conforms(
                FakeSpan({"gen_ai.request.model": "x"}),
                "span.gen_ai.inference.client",
                enforcing=False,
                _spans={"span.gen_ai.inference.client": SPEC},
            )
        assert [v.kind for v in violations] == ["missing_required"]

    def test_warn_mode_is_silent_when_conforming(self, recwarn):
        violations = assert_conforms(
            FakeSpan({"gen_ai.operation.name": "chat"}),
            "span.gen_ai.inference.client",
            enforcing=False,
            _spans={"span.gen_ai.inference.client": SPEC},
        )
        assert violations == []
        assert not [w for w in recwarn if issubclass(w.category, ConformanceWarning)]

    def test_unknown_group_id_raises_regardless_of_mode(self):
        with pytest.raises(KeyError, match="no such span group"):
            assert_conforms(FakeSpan({}), "span.nope", enforcing=False, _spans={})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/opentelemetry-semantic-conventions-ai && uv run pytest tests/test_conformance.py -v`

Expected: FAIL — `ImportError: cannot import name 'ConformanceWarning'`

- [ ] **Step 3: Write the harness**

Create `packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/conformance.py`:

```python
"""Check emitted spans against the OpenTelemetry GenAI semantic-convention contract.

`check_attributes` is pure — it takes a mapping, not an OTel object — so it can
be tested without a tracer. `assert_conforms` is the pytest-facing adapter.

Scope: only the `gen_ai.*` namespace is checked. Attributes in other namespaces
(`traceloop.*`, `llm.*`, `db.*`) are outside this contract and ignored.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Mapping, Optional

from opentelemetry.semconv_ai._contract import Level, SpanSpec
from opentelemetry.semconv_ai._contract.generated import SPANS

GEN_AI_PREFIX = "gen_ai."


class ConformanceWarning(UserWarning):
    """Raised as a warning while a package is still in warn-only mode."""


@dataclass(frozen=True)
class Violation:
    kind: str  # missing_required | undeclared_gen_ai | bad_enum_value
    attribute: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.kind}] {self.attribute}: {self.detail}"


def check_attributes(
    attributes: Mapping[str, Any],
    spec: SpanSpec,
    extensions: FrozenSet[str] = frozenset(),
) -> List[Violation]:
    """Return every way `attributes` violates `spec`. Empty list means conforming."""
    violations: List[Violation] = []

    for attr in spec.required():
        if attr.name not in attributes:
            violations.append(
                Violation(
                    "missing_required",
                    attr.name,
                    f"required by {spec.id} but not present on the span",
                )
            )

    for name, value in attributes.items():
        if not name.startswith(GEN_AI_PREFIX):
            continue  # outside the contract's namespace

        declared = spec.by_name(name)
        if declared is None:
            if name in extensions:
                continue
            violations.append(
                Violation(
                    "undeclared_gen_ai",
                    name,
                    "not defined by the contract for this span. Either use a "
                    "contract attribute or declare it in _contract/extensions.py",
                )
            )
            continue

        if declared.enum_members and value not in declared.enum_members:
            violations.append(
                Violation(
                    "bad_enum_value",
                    name,
                    f"value {value!r} not in {list(declared.enum_members)}",
                )
            )

    violations.sort(key=lambda v: (v.attribute, v.kind))
    return violations


def assert_conforms(
    span: Any,
    group_id: str,
    *,
    enforcing: bool,
    extensions: FrozenSet[str] = frozenset(),
    _spans: Optional[Dict[str, SpanSpec]] = None,
) -> List[Violation]:
    """Check one span against a contract group.

    Enforcing mode fails the test on any violation. Warn-only mode emits a
    ConformanceWarning and returns the violations, so a package can adopt the
    harness before it is clean. `_spans` is a seam for testing this module.
    """
    table = SPANS if _spans is None else _spans
    if group_id not in table:
        raise KeyError(f"no such span group in the contract: {group_id!r}")

    violations = check_attributes(
        dict(span.attributes or {}), table[group_id], extensions
    )
    if not violations:
        return []

    report = "\n".join(f"  {v}" for v in violations)
    message = (
        f"span {getattr(span, 'name', '<unnamed>')!r} violates {group_id}:\n{report}"
    )

    if enforcing:
        raise AssertionError(message)

    warnings.warn(message, ConformanceWarning, stacklevel=2)
    return violations


__all__ = [
    "ConformanceWarning",
    "Violation",
    "assert_conforms",
    "check_attributes",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/opentelemetry-semantic-conventions-ai && uv run pytest tests/test_conformance.py -v`

Expected: PASS, 13 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/conformance.py \
        packages/opentelemetry-semantic-conventions-ai/tests/test_conformance.py
git commit -m "feat(semconv): add span conformance harness"
```

---

### Task 5: Declared extensions registry

**Files:**
- Create: `packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/_contract/extensions.py`
- Test: `packages/opentelemetry-semantic-conventions-ai/tests/test_extensions.py`

**Interfaces:**
- Consumes: `SPANS` (Task 3).
- Produces: `EXTENSIONS: frozenset[str]` and `EXTENSION_RATIONALE: dict[str, str]`. Tasks 6–7 pass `EXTENSIONS` to `assert_conforms`.

**Why this task exists:** OpenLLMetry emits `gen_ai.*` attributes that do not exist upstream (`gen_ai.usage.total_tokens`, `gen_ai.user`, `gen_ai.headers`, `gen_ai.is_streaming`, `gen_ai.completion`). Squatting the official namespace means downstream consumers read non-standard data as if it were conventional. This file makes each one a deliberate, documented decision with an exit path rather than an accident.

- [ ] **Step 1: Write the failing test**

Create `packages/opentelemetry-semantic-conventions-ai/tests/test_extensions.py`:

```python
from opentelemetry.semconv_ai._contract.extensions import (
    EXTENSION_RATIONALE,
    EXTENSIONS,
)
from opentelemetry.semconv_ai._contract.generated import SPANS


class TestExtensions:
    def test_every_extension_has_a_rationale(self):
        assert set(EXTENSIONS) == set(EXTENSION_RATIONALE)

    def test_rationales_are_non_empty(self):
        assert all(v.strip() for v in EXTENSION_RATIONALE.values())

    def test_all_extensions_are_in_the_gen_ai_namespace(self):
        # Anything outside gen_ai.* needs no declaration; the harness ignores it.
        assert all(name.startswith("gen_ai.") for name in EXTENSIONS)

    def test_no_extension_shadows_a_contract_attribute(self):
        """An extension that upstream has since defined must be removed, not kept.

        This is the mechanism that stops the extension list becoming permanent:
        when a pin bump adds one of these upstream, this test fails and forces
        us to delete the extension and conform.
        """
        contract_names = {
            attr.name for spec in SPANS.values() for attr in spec.attributes
        }
        shadowed = sorted(set(EXTENSIONS) & contract_names)
        assert not shadowed, (
            f"these are now defined by the contract and must be removed from "
            f"EXTENSIONS: {shadowed}"
        )

    def test_known_openllmetry_extensions_are_declared(self):
        for name in (
            "gen_ai.usage.total_tokens",
            "gen_ai.user",
            "gen_ai.headers",
            "gen_ai.is_streaming",
            "gen_ai.completion",
        ):
            assert name in EXTENSIONS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/opentelemetry-semantic-conventions-ai && uv run pytest tests/test_extensions.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'opentelemetry.semconv_ai._contract.extensions'`

- [ ] **Step 3: Write the extensions registry**

Create `packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/_contract/extensions.py`:

```python
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
        "Whether the request used streaming. Upstream expresses this through span "
        "structure rather than an attribute."
    ),
    "gen_ai.completion": (
        "Legacy prompt/completion content attribute, predating upstream's "
        "gen_ai.output.messages. Retained for backwards compatibility; migration "
        "to gen_ai.output.messages is the intended path."
    ),
}

EXTENSIONS = frozenset(EXTENSION_RATIONALE)

__all__ = ["EXTENSIONS", "EXTENSION_RATIONALE"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/opentelemetry-semantic-conventions-ai && uv run pytest tests/test_extensions.py -v`

Expected: PASS, 5 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/_contract/extensions.py \
        packages/opentelemetry-semantic-conventions-ai/tests/test_extensions.py
git commit -m "feat(semconv): declare non-standard gen_ai.* extensions"
```

---

### Task 6: Wire the harness into one package (anthropic), warn-only

**Files:**
- Create: `packages/opentelemetry-instrumentation-anthropic/tests/test_conformance.py`
- Modify: `packages/opentelemetry-instrumentation-anthropic/pyproject.toml` (dev dependency on the semconv package, if not already present)

**Interfaces:**
- Consumes: `assert_conforms`, `Violation` (Task 4); `EXTENSIONS` (Task 5).
- Produces: the per-package test pattern that Task 7 replicates across the other 31 packages.

**Why anthropic first:** it has the richest existing cassette coverage (streaming, thinking, prompt caching, tool use), and issue #4362 is an anthropic streaming conformance bug — so this package can demonstrate the harness catching a real, reported defect.

- [ ] **Step 1: Confirm the fixture for capturing spans**

Run: `cd packages/opentelemetry-instrumentation-anthropic && sed -n '1,60p' tests/conftest.py`

Expected: a fixture exporting spans (typically an `InMemorySpanExporter` behind a `span_exporter` fixture). Note its exact name — the next step uses it. If the fixture is named differently, substitute that name throughout this task.

- [ ] **Step 2: Write the conformance test**

Create `packages/opentelemetry-instrumentation-anthropic/tests/test_conformance.py`:

```python
"""Conformance of emitted spans against the OTel GenAI semantic conventions.

Warn-only until this package is listed as enforcing (see project.json tags).
Flipping the switch is a one-line change to ENFORCING below plus the tag.
"""

import pytest
from opentelemetry.semconv_ai._contract.extensions import EXTENSIONS
from opentelemetry.semconv_ai.conformance import assert_conforms

# Flip to True once this package emits conforming spans. Keep in sync with the
# semconv:enforcing tag in project.json.
ENFORCING = False

CONTRACT_GROUP = "anthropic.inference.client"


def _gen_ai_spans(exporter):
    return [
        s
        for s in exporter.get_finished_spans()
        if (s.attributes or {}).get("gen_ai.operation.name") is not None
    ]


@pytest.mark.vcr
def test_messages_span_conforms(instrument_legacy, anthropic_client, span_exporter):
    anthropic_client.messages.create(
        max_tokens=64,
        messages=[{"role": "user", "content": "Tell me a joke about OpenTelemetry"}],
        model="claude-3-5-sonnet-20240620",
    )

    spans = _gen_ai_spans(span_exporter)
    assert spans, "expected at least one gen_ai span"

    for span in spans:
        assert_conforms(
            span, CONTRACT_GROUP, enforcing=ENFORCING, extensions=EXTENSIONS
        )


@pytest.mark.vcr
def test_streaming_span_conforms(instrument_legacy, anthropic_client, span_exporter):
    """Streaming is where #4362 reports dropped attributes."""
    stream = anthropic_client.messages.create(
        max_tokens=64,
        messages=[{"role": "user", "content": "Tell me a joke about OpenTelemetry"}],
        model="claude-3-5-sonnet-20240620",
        stream=True,
    )
    for _ in stream:
        pass

    spans = _gen_ai_spans(span_exporter)
    assert spans, "expected at least one gen_ai span"

    for span in spans:
        assert_conforms(
            span, CONTRACT_GROUP, enforcing=ENFORCING, extensions=EXTENSIONS
        )
```

Adapt the client fixture name and model to whatever `tests/conftest.py` and the existing `tests/test_messages.py` use. Reuse an existing cassette rather than recording: this task must not require API keys.

- [ ] **Step 3: Run and capture the warnings**

Run: `cd packages/opentelemetry-instrumentation-anthropic && uv run pytest tests/test_conformance.py -v -W "always::UserWarning" 2>&1 | tail -40`

Expected: tests PASS (warn-only mode never fails), with `ConformanceWarning` output listing real violations. **Record that output** — it is the baseline of what spec 3 will fix.

- [ ] **Step 4: Verify the harness actually detects a violation**

This step proves the harness is not vacuous. Temporarily set `ENFORCING = True`, rerun, and confirm the tests now FAIL with a `missing_required` or `undeclared_gen_ai` assertion. Then set it back to `False`.

Run:

```bash
cd packages/opentelemetry-instrumentation-anthropic
sed -i '' 's/^ENFORCING = False/ENFORCING = True/' tests/test_conformance.py
uv run pytest tests/test_conformance.py -v 2>&1 | tail -20
sed -i '' 's/^ENFORCING = True/ENFORCING = False/' tests/test_conformance.py
```

Expected: FAIL under enforcing with a readable violation report, then the file is restored to `ENFORCING = False`.

If the tests PASS under enforcing, the harness found nothing — stop and investigate. Either the span attributes are not being read (wrong fixture), or `CONTRACT_GROUP` is wrong. A harness that reports zero violations against a package with known conformance bugs is broken, not clean.

- [ ] **Step 5: Ensure the semconv package is a test dependency**

Run: `grep -n "semantic-conventions-ai" packages/opentelemetry-instrumentation-anthropic/pyproject.toml`

Expected: a dependency line already exists. If it does not, add it to the dev dependency group and run `uv lock`.

- [ ] **Step 6: Run the package's full suite for regressions**

Run: `npx nx run opentelemetry-instrumentation-anthropic:test`

Expected: PASS. The new file adds tests; it must not break existing ones.

- [ ] **Step 7: Commit**

```bash
git add packages/opentelemetry-instrumentation-anthropic/tests/test_conformance.py
git commit -m "test(anthropic): add warn-only semconv conformance checks"
```

---

### Task 7: Roll out across the remaining packages

**Files:**
- Create: `packages/<pkg>/tests/test_conformance.py` for each instrumentation package with a mappable contract group
- Modify: `packages/<pkg>/project.json` — add a `semconv:warn` tag
- Create: `docs/ai/semconv-rollout.md`

**Interfaces:**
- Consumes: the pattern from Task 6.
- Produces: `semconv:warn` / `semconv:enforcing` tags in `project.json`, making the migration frontier greppable.

**Group mapping.** Packages map to contract groups as follows. Packages with no upstream group use the generic `span.gen_ai.inference.client`.

| Package | Contract group |
|---|---|
| anthropic | `anthropic.inference.client` |
| openai, openai-agents | `openai.inference.client` |
| bedrock, sagemaker | `aws.bedrock.inference.client` |
| mcp | `span.mcp.client` |
| all other inference instrumentations | `span.gen_ai.inference.client` |

Vector-store packages (chromadb, pinecone, qdrant, weaviate, milvus, marqo, lancedb) emit `db.*`, not `gen_ai.*`, and are **out of scope** — the GenAI contract does not cover them. Do not add conformance tests there.

- [ ] **Step 1: Confirm the in-scope package list**

Run:

```bash
cd /Users/gal.kleinman/dev/openllmetry
for p in packages/opentelemetry-instrumentation-*; do
  n=$(basename "$p")
  if grep -rqs 'gen_ai\.' "$p"/opentelemetry 2>/dev/null; then echo "IN-SCOPE  $n"; else echo "skip      $n"; fi
done
```

Expected: roughly 21 in-scope packages, matching the design's finding. Work only through the `IN-SCOPE` list.

- [ ] **Step 2: Add the tag to one package and verify Nx reads it**

Modify `packages/opentelemetry-instrumentation-anthropic/project.json`, changing:

```json
  "tags": [
    "instrumentation"
  ]
```

to:

```json
  "tags": [
    "instrumentation",
    "semconv:warn"
  ]
```

Run: `npx nx show projects --with-tag semconv:warn`

Expected: `opentelemetry-instrumentation-anthropic`

- [ ] **Step 3: Commit the tag mechanism**

```bash
git add packages/opentelemetry-instrumentation-anthropic/project.json
git commit -m "chore(anthropic): tag semconv rollout state"
```

- [ ] **Step 4: Extract the shared helper**

Per-package tests must not be verbatim copies of each other. Extract the shared logic into
the semconv package, mirroring the existing `_testing.py` precedent that packages already
import with a single line.

Create `packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/_testing_conformance.py`:

```python
"""Shared conformance-check helper for instrumentation package test suites.

Mirrors the existing `_testing.py` pattern: each package imports one function
and supplies its own fixtures, so the check logic lives in exactly one place.

    from opentelemetry.semconv_ai._testing_conformance import check_exported_spans

    def test_conforms(span_exporter):
        check_exported_spans(span_exporter, "anthropic.inference.client",
                             enforcing=False)
"""

from typing import Any, FrozenSet, List

from opentelemetry.semconv_ai._contract.extensions import EXTENSIONS
from opentelemetry.semconv_ai.conformance import Violation, assert_conforms

GEN_AI_OPERATION = "gen_ai.operation.name"


def gen_ai_spans(span_exporter: Any) -> List[Any]:
    """Finished spans that carry GenAI attributes, i.e. those the contract covers."""
    return [
        span
        for span in span_exporter.get_finished_spans()
        if (span.attributes or {}).get(GEN_AI_OPERATION) is not None
    ]


def check_exported_spans(
    span_exporter: Any,
    group_id: str,
    *,
    enforcing: bool,
    expected: FrozenSet[str] = frozenset(),
    extensions: FrozenSet[str] = EXTENSIONS,
    require_spans: bool = True,
) -> List[Violation]:
    """Check every exported GenAI span against one contract group.

    `expected` names the attributes this package promises to emit. Registry
    requirement levels alone are far too weak to rely on — only ~9% of contract
    attributes are `required`, and `gen_ai.response.finish_reasons` is merely
    `recommended` — so a package's own declared promise is what actually catches
    a dropped attribute. Pass it; do not rely on the contract alone.

    Returns the accumulated violations. In enforcing mode the first span with a
    blocking violation raises instead.
    """
    spans = gen_ai_spans(span_exporter)
    if require_spans:
        assert spans, (
            f"expected at least one span carrying {GEN_AI_OPERATION}; "
            "the instrumentation emitted none, so this test would pass vacuously"
        )

    violations: List[Violation] = []
    for span in spans:
        violations.extend(
            assert_conforms(
                span,
                group_id,
                enforcing=enforcing,
                expected=expected,
                extensions=extensions,
            )
        )
    return violations


__all__ = ["check_exported_spans", "gen_ai_spans"]
```

The `require_spans` assertion matters: without it, a package whose instrumentation emits
nothing would report a clean conformance pass. That is the failure mode this whole spec
exists to prevent.

- [ ] **Step 5: Rewrite the anthropic test to use the helper**

Replace the body of `packages/opentelemetry-instrumentation-anthropic/tests/test_conformance.py`
so it holds no duplicated check logic:

```python
"""Conformance of emitted spans against the OTel GenAI semantic conventions.

Warn-only until this package is tagged semconv:enforcing in project.json.
"""

import pytest
from opentelemetry.semconv_ai._testing_conformance import check_exported_spans

# Keep in sync with the semconv:* tag in project.json.
ENFORCING = False

CONTRACT_GROUP = "anthropic.inference.client"

# Attributes this package promises to emit on every inference span. The registry
# marks most of these merely `recommended`, so without this set the harness would
# not notice them going missing — which is exactly how #4362 (streaming drops
# finish_reasons) escaped detection. Adding a name here is a commitment.
EXPECTED = frozenset(
    {
        "gen_ai.operation.name",
        "gen_ai.provider.name",
        "gen_ai.request.model",
        "gen_ai.response.model",
        "gen_ai.response.finish_reasons",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
    }
)


@pytest.mark.vcr
def test_messages_span_conforms(instrument_legacy, anthropic_client, span_exporter):
    anthropic_client.messages.create(
        max_tokens=64,
        messages=[{"role": "user", "content": "Tell me a joke about OpenTelemetry"}],
        model="claude-3-5-sonnet-20240620",
    )
    check_exported_spans(
        span_exporter, CONTRACT_GROUP, enforcing=ENFORCING, expected=EXPECTED
    )


@pytest.mark.vcr
def test_streaming_span_conforms(instrument_legacy, anthropic_client, span_exporter):
    """Streaming is where #4362 reports dropped attributes."""
    stream = anthropic_client.messages.create(
        max_tokens=64,
        messages=[{"role": "user", "content": "Tell me a joke about OpenTelemetry"}],
        model="claude-3-5-sonnet-20240620",
        stream=True,
    )
    for _ in stream:
        pass
    check_exported_spans(
        span_exporter, CONTRACT_GROUP, enforcing=ENFORCING, expected=EXPECTED
    )
```

Every name in `EXPECTED` must be declared by the contract group (or listed in
`extensions`), or `check_attributes` raises `ValueError` — a typo cannot silently
pass. Verify each name exists in `SPANS["anthropic.inference.client"]` before adding it.

Run: `npx nx run opentelemetry-instrumentation-anthropic:test`

Expected: PASS.

- [ ] **Step 6: Commit the helper**

```bash
git add packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/_testing_conformance.py \
        packages/opentelemetry-instrumentation-anthropic/tests/test_conformance.py
git commit -m "refactor(semconv): extract shared conformance test helper"
```

- [ ] **Step 7: Replicate for each remaining in-scope package**

For each package in the in-scope list, one commit per package. Each test file is only the
imports, `ENFORCING`, `CONTRACT_GROUP`, and one test function per exercised call — all
check logic stays in the shared helper.

1. Create `tests/test_conformance.py` following the anthropic file above.
2. Set `CONTRACT_GROUP` per the mapping table.
3. Use the client fixture, model, and call from that package's existing tests, reusing an existing cassette. **Do not record new cassettes** — this task requires no API keys.
4. Add `"semconv:warn"` to `project.json` tags.
5. Run `npx nx run <pkg>:test` and confirm PASS.
6. Commit: `test(<pkg>): add warn-only semconv conformance checks`

If a package has no cassette exercising a `gen_ai` span, skip it and record it in the rollout doc under "no coverage" rather than inventing a fixture.

- [ ] **Step 8: Write the rollout tracking doc**

Create `docs/ai/semconv-rollout.md`:

```markdown
# Semconv conformance rollout

Every instrumentation package carries a tag in `project.json` recording its
state against the GenAI contract:

- `semconv:warn` — harness wired up, violations reported as warnings
- `semconv:enforcing` — violations fail CI

List the current frontier:

    npx nx show projects --with-tag semconv:warn
    npx nx show projects --with-tag semconv:enforcing

## Flipping a package to enforcing

1. Fix the violations the warnings report.
2. Set `ENFORCING = True` in that package's `tests/test_conformance.py`.
3. Change the tag from `semconv:warn` to `semconv:enforcing`.
4. Confirm `npx nx run <pkg>:test` passes.

Never flip a package that still warns.

## Out of scope

Vector-store instrumentations (chromadb, pinecone, qdrant, weaviate, milvus,
marqo, lancedb) emit `db.*` attributes. The GenAI contract does not cover them.

## No coverage

Packages with no cassette exercising a `gen_ai` span are listed here and have no
conformance test yet.
```

- [ ] **Step 9: Verify the whole workspace still passes**

Run: `npx nx run-many -t test --exclude=sample-app --parallel=2`

Expected: PASS across all packages. Warnings are expected and do not fail the build.

- [ ] **Step 10: Commit the rollout doc**

```bash
git add docs/ai/semconv-rollout.md
git commit -m "docs: track semconv conformance rollout state"
```

---

### Task 8: CI integration

**Files:**
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `make -C .semconv check` (Task 1), the conformance tests (Tasks 6–7).
- Produces: a `semconv-contract` CI job.

- [ ] **Step 1: Add the contract-freshness job**

Add to `.github/workflows/ci.yml`, as a new top-level entry under `jobs:`:

```yaml
  semconv-contract:
    name: Semconv Contract
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          ref: ${{ github.event.pull_request.head.sha }}

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: 3.11

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: "latest"

      # Regenerates the committed contract module from the vendored registry and
      # fails if it differs. This is what stops _contract/generated.py drifting
      # from .semconv/registry/ without anyone noticing.
      - name: Verify committed contract is current
        run: make -C .semconv check
```

Note: `make check` depends on `resolve`, which runs the `otel/weaver` container. GitHub-hosted Ubuntu runners have Docker preinstalled, so no setup step is needed.

- [ ] **Step 2: Verify the job locally**

Run: `make -C .semconv check`

Expected: exit 0, no output from `git diff`.

- [ ] **Step 3: Verify it fails when the artifact is stale**

Prove the check is not vacuous:

The tamper must be COMMITTED. `make check` runs `generate` first, which overwrites an
uncommitted edit before the diff ever runs — so an uncommitted tamper always passes and
proves nothing. The real threat is a hand-edit that got merged, and CI always starts from
a clean checkout of committed state.

```bash
cd /Users/gal.kleinman/dev/openllmetry
# require a clean tree before tampering — `git reset --hard` below would
# otherwise silently discard any uncommitted work
git diff --quiet && git diff --cached --quiet || { echo "working tree not clean; aborting"; exit 1; }
GEN=packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/_contract/generated.py
# weaken the contract the way a bad hand-edit would
sed -i '' "0,/Level.REQUIRED/s//Level.OPT_IN/" "$GEN"
git add "$GEN" && git commit -q -m "TEMP tamper test"
make -C .semconv check; echo "EXIT=$?"
git reset --hard HEAD~1
```

Expected: non-zero exit with a diff showing the weakened level being reverted. If it exits
0, the check does not work — investigate before proceeding. The tree must be clean before
this runs, since the tamper is reverted with `git reset --hard`, which would otherwise
destroy uncommitted work.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: verify semconv contract artifact is current"
```

- [ ] **Step 5: Open the PR**

```bash
git push -u origin gk/ai-native-maintainance
gh pr create --title "feat(semconv): executable GenAI semantic-convention contract" --body "$(cat <<'EOF'
Implements spec 1 of the AI-native maintenance design.

Vendors the OTel GenAI semantic conventions at a pinned SHA, generates a
committed Python contract module from them via weaver, and adds a conformance
harness wired into each instrumentation package in warn-only mode.

## What this does not do

Fixes no violations. The warnings this surfaces are the input to spec 3.

## Notes for review

- `_contract/generated.py` is generated; review `_contract/_generator.py` instead.
- `_contract/extensions.py` documents five `gen_ai.*` attributes OpenLLMetry
  emits that upstream does not define. Each needs a decision: keep as a declared
  extension, or migrate.
- Every package is `semconv:warn`. Nothing fails CI on a violation yet.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review

**Spec coverage.** Every spec-1 requirement maps to a task: vendoring and the pinned SHA (T1), weaver codegen (T1, T3), the committed artifact with CI diff-check (T3, T8), the conformance harness (T4), warn-only rollout (T6, T7), `project.json` enforcement tags (T7), and JSON-Schema payload validation — **deferred**, see below.

**Deliberate deferral.** The design mentions validating message payloads against upstream's `gen-ai-*-messages.json` JSON Schemas. That is not in this plan. Those schemas validate the *contents* of `gen_ai.input.messages` / `gen_ai.output.messages`, which are `opt_in` attributes that most packages do not emit at the pinned ref. Adding schema validation before any package emits the attributes would be untestable. It belongs in a follow-up once the attribute-level harness is enforcing somewhere. Flagging rather than silently dropping it.

**Placeholder scan.** No TBDs. Every code step carries complete code. Task 6 Step 2 and Task 7 Step 4 require substituting real fixture names from each package's existing tests — these are explicitly called out as adaptation points with instructions on how to find the right names, not hidden gaps.

**Type consistency.** `Level`, `AttributeSpec`, `SpanSpec`, `parse_requirement_level`, `enum_members_of` defined in T2 and used consistently in T3–T5. `check_attributes` and `assert_conforms` defined in T4 with the same signatures used in T6–T7. `EXTENSIONS` defined in T5, consumed in T6–T7. `SPANS` and `CONTRACT_REF` produced by T3's generator, consumed in T4 and T5. Violation `kind` values are fixed to three strings in T4 and asserted against those exact strings in T4's tests.

**Verification steps that prove non-vacuousness.** Three tasks include a deliberate "break it and confirm the check fails" step — T6 Step 4 (harness catches real violations), T8 Step 3 (diff-check catches staleness), T3 Step 7 (`make check` clean on a fresh tree). These exist because a conformance suite that silently reports zero violations is worse than none: it manufactures false confidence.
