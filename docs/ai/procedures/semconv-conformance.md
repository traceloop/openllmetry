# Working with the semconv contract

The OpenTelemetry GenAI semantic conventions are vendored and executable here. The
contract is the authority on attribute names — do not invent `gen_ai.*` names.

## Layout

    .semconv/versions.env      pinned upstream SHA, weaver version
    .semconv/registry/         vendored upstream registry (committed)
    packages/opentelemetry-semantic-conventions-ai/
      .../_contract/generated.py    committed, generated — never hand-edit
      .../_contract/extensions.py   non-standard gen_ai.* we knowingly emit
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
