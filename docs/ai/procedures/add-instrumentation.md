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

`project.json` targets are copied verbatim across all packages — do not change the
target definitions themselves. Only the identity fields are package-specific: set
`name`, `sourceRoot`, and the `cwd` values to match the new package, and add
`"instrumentation"` to `tags`.

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
