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
