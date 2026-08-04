# Vendored GenAI semantic conventions

This directory vendors https://github.com/open-telemetry/semantic-conventions-genai
at the SHA pinned in `versions.env`. It is the contract every instrumentation
package with conformance coverage is tested against. See
`docs/ai/semconv-rollout.md` for which packages currently have that coverage.

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
