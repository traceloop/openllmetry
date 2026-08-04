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
