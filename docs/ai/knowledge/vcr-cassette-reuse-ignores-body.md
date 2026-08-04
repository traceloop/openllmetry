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
