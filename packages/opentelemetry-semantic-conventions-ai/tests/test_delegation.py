"""Tests for the opt-in delegation / authority span attributes."""

import asyncio
import base64
import json

import pytest
from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.semconv_ai.delegation import (
    AUTHZ_DECISION,
    AUTHZ_REASON,
    DELEGATION_ACTOR,
    DELEGATION_CHAIN_HASH,
    DELEGATION_CHAIN_ID,
    DELEGATION_CONTEXT_KEY,
    DELEGATION_CREDENTIAL_REF,
    DELEGATION_DEPTH,
    DELEGATION_PARENT_SCOPE_HASH,
    DELEGATION_ROOT_PRINCIPAL,
    DELEGATION_SCOPE_HASH,
    attach_delegation,
    build_delegation_attributes,
    delegation,
    detach_delegation,
    get_delegation_attributes,
    hash_chain,
    hash_scope,
    record_authorization,
    validate_delegation_attributes,
)

SCOPE_HASH = hash_scope(["crm:read", "email:send"])
CHAIN_HASH = hash_chain(["user:42", "agent:triage", "agent:billing"])


def _b64url(obj):
    """Base64url-encode a JSON object without padding, as in a JWT segment."""
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()


# Built at runtime so no JWT-shaped literal appears in source (secret scanners flag them).
JWT_LIKE = ".".join([_b64url({"alg": "HS256"}), _b64url({"sub": "1234567890"}), "s" * 43])


@pytest.fixture
def tracer():
    """Return a tracer whose finished spans are collected in memory."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)
    tracer.exporter = exporter
    return tracer


def test_constants_match_span_attributes():
    """Module constants mirror SpanAttributes and share the delegation prefix."""
    assert SpanAttributes.TRACELOOP_DELEGATION_ROOT_PRINCIPAL == DELEGATION_ROOT_PRINCIPAL
    assert SpanAttributes.TRACELOOP_DELEGATION_CHAIN_ID == DELEGATION_CHAIN_ID
    assert SpanAttributes.TRACELOOP_DELEGATION_DEPTH == DELEGATION_DEPTH
    assert SpanAttributes.TRACELOOP_DELEGATION_SCOPE_HASH == DELEGATION_SCOPE_HASH
    assert SpanAttributes.TRACELOOP_DELEGATION_ACTOR == DELEGATION_ACTOR
    assert SpanAttributes.TRACELOOP_DELEGATION_CHAIN_HASH == DELEGATION_CHAIN_HASH
    assert SpanAttributes.TRACELOOP_DELEGATION_PARENT_SCOPE_HASH == DELEGATION_PARENT_SCOPE_HASH
    assert SpanAttributes.TRACELOOP_DELEGATION_CREDENTIAL_REF == DELEGATION_CREDENTIAL_REF
    assert SpanAttributes.TRACELOOP_AUTHZ_DECISION == AUTHZ_DECISION
    assert SpanAttributes.TRACELOOP_AUTHZ_REASON == AUTHZ_REASON
    for name in (
        DELEGATION_ROOT_PRINCIPAL,
        DELEGATION_CHAIN_ID,
        DELEGATION_DEPTH,
        DELEGATION_SCOPE_HASH,
        DELEGATION_ACTOR,
        DELEGATION_CHAIN_HASH,
        DELEGATION_PARENT_SCOPE_HASH,
        DELEGATION_CREDENTIAL_REF,
    ):
        assert name.startswith(DELEGATION_CONTEXT_KEY + ".")


def test_nothing_recorded_without_opt_in():
    """Without a delegation block nothing is recorded."""
    assert get_delegation_attributes() == {}


def test_delegation_block_sets_and_restores_context():
    """The block sets every supplied attribute and restores the context on exit."""
    with delegation("user:42", chain_id="dlg_7f3a", scope_hash=SCOPE_HASH, depth=1):
        assert get_delegation_attributes() == {
            DELEGATION_ROOT_PRINCIPAL: "user:42",
            DELEGATION_CHAIN_ID: "dlg_7f3a",
            DELEGATION_SCOPE_HASH: SCOPE_HASH,
            DELEGATION_DEPTH: 1,
        }
    assert get_delegation_attributes() == {}


def test_only_root_principal_is_required():
    """Only the root principal is mandatory."""
    assert dict(build_delegation_attributes("svc:billing")) == {DELEGATION_ROOT_PRINCIPAL: "svc:billing"}


def test_nested_blocks_override_and_restore():
    """Inner blocks override outer values until they exit."""
    with delegation("user:1"):
        with delegation("user:1", chain_id="inner", depth=2):
            assert get_delegation_attributes()[DELEGATION_CHAIN_ID] == "inner"
        assert get_delegation_attributes() == {DELEGATION_ROOT_PRINCIPAL: "user:1"}


def test_attach_and_detach_token():
    """attach/detach work for code that cannot use a with-block."""
    token = attach_delegation("user:9")
    try:
        assert get_delegation_attributes()[DELEGATION_ROOT_PRINCIPAL] == "user:9"
    finally:
        detach_delegation(token)
    assert get_delegation_attributes() == {}


def test_context_isolated_between_asyncio_tasks():
    """Delegation context does not leak between asyncio tasks."""
    async def read():
        """Read the delegation attributes from inside a task."""
        return get_delegation_attributes()

    async def main():
        """Run one task inside and one outside a delegation block."""
        with delegation("user:task"):
            inside = await asyncio.create_task(read())
        outside = await asyncio.create_task(read())
        return inside, outside

    inside, outside = asyncio.run(main())
    assert inside == {DELEGATION_ROOT_PRINCIPAL: "user:task"}
    assert outside == {}


@pytest.mark.parametrize(
    "value",
    [
        "Bearer abc.def.ghi",
        JWT_LIKE,
        "user 42",
        "",
        "x" * 257,
        None,
        42,
    ],
)
def test_rejects_credentials_and_bad_identifiers(value):
    """Tokens, whitespace, empty, oversized and non-string identifiers are rejected."""
    with pytest.raises(ValueError):
        build_delegation_attributes(value)
    if value is not None:  # chain_id is optional
        with pytest.raises(ValueError):
            build_delegation_attributes("user:1", chain_id=value)


@pytest.mark.parametrize(
    "value", ["crm:read", "sha256:xyz", "md5:" + "0" * 32, "deadbeef" * 8, "sha256:" + "ab" * 32 + "\n"]
)
def test_scope_hash_must_be_a_digest(value):
    """Only digests are accepted for the scope hash."""
    with pytest.raises(ValueError):
        build_delegation_attributes("user:1", scope_hash=value)


@pytest.mark.parametrize("value", [-1, 1.5, True, "2"])
def test_depth_must_be_non_negative_int(value):
    """Depth must be a non-negative int (bools excluded)."""
    with pytest.raises(ValueError):
        build_delegation_attributes("user:1", depth=value)


def test_hash_scope_is_stable_and_order_independent():
    """Scope digests ignore ordering and never contain the scope itself."""
    assert hash_scope(["b", "a"]) == hash_scope(("a", "b")) == hash_scope({"a", "b"})
    assert hash_scope({"x": 1, "y": [1, 2]}) == hash_scope({"y": [1, 2], "x": 1})
    assert hash_scope("crm:read").startswith("sha256:")
    assert hash_scope("crm:read", algorithm="sha512").startswith("sha512:")
    assert "crm:read" not in hash_scope("crm:read")
    with pytest.raises(ValueError):
        hash_scope("crm:read", algorithm="md5")


def test_get_ignores_foreign_or_malformed_values():
    """Foreign keys, wrong types and non-mapping values are filtered out."""
    token = context_api.attach(
        context_api.set_value(
            DELEGATION_CONTEXT_KEY,
            {DELEGATION_ROOT_PRINCIPAL: "user:1", "other.key": "x", DELEGATION_DEPTH: True},
        )
    )
    try:
        assert get_delegation_attributes() == {DELEGATION_ROOT_PRINCIPAL: "user:1"}
    finally:
        context_api.detach(token)

    token = context_api.attach(context_api.set_value(DELEGATION_CONTEXT_KEY, "not-a-mapping"))
    try:
        assert get_delegation_attributes() == {}
    finally:
        context_api.detach(token)


def test_get_validates_values_stored_outside_the_helper():
    """A mapping attached directly is held to the builder's rules."""
    value = {
        DELEGATION_ROOT_PRINCIPAL: "user:1",
        DELEGATION_CHAIN_ID: "Bearer secret",
        DELEGATION_ACTOR: JWT_LIKE,
        DELEGATION_SCOPE_HASH: "crm:read email:send",
        DELEGATION_DEPTH: -1,
        "traceloop.delegation.token": "not-an-approved-name",
    }
    token = context_api.attach(context_api.set_value(DELEGATION_CONTEXT_KEY, value))
    try:
        assert get_delegation_attributes() == {DELEGATION_ROOT_PRINCIPAL: "user:1"}
    finally:
        context_api.detach(token)


@pytest.mark.parametrize("root_principal", ["Bearer secret", "", 42, None])
def test_get_returns_nothing_without_a_valid_root_principal(root_principal):
    """Without a valid root principal the remaining values are not recorded."""
    value = {DELEGATION_CHAIN_ID: "dlg_1", DELEGATION_DEPTH: 1}
    if root_principal is not None:
        value[DELEGATION_ROOT_PRINCIPAL] = root_principal
    token = context_api.attach(context_api.set_value(DELEGATION_CONTEXT_KEY, value))
    try:
        assert get_delegation_attributes() == {}
    finally:
        context_api.detach(token)


def test_validate_delegation_attributes_accepts_builder_output():
    """Everything the builder produces passes validation unchanged."""
    built = build_delegation_attributes(
        "user:42", "dlg_1", SCOPE_HASH, 2, "agent:a", CHAIN_HASH,
        parent_scope_hash=CHAIN_HASH, credential_ref="txn:1",
    )
    assert validate_delegation_attributes(built) == dict(built)
    assert validate_delegation_attributes("not-a-mapping") == {}
    assert validate_delegation_attributes({"traceloop.delegation.root_principal": JWT_LIKE}) == {}


def test_actor_and_chain_hash_are_recorded():
    """The current actor and verified-chain digest are recorded when supplied."""
    with delegation("user:42", actor="agent:triage", chain_hash=CHAIN_HASH, depth=2):
        assert get_delegation_attributes() == {
            DELEGATION_ROOT_PRINCIPAL: "user:42",
            DELEGATION_ACTOR: "agent:triage",
            DELEGATION_CHAIN_HASH: CHAIN_HASH,
            DELEGATION_DEPTH: 2,
        }


def test_nested_block_updates_actor_on_handoff():
    """An inner block switches the actor for the handed-off agent and restores it on exit."""
    with delegation("user:42", actor="agent:triage"):
        with delegation("user:42", actor="agent:billing", depth=1):
            assert get_delegation_attributes()[DELEGATION_ACTOR] == "agent:billing"
        assert get_delegation_attributes()[DELEGATION_ACTOR] == "agent:triage"


@pytest.mark.parametrize("value", ["Bearer abc.def.ghi", JWT_LIKE, "agent triage", ""])
def test_actor_rejects_credentials_and_bad_identifiers(value):
    """The actor is validated like every other identifier."""
    with pytest.raises(ValueError):
        build_delegation_attributes("user:1", actor=value)


@pytest.mark.parametrize("value", ["chain-1", "sha256:xyz", "md5:" + "0" * 32, "sha256:" + "ab" * 32 + "\n"])
def test_chain_hash_must_be_a_digest(value):
    """Only digests are accepted for the chain hash, never the chain itself."""
    with pytest.raises(ValueError, match="hash_chain"):
        build_delegation_attributes("user:1", chain_hash=value)


def test_hash_chain_is_stable_and_order_sensitive():
    """Equal chains hash equally; reordering hops changes the digest."""
    hops = [{"sub": "user:42", "grant": "g1"}, {"aud": "agent:triage", "grant": "g2"}]
    reordered_keys = [{"grant": "g1", "sub": "user:42"}, {"grant": "g2", "aud": "agent:triage"}]
    assert hash_chain(hops) == hash_chain(reordered_keys) == hash_chain(tuple(hops))
    assert hash_chain(["user:1", "agent:a"]) != hash_chain(["agent:a", "user:1"])
    assert hash_chain(["user:1"], algorithm="sha512").startswith("sha512:")
    assert "user:42" not in hash_chain(hops)
    for bad in ("user:1", b"user:1", {"user:1"}, None):
        with pytest.raises(ValueError):
            hash_chain(bad)
    with pytest.raises(ValueError):
        hash_chain(["user:1"], algorithm="md5")


def test_record_authorization_on_current_span(tracer):
    """Allow and deny decisions land on the span active when the guard runs."""
    with tracer.start_as_current_span("tool.allowed"):
        record_authorization("allow")
    with tracer.start_as_current_span("tool.denied"):
        record_authorization("deny", reason="scope_exceeded")
    allowed, denied = tracer.exporter.get_finished_spans()
    assert allowed.attributes[AUTHZ_DECISION] == "allow"
    assert AUTHZ_REASON not in allowed.attributes
    assert denied.attributes[AUTHZ_DECISION] == "deny"
    assert denied.attributes[AUTHZ_REASON] == "scope_exceeded"


def test_record_authorization_on_explicit_span(tracer):
    """An explicit span can be annotated even when it is not the current one."""
    span = tracer.start_span("tool.call")
    record_authorization("deny", span=span)
    span.end()
    (finished,) = tracer.exporter.get_finished_spans()
    assert finished.attributes == {AUTHZ_DECISION: "deny"}


def test_record_authorization_without_recording_span_is_a_no_op():
    """With no active recording span nothing happens and nothing raises."""
    assert not trace.get_current_span().is_recording()
    record_authorization("allow")
    record_authorization("deny", reason="policy:missing")


@pytest.mark.parametrize(
    "decision, reason",
    [
        ("ALLOW", None),
        ("maybe", None),
        (True, None),
        ("allow", "scope_exceeded"),
        ("deny", "User asked for something odd"),
        ("deny", "x" * 65),
        ("deny", "scope_exceeded\n"),
        ("deny", JWT_LIKE),
        ("deny", 403),
    ],
)
def test_record_authorization_rejects_bad_input(tracer, decision, reason):
    """Unknown decisions, free-text reasons and reasons on 'allow' are rejected."""
    with tracer.start_as_current_span("tool.call"):
        with pytest.raises(ValueError):
            record_authorization(decision, reason=reason)


NARROW_SCOPE_HASH = hash_scope(["crm:read"])


def test_nested_scope_records_parent_scope_hash():
    """Each hop records its own scope and the scope of the hop it came from."""
    with delegation("user:42", scope_hash=SCOPE_HASH):
        assert DELEGATION_PARENT_SCOPE_HASH not in get_delegation_attributes()
        with delegation("user:42", actor="agent:billing", depth=1, scope_hash=NARROW_SCOPE_HASH):
            attrs = get_delegation_attributes()
            assert attrs[DELEGATION_SCOPE_HASH] == NARROW_SCOPE_HASH
            assert attrs[DELEGATION_PARENT_SCOPE_HASH] == SCOPE_HASH
            third = hash_scope([])
            with delegation("user:42", depth=2, scope_hash=third):
                attrs = get_delegation_attributes()
                assert attrs[DELEGATION_SCOPE_HASH] == third
                assert attrs[DELEGATION_PARENT_SCOPE_HASH] == NARROW_SCOPE_HASH
            assert get_delegation_attributes()[DELEGATION_PARENT_SCOPE_HASH] == SCOPE_HASH
        assert DELEGATION_PARENT_SCOPE_HASH not in get_delegation_attributes()


def test_parent_scope_hash_needs_own_scope_hash_to_be_inferred():
    """No parent is inferred for a hop that declares no scope of its own."""
    with delegation("user:42", scope_hash=SCOPE_HASH):
        with delegation("user:42", actor="agent:billing"):
            assert DELEGATION_PARENT_SCOPE_HASH not in get_delegation_attributes()


def test_explicit_parent_scope_hash_wins():
    """An explicit parent (e.g. from an incoming grant) overrides the enclosing block."""
    with delegation("user:42", scope_hash=SCOPE_HASH):
        with delegation("user:42", scope_hash=NARROW_SCOPE_HASH, parent_scope_hash=CHAIN_HASH):
            assert get_delegation_attributes()[DELEGATION_PARENT_SCOPE_HASH] == CHAIN_HASH
    token = attach_delegation("user:1", parent_scope_hash=SCOPE_HASH)
    try:
        assert get_delegation_attributes()[DELEGATION_PARENT_SCOPE_HASH] == SCOPE_HASH
    finally:
        detach_delegation(token)


@pytest.mark.parametrize("value", ["crm:read", "sha256:xyz", "md5:" + "0" * 32, "sha256:" + "ab" * 32 + "\n"])
def test_parent_scope_hash_must_be_a_digest(value):
    """Only digests are accepted for the parent scope hash."""
    with pytest.raises(ValueError):
        build_delegation_attributes("user:1", parent_scope_hash=value)


@pytest.mark.parametrize(
    "value",
    [
        "txn:9c1e",
        "vaid:agent-billing-01",
        "spiffe://example.org/ns/prod/sa/billing",
        "urn:ietf:params:oauth:token-type:access_token#exchange-42",
    ],
)
def test_credential_ref_is_recorded(value):
    """Opaque credential references are recorded as-is."""
    with delegation("user:42", credential_ref=value):
        assert get_delegation_attributes()[DELEGATION_CREDENTIAL_REF] == value


@pytest.mark.parametrize(
    "value",
    ["Bearer abc.def.ghi", "basic placeholder", JWT_LIKE, "ref with space", "", "x" * 257, 7],
)
def test_credential_ref_rejects_credentials(value):
    """A credential_ref must be a reference, never a token."""
    with pytest.raises(ValueError):
        build_delegation_attributes("user:1", credential_ref=value)
    with pytest.raises(ValueError):
        with delegation("user:1", credential_ref=value):
            pass  # pragma: no cover
    assert get_delegation_attributes() == {}


def test_credential_ref_is_not_inherited_by_nested_hops():
    """Each hop names the credential that carried its own authority."""
    with delegation("user:42", credential_ref="txn:outer"):
        with delegation("user:42", actor="agent:billing"):
            assert DELEGATION_CREDENTIAL_REF not in get_delegation_attributes()
        assert get_delegation_attributes()[DELEGATION_CREDENTIAL_REF] == "txn:outer"


def test_new_arguments_are_keyword_only():
    """parent_scope_hash and credential_ref can't be passed positionally."""
    with pytest.raises(TypeError):
        build_delegation_attributes("user:1", None, None, None, None, None, SCOPE_HASH)
