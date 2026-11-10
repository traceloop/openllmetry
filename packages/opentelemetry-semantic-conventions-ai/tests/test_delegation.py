import asyncio

import pytest
from opentelemetry import context as context_api

from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.semconv_ai.delegation import (
    DELEGATION_CHAIN_ID,
    DELEGATION_CONTEXT_KEY,
    DELEGATION_DEPTH,
    DELEGATION_ROOT_PRINCIPAL,
    DELEGATION_SCOPE_HASH,
    attach_delegation,
    build_delegation_attributes,
    delegation,
    detach_delegation,
    get_delegation_attributes,
    hash_scope,
)

SCOPE_HASH = hash_scope(["crm:read", "email:send"])


def test_constants_match_span_attributes():
    assert SpanAttributes.TRACELOOP_DELEGATION_ROOT_PRINCIPAL == DELEGATION_ROOT_PRINCIPAL
    assert SpanAttributes.TRACELOOP_DELEGATION_CHAIN_ID == DELEGATION_CHAIN_ID
    assert SpanAttributes.TRACELOOP_DELEGATION_DEPTH == DELEGATION_DEPTH
    assert SpanAttributes.TRACELOOP_DELEGATION_SCOPE_HASH == DELEGATION_SCOPE_HASH
    for name in (DELEGATION_ROOT_PRINCIPAL, DELEGATION_CHAIN_ID, DELEGATION_DEPTH, DELEGATION_SCOPE_HASH):
        assert name.startswith(DELEGATION_CONTEXT_KEY + ".")


def test_nothing_recorded_without_opt_in():
    assert get_delegation_attributes() == {}


def test_delegation_block_sets_and_restores_context():
    with delegation("user:42", chain_id="dlg_7f3a", scope_hash=SCOPE_HASH, depth=1):
        assert get_delegation_attributes() == {
            DELEGATION_ROOT_PRINCIPAL: "user:42",
            DELEGATION_CHAIN_ID: "dlg_7f3a",
            DELEGATION_SCOPE_HASH: SCOPE_HASH,
            DELEGATION_DEPTH: 1,
        }
    assert get_delegation_attributes() == {}


def test_only_root_principal_is_required():
    assert dict(build_delegation_attributes("svc:billing")) == {DELEGATION_ROOT_PRINCIPAL: "svc:billing"}


def test_nested_blocks_override_and_restore():
    with delegation("user:1"):
        with delegation("user:1", chain_id="inner", depth=2):
            assert get_delegation_attributes()[DELEGATION_CHAIN_ID] == "inner"
        assert get_delegation_attributes() == {DELEGATION_ROOT_PRINCIPAL: "user:1"}


def test_attach_and_detach_token():
    token = attach_delegation("user:9")
    try:
        assert get_delegation_attributes()[DELEGATION_ROOT_PRINCIPAL] == "user:9"
    finally:
        detach_delegation(token)
    assert get_delegation_attributes() == {}


def test_context_isolated_between_asyncio_tasks():
    async def read():
        return get_delegation_attributes()

    async def main():
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
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
        "user 42",
        "",
        "x" * 257,
        None,
        42,
    ],
)
def test_rejects_credentials_and_bad_identifiers(value):
    with pytest.raises(ValueError):
        build_delegation_attributes(value)
    if value is not None:  # chain_id is optional
        with pytest.raises(ValueError):
            build_delegation_attributes("user:1", chain_id=value)


@pytest.mark.parametrize("value", ["crm:read", "sha256:xyz", "md5:" + "0" * 32, "deadbeef" * 8])
def test_scope_hash_must_be_a_digest(value):
    with pytest.raises(ValueError):
        build_delegation_attributes("user:1", scope_hash=value)


@pytest.mark.parametrize("value", [-1, 1.5, True, "2"])
def test_depth_must_be_non_negative_int(value):
    with pytest.raises(ValueError):
        build_delegation_attributes("user:1", depth=value)


def test_hash_scope_is_stable_and_order_independent():
    assert hash_scope(["b", "a"]) == hash_scope(("a", "b")) == hash_scope({"a", "b"})
    assert hash_scope({"x": 1, "y": [1, 2]}) == hash_scope({"y": [1, 2], "x": 1})
    assert hash_scope("crm:read").startswith("sha256:")
    assert hash_scope("crm:read", algorithm="sha512").startswith("sha512:")
    assert "crm:read" not in hash_scope("crm:read")
    with pytest.raises(ValueError):
        hash_scope("crm:read", algorithm="md5")


def test_get_ignores_foreign_or_malformed_values():
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
