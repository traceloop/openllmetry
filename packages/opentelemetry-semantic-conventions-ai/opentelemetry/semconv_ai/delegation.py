"""Delegation / authority context for agent spans.

Multi-agent systems act on someone's behalf: a user (or service) delegates to an
agent, which may hand off to further agents. Auditors want to know, for any agent
or handoff span, *on whose behalf* it ran and *under which grant*. This module
lets the application declare that once, and lets instrumentations stamp it onto
the agent and handoff spans they create.

Design rules:

* **Opt-in.** Nothing is recorded unless the application calls ``delegation()``
  (or ``attach_delegation()``). Instrumentations only read what was supplied.
* **References and hashes, never credentials.** Only identifiers, a chain
  reference and a hash of the granted scope are accepted. Values that look like
  bearer tokens / JWTs are rejected, and the scope itself is never recorded:
  use :func:`hash_scope` to turn it into a stable digest.
* **Vendor namespace.** Attributes live under ``traceloop.delegation.*`` until
  the OpenTelemetry GenAI semantic conventions define an equivalent, so they
  can't collide with a future upstream ``gen_ai.*`` definition.
* **In-process only.** The value lives in the OpenTelemetry ``Context`` (not
  Baggage), so it is never propagated over the wire in request headers.

Example::

    from opentelemetry.semconv_ai.delegation import delegation, hash_scope

    with delegation(
        root_principal="user:42",
        chain_id="dlg_7f3a",
        scope_hash=hash_scope(["crm:read", "email:send"]),
    ):
        Runner.run_sync(triage_agent, "...")
"""

import hashlib
import json
import re
from contextlib import contextmanager
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Optional

from opentelemetry import context as context_api

# Context key under which the (already validated) span attributes are stored.
# A plain string so instrumentations can read it without importing this module,
# the same way SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY is shared.
DELEGATION_CONTEXT_KEY = "traceloop.delegation"

DELEGATION_ROOT_PRINCIPAL = "traceloop.delegation.root_principal"
DELEGATION_CHAIN_ID = "traceloop.delegation.chain_id"
DELEGATION_DEPTH = "traceloop.delegation.depth"
DELEGATION_SCOPE_HASH = "traceloop.delegation.scope_hash"

_MAX_ID_LENGTH = 256
_SCOPE_HASH_RE = re.compile(r"^(sha256|sha384|sha512|blake2b):[0-9a-f]{32,128}$")
# Three base64url segments separated by dots: the shape of a JWT / JWS.
_JWT_RE = re.compile(r"^[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]*$")
_AUTH_SCHEME_RE = re.compile(r"^(bearer|basic|token|apikey)\s", re.IGNORECASE)


def _validate_identifier(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if len(value) > _MAX_ID_LENGTH:
        raise ValueError(f"{name} must be at most {_MAX_ID_LENGTH} characters")
    if any(ch.isspace() for ch in value) or _AUTH_SCHEME_RE.match(value) or _JWT_RE.match(value):
        raise ValueError(
            f"{name} looks like a credential; pass an identifier or reference, "
            "never a token"
        )
    return value


def hash_scope(scope: Any, algorithm: str = "sha256") -> str:
    """Return a stable ``"<algorithm>:<hexdigest>"`` digest of a granted scope.

    ``scope`` may be a string, a list/tuple/set of scope strings (order does not
    matter), or any JSON-serialisable mapping. The scope itself is not recorded.
    """
    if algorithm not in ("sha256", "sha384", "sha512", "blake2b"):
        raise ValueError(f"unsupported hash algorithm: {algorithm}")
    if isinstance(scope, (list, tuple, set, frozenset)):
        canonical = json.dumps(sorted(str(s) for s in scope), separators=(",", ":"))
    elif isinstance(scope, str):
        canonical = scope
    else:
        canonical = json.dumps(scope, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.new(algorithm, canonical.encode("utf-8")).hexdigest()
    return f"{algorithm}:{digest}"


def build_delegation_attributes(
    root_principal: str,
    chain_id: Optional[str] = None,
    scope_hash: Optional[str] = None,
    depth: Optional[int] = None,
) -> Mapping[str, Any]:
    """Validate the inputs and return the span attributes to record."""
    attributes = {DELEGATION_ROOT_PRINCIPAL: _validate_identifier("root_principal", root_principal)}
    if chain_id is not None:
        attributes[DELEGATION_CHAIN_ID] = _validate_identifier("chain_id", chain_id)
    if scope_hash is not None:
        if not isinstance(scope_hash, str) or not _SCOPE_HASH_RE.match(scope_hash):
            raise ValueError(
                "scope_hash must look like '<sha256|sha384|sha512|blake2b>:<hex digest>'; "
                "use hash_scope() to build it"
            )
        attributes[DELEGATION_SCOPE_HASH] = scope_hash
    if depth is not None:
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
            raise ValueError("depth must be a non-negative integer")
        attributes[DELEGATION_DEPTH] = depth
    return MappingProxyType(attributes)


def attach_delegation(
    root_principal: str,
    chain_id: Optional[str] = None,
    scope_hash: Optional[str] = None,
    depth: Optional[int] = None,
) -> object:
    """Attach delegation info to the current context; returns a detach token."""
    attributes = build_delegation_attributes(root_principal, chain_id, scope_hash, depth)
    return context_api.attach(context_api.set_value(DELEGATION_CONTEXT_KEY, attributes))


def detach_delegation(token: object) -> None:
    context_api.detach(token)


@contextmanager
def delegation(
    root_principal: str,
    chain_id: Optional[str] = None,
    scope_hash: Optional[str] = None,
    depth: Optional[int] = None,
) -> Iterator[Mapping[str, Any]]:
    """Record delegation info on agent / handoff spans created inside the block."""
    token = attach_delegation(root_principal, chain_id, scope_hash, depth)
    try:
        yield context_api.get_value(DELEGATION_CONTEXT_KEY)
    finally:
        detach_delegation(token)


def get_delegation_attributes(ctx: Optional[context_api.Context] = None) -> Mapping[str, Any]:
    """Return the delegation span attributes active in ``ctx`` (or the current context)."""
    value = context_api.get_value(DELEGATION_CONTEXT_KEY, ctx)
    if not isinstance(value, Mapping):
        return {}
    return {
        k: v
        for k, v in value.items()
        if isinstance(k, str) and k.startswith(DELEGATION_CONTEXT_KEY + ".")
        and isinstance(v, (str, int)) and not isinstance(v, bool)
    }
