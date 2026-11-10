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
  reference and hashes of the granted scope / verified chain are accepted.
  Values that look like bearer tokens / JWTs are rejected, and neither the
  scope nor the chain is ever recorded: use :func:`hash_scope` and
  :func:`hash_chain` to turn them into stable digests.
* **Per-hop scope.** When a nested block declares its own ``scope_hash``, the
  enclosing block's ``scope_hash`` is recorded next to it as
  ``parent_scope_hash``, so each hop shows the scope it received and the scope
  it came from. A verifier holding the scopes can then check that authority
  only narrowed along the chain.
* **Credential references, not credentials.** ``credential_ref`` can point at
  whatever carried the authority (an OAuth token exchange result id, a VAID, a
  SPIFFE ID) so a verifier can join the trace to the evidence. It must be an
  identifier: token-shaped values are rejected like everywhere else.
* **Authorization decisions are explicit.** :func:`record_authorization` lets
  the application's tool guard record ``allow`` / ``deny`` (plus a short
  machine-readable reason for a denial) on the current span *before* the tool
  runs. The library never makes or infers the decision itself.
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
        actor="agent:triage",
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
from opentelemetry import trace

# Context key under which the (already validated) span attributes are stored.
# A plain string so instrumentations can read it without importing this module,
# the same way SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY is shared.
DELEGATION_CONTEXT_KEY = "traceloop.delegation"

DELEGATION_ROOT_PRINCIPAL = "traceloop.delegation.root_principal"
DELEGATION_CHAIN_ID = "traceloop.delegation.chain_id"
DELEGATION_DEPTH = "traceloop.delegation.depth"
DELEGATION_SCOPE_HASH = "traceloop.delegation.scope_hash"
DELEGATION_ACTOR = "traceloop.delegation.actor"
DELEGATION_CHAIN_HASH = "traceloop.delegation.chain_hash"
DELEGATION_PARENT_SCOPE_HASH = "traceloop.delegation.parent_scope_hash"
DELEGATION_CREDENTIAL_REF = "traceloop.delegation.credential_ref"

AUTHZ_DECISION = "traceloop.authz.decision"
AUTHZ_REASON = "traceloop.authz.reason"
AUTHZ_ALLOW = "allow"
AUTHZ_DENY = "deny"

_HASH_ALGORITHMS = ("sha256", "sha384", "sha512", "blake2b")

_MAX_ID_LENGTH = 256
_SCOPE_HASH_RE = re.compile(r"^(sha256|sha384|sha512|blake2b):[0-9a-f]{32,128}$")
# Three base64url segments separated by dots: the shape of a JWT / JWS.
_JWT_RE = re.compile(r"^[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]*$")
_AUTH_SCHEME_RE = re.compile(r"^(bearer|basic|token|apikey)\s", re.IGNORECASE)
# Denial reasons are short machine-readable codes such as "scope_exceeded".
_AUTHZ_REASON_RE = re.compile(r"^[a-z0-9_.:-]{1,64}$")


def _validate_identifier(name: str, value: Any) -> str:
    """Return ``value`` if it is a safe identifier, else raise ``ValueError``.

    Identifiers must be non-empty strings of at most 256 characters with no
    whitespace. Anything shaped like a credential (an ``Authorization`` scheme
    prefix such as ``Bearer`` or a three-segment JWT) is rejected so tokens can
    never end up in span attributes by mistake.
    """
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


def _validate_digest(name: str, value: Any) -> str:
    """Return ``value`` if it is an ``"<algorithm>:<hex digest>"`` string, else raise."""
    if not isinstance(value, str) or not _SCOPE_HASH_RE.match(value):
        raise ValueError(
            f"{name} must look like '<sha256|sha384|sha512|blake2b>:<hex digest>'; "
            f"use hash_{name.split('_')[0]}() to build it"
        )
    return value


def _digest(canonical: str, algorithm: str) -> str:
    """Hash ``canonical`` with ``algorithm`` and return ``"<algorithm>:<hexdigest>"``."""
    if algorithm not in _HASH_ALGORITHMS:
        raise ValueError(f"unsupported hash algorithm: {algorithm}")
    return f"{algorithm}:{hashlib.new(algorithm, canonical.encode('utf-8')).hexdigest()}"


def hash_scope(scope: Any, algorithm: str = "sha256") -> str:
    """Return a stable ``"<algorithm>:<hexdigest>"`` digest of a granted scope.

    ``scope`` may be a string, a list/tuple/set of scope strings (order does not
    matter), or any JSON-serialisable mapping. The scope itself is not recorded.
    """
    if isinstance(scope, (list, tuple, set, frozenset)):
        canonical = json.dumps(sorted(str(s) for s in scope), separators=(",", ":"))
    elif isinstance(scope, str):
        canonical = scope
    else:
        canonical = json.dumps(scope, sort_keys=True, separators=(",", ":"), default=str)
    return _digest(canonical, algorithm)


def hash_chain(chain: Any, algorithm: str = "sha256") -> str:
    """Return a stable ``"<algorithm>:<hexdigest>"`` digest of a verified delegation chain.

    ``chain`` is the ordered list of hops the application has already verified
    (for example principal ids, grant ids or decoded grant claims as mappings).
    Unlike :func:`hash_scope`, order matters: ``["user:1", "agent:a"]`` and
    ``["agent:a", "user:1"]`` are different chains. Mapping keys are sorted so
    equal hops always hash the same. The chain itself is not recorded.
    """
    if isinstance(chain, (str, bytes)) or not isinstance(chain, (list, tuple)):
        raise ValueError("chain must be an ordered list or tuple of hops")
    canonical = json.dumps(list(chain), sort_keys=True, separators=(",", ":"), default=str)
    return _digest(canonical, algorithm)


def build_delegation_attributes(
    root_principal: str,
    chain_id: Optional[str] = None,
    scope_hash: Optional[str] = None,
    depth: Optional[int] = None,
    actor: Optional[str] = None,
    chain_hash: Optional[str] = None,
    *,
    parent_scope_hash: Optional[str] = None,
    credential_ref: Optional[str] = None,
) -> Mapping[str, Any]:
    """Validate the inputs and return the span attributes to record.

    Args:
        root_principal: Who the delegation chain acts for (e.g. ``"user:42"``).
            Required. An identifier, never a token.
        chain_id: Optional reference to the delegation chain / grant, so it can
            be verified out of band.
        scope_hash: Optional digest of the granted scope, from :func:`hash_scope`.
        depth: Optional number of delegation hops so far (non-negative int).
        actor: Optional identifier of the agent / service acting right now
            (e.g. ``"agent:triage"``). Update it with a nested :func:`delegation`
            block when control passes to another agent.
        chain_hash: Optional digest of the verified chain, from :func:`hash_chain`.
        parent_scope_hash: Optional digest of the scope held by the previous hop.
            :func:`delegation` and :func:`attach_delegation` fill it in from the
            enclosing block when it is not given.
        credential_ref: Optional opaque reference to the credential that carried
            the authority (e.g. a token exchange id, a VAID or a SPIFFE ID). An
            identifier only: the credential itself is never accepted.

    Returns:
        A read-only mapping of ``traceloop.delegation.*`` attribute names to values.

    Raises:
        ValueError: If any value is missing, malformed or looks like a credential.
    """
    attributes = {DELEGATION_ROOT_PRINCIPAL: _validate_identifier("root_principal", root_principal)}
    if chain_id is not None:
        attributes[DELEGATION_CHAIN_ID] = _validate_identifier("chain_id", chain_id)
    if scope_hash is not None:
        attributes[DELEGATION_SCOPE_HASH] = _validate_digest("scope_hash", scope_hash)
    if depth is not None:
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
            raise ValueError("depth must be a non-negative integer")
        attributes[DELEGATION_DEPTH] = depth
    if actor is not None:
        attributes[DELEGATION_ACTOR] = _validate_identifier("actor", actor)
    if chain_hash is not None:
        attributes[DELEGATION_CHAIN_HASH] = _validate_digest("chain_hash", chain_hash)
    if parent_scope_hash is not None:
        attributes[DELEGATION_PARENT_SCOPE_HASH] = _validate_digest(
            "parent_scope_hash", parent_scope_hash
        )
    if credential_ref is not None:
        attributes[DELEGATION_CREDENTIAL_REF] = _validate_identifier("credential_ref", credential_ref)
    return MappingProxyType(attributes)


def attach_delegation(
    root_principal: str,
    chain_id: Optional[str] = None,
    scope_hash: Optional[str] = None,
    depth: Optional[int] = None,
    actor: Optional[str] = None,
    chain_hash: Optional[str] = None,
    *,
    parent_scope_hash: Optional[str] = None,
    credential_ref: Optional[str] = None,
) -> object:
    """Attach delegation info to the current context and return a detach token.

    Use this when a ``with`` block does not fit (e.g. framework middleware with
    separate start / end hooks). Arguments are validated exactly as in
    :func:`build_delegation_attributes`. Always pass the returned token to
    :func:`detach_delegation` to restore the previous context.

    When ``scope_hash`` is given and ``parent_scope_hash`` is not, the
    ``scope_hash`` of the enclosing delegation (if any) is recorded as
    ``parent_scope_hash``, so each hop carries its own and its parent's scope.
    """
    if scope_hash is not None and parent_scope_hash is None:
        parent_scope_hash = get_delegation_attributes().get(DELEGATION_SCOPE_HASH)
    attributes = build_delegation_attributes(
        root_principal,
        chain_id,
        scope_hash,
        depth,
        actor,
        chain_hash,
        parent_scope_hash=parent_scope_hash,
        credential_ref=credential_ref,
    )
    return context_api.attach(context_api.set_value(DELEGATION_CONTEXT_KEY, attributes))


def detach_delegation(token: object) -> None:
    """Restore the context that was active before :func:`attach_delegation`.

    Args:
        token: The token returned by :func:`attach_delegation`.
    """
    context_api.detach(token)


@contextmanager
def delegation(
    root_principal: str,
    chain_id: Optional[str] = None,
    scope_hash: Optional[str] = None,
    depth: Optional[int] = None,
    actor: Optional[str] = None,
    chain_hash: Optional[str] = None,
    *,
    parent_scope_hash: Optional[str] = None,
    credential_ref: Optional[str] = None,
) -> Iterator[Mapping[str, Any]]:
    """Record delegation info on agent / handoff spans created inside the block.

    Arguments are validated as in :func:`build_delegation_attributes`. Blocks
    nest: an inner block replaces the outer values until it exits, and an inner
    ``scope_hash`` records the outer one as ``parent_scope_hash``. The context
    is task-local, so concurrent asyncio tasks never see each other's values.

    Yields:
        The read-only mapping of attributes that is active inside the block.
    """
    token = attach_delegation(
        root_principal,
        chain_id,
        scope_hash,
        depth,
        actor,
        chain_hash,
        parent_scope_hash=parent_scope_hash,
        credential_ref=credential_ref,
    )
    try:
        yield context_api.get_value(DELEGATION_CONTEXT_KEY)
    finally:
        detach_delegation(token)


_IDENTIFIER_ATTRIBUTES = frozenset(
    {DELEGATION_ROOT_PRINCIPAL, DELEGATION_CHAIN_ID, DELEGATION_ACTOR, DELEGATION_CREDENTIAL_REF}
)
_DIGEST_ATTRIBUTES = frozenset(
    {DELEGATION_SCOPE_HASH, DELEGATION_CHAIN_HASH, DELEGATION_PARENT_SCOPE_HASH}
)


def validate_delegation_attributes(value: Any) -> Mapping[str, Any]:
    """Return the valid ``traceloop.delegation.*`` attributes contained in ``value``.

    This applies the rules of :func:`build_delegation_attributes` to a mapping
    that did not necessarily come from it, for example one an application stored
    under :data:`DELEGATION_CONTEXT_KEY` directly. Unknown names, wrong types and
    credential-shaped values are dropped rather than raised, so a reader can use
    the result as span attributes as it is. When ``root_principal`` is missing or
    invalid, ``{}`` is returned: the other values cannot be attributed to anyone.
    """
    if not isinstance(value, Mapping):
        return {}
    attributes: dict = {}
    for name, item in value.items():
        try:
            if name in _IDENTIFIER_ATTRIBUTES:
                attributes[name] = _validate_identifier(name, item)
            elif name in _DIGEST_ATTRIBUTES:
                attributes[name] = _validate_digest(name, item)
            elif name == DELEGATION_DEPTH:
                if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                    raise ValueError("depth must be a non-negative integer")
                attributes[name] = item
        except (TypeError, ValueError):
            continue
    if DELEGATION_ROOT_PRINCIPAL not in attributes:
        return {}
    return attributes


def get_delegation_attributes(ctx: Optional[context_api.Context] = None) -> Mapping[str, Any]:
    """Return the delegation span attributes active in ``ctx`` (or the current context).

    The value is passed through :func:`validate_delegation_attributes`, so only
    the approved ``traceloop.delegation.*`` names with values that satisfy the
    builder's rules are returned, even when something other than
    :func:`delegation` stored the mapping. Returns ``{}`` when nothing was set.
    """
    return validate_delegation_attributes(context_api.get_value(DELEGATION_CONTEXT_KEY, ctx))


def record_authorization(
    decision: str,
    reason: Optional[str] = None,
    span: Optional[trace.Span] = None,
) -> None:
    """Record a pre-tool authorization decision on a span.

    Call this from the application's own tool guard *before* the tool runs, so
    the decision lands on the tool (or agent) span even when the call is denied
    and the tool never executes. The library does not make or infer decisions.

    Args:
        decision: ``"allow"`` or ``"deny"``.
        reason: Optional short machine-readable code for a denial, such as
            ``"scope_exceeded"`` (lowercase letters, digits and ``_ . : -``,
            at most 64 characters). Only accepted with ``"deny"``; free-form
            text is rejected so user input or secrets can't leak into traces.
        span: The span to annotate. Defaults to the current span. Nothing is
            recorded when the span is not recording.

    Raises:
        ValueError: If ``decision`` or ``reason`` is invalid.
    """
    if decision not in (AUTHZ_ALLOW, AUTHZ_DENY):
        raise ValueError("decision must be 'allow' or 'deny'")
    if reason is not None:
        if decision != AUTHZ_DENY:
            raise ValueError("reason is only recorded for a 'deny' decision")
        if not isinstance(reason, str) or not _AUTHZ_REASON_RE.match(reason):
            raise ValueError(
                "reason must be a short machine-readable code such as 'scope_exceeded'"
            )
    target = span if span is not None else trace.get_current_span()
    if not target.is_recording():
        return
    target.set_attribute(AUTHZ_DECISION, decision)
    if reason is not None:
        target.set_attribute(AUTHZ_REASON, reason)
