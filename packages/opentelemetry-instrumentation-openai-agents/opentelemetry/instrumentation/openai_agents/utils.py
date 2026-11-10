import asyncio
import dataclasses
import functools
import inspect
import json
import logging
import os
import re
import traceback
from collections.abc import Mapping
from opentelemetry import context as context_api

# Handoff span attribute names
GEN_AI_HANDOFF_FROM_AGENT = "gen_ai.handoff.from_agent"
GEN_AI_HANDOFF_TO_AGENT = "gen_ai.handoff.to_agent"
GEN_AI_HANDOFF_PARENT_AGENT = "gen_ai.agent.handoff_parent"
OPENAI_AGENT_HANDOFFS = "openai.agent.handoffs"
_TRACELOOP_TRACE_CONTENT = "TRACELOOP_TRACE_CONTENT"


# Delegation / authority attributes are opt-in: the application declares them with
# opentelemetry.semconv_ai.delegation.delegation(...), which stores validated
# span attributes in the OTel context under this key. Read through the plain key
# so this also works with semconv-ai releases that predate the helper module.
_DELEGATION_CONTEXT_KEY = "traceloop.delegation"


_DELEGATION_ROOT_PRINCIPAL = _DELEGATION_CONTEXT_KEY + ".root_principal"
_DELEGATION_DEPTH = _DELEGATION_CONTEXT_KEY + ".depth"
_DELEGATION_IDENTIFIERS = frozenset(
    _DELEGATION_CONTEXT_KEY + "." + n for n in ("root_principal", "chain_id", "actor", "credential_ref")
)
_DELEGATION_DIGESTS = frozenset(
    _DELEGATION_CONTEXT_KEY + "." + n for n in ("scope_hash", "chain_hash", "parent_scope_hash")
)
# Same rules as opentelemetry.semconv_ai.delegation: identifiers are short, have no
# whitespace and are not shaped like a credential; digests are "<algorithm>:<hex>".
_DELEGATION_DIGEST_RE = re.compile(r"(sha256|sha384|sha512|blake2b):[0-9a-f]{32,128}")
_DELEGATION_JWT_RE = re.compile(r"^[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]*$")
_DELEGATION_AUTH_SCHEME_RE = re.compile(r"^(bearer|basic|token|apikey)\s", re.IGNORECASE)

try:  # semconv-ai releases with the helper ship the validator; reuse it when present.
    from opentelemetry.semconv_ai.delegation import (
        validate_delegation_attributes as _validate_delegation_attributes,
    )
except ImportError:  # pragma: no cover - exercised against older semconv-ai releases
    _validate_delegation_attributes = None


def _is_delegation_identifier(value) -> bool:
    """Return whether ``value`` is an identifier the delegation contract accepts."""
    return (
        isinstance(value, str)
        and bool(value.strip())
        and len(value) <= 256
        and not any(ch.isspace() for ch in value)
        and not _DELEGATION_AUTH_SCHEME_RE.match(value)
        and not _DELEGATION_JWT_RE.match(value)
    )


def _validate_delegation_locally(value) -> dict:
    """Apply the delegation contract's rules without importing the helper module."""
    if not isinstance(value, Mapping):
        return {}
    attributes = {}
    for name, item in value.items():
        if name in _DELEGATION_IDENTIFIERS and _is_delegation_identifier(item):
            attributes[name] = item
        elif name in _DELEGATION_DIGESTS and isinstance(item, str) and _DELEGATION_DIGEST_RE.fullmatch(item):
            attributes[name] = item
        elif (
            name == _DELEGATION_DEPTH
            and isinstance(item, int)
            and not isinstance(item, bool)
            and item >= 0
        ):
            attributes[name] = item
    if _DELEGATION_ROOT_PRINCIPAL not in attributes:
        return {}
    return attributes


def get_delegation_attributes() -> dict:
    """Return the delegation attributes the application put in the current context.

    The mapping is validated against the delegation contract before it is used:
    only the approved ``traceloop.delegation.*`` names are kept, identifiers must
    not look like credentials, digests must be ``"<algorithm>:<hex>"`` and depth a
    non-negative int. Without a valid root principal nothing is returned. So a
    mapping stored under the context key by hand never puts a token on a span.
    Returns an empty dict when the application did not opt in.
    """
    value = context_api.get_value(_DELEGATION_CONTEXT_KEY)
    if _validate_delegation_attributes is not None:
        return dict(_validate_delegation_attributes(value))
    return _validate_delegation_locally(value)


def set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


def _is_truthy(value):
    return str(value).strip().lower() in ("true", "1", "yes", "on")


def should_send_prompts() -> bool:
    """Determine if LLM content tracing should be enabled.

    Content includes not only prompts, but also responses.
    """
    env_setting = os.getenv(_TRACELOOP_TRACE_CONTENT, "true")
    override = context_api.get_value("override_enable_content_tracing")
    return _is_truthy(env_setting) or _is_truthy(override)


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)

        if hasattr(o, "to_json"):
            return o.to_json()

        if hasattr(o, "model_dump"):
            return o.model_dump()

        if hasattr(o, "dict"):
            dict_method = o.dict
            if callable(dict_method) and not inspect.iscoroutinefunction(dict_method):
                result = dict_method()
                if not inspect.iscoroutine(result):
                    return result
                result.close()

        if hasattr(o, "json"):
            json_method = o.json
            if callable(json_method) and not inspect.iscoroutinefunction(json_method):
                result = json_method()
                if inspect.iscoroutine(result):
                    result.close()
                elif isinstance(result, str):
                    # .json() returns a JSON string; parse to avoid double-encoding.
                    try:
                        return json.loads(result)
                    except (ValueError, TypeError):
                        return result
                else:
                    return result

        if hasattr(o, "__class__"):
            return o.__class__.__name__

        return super().default(o)


def dont_throw(func):
    """
    A decorator that wraps the passed in function and logs exceptions instead of throwing them.
    Works for both synchronous and asynchronous functions.
    """
    logger = logging.getLogger(func.__module__)

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            _handle_exception(e, func, logger)

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            _handle_exception(e, func, logger)

    def _handle_exception(e, func, logger):
        logger.debug(
            "OpenLLMetry failed to trace in %s, error: %s",
            func.__name__,
            traceback.format_exc(),
        )

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
