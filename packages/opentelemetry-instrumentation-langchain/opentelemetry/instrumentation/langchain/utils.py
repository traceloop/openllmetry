import dataclasses
import datetime
import importlib.util
import json
import logging
import os
import re
import traceback
from collections.abc import Mapping

from opentelemetry import context as context_api
from opentelemetry._logs import Logger
from opentelemetry.instrumentation.langchain.config import Config
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from pydantic import BaseModel

TRACELOOP_TRACE_CONTENT = "TRACELOOP_TRACE_CONTENT"

EVENT_ATTRIBUTES = {GenAIAttributes.GEN_AI_PROVIDER_NAME: "langchain"}


class CallbackFilteredJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, dict):
            if "callbacks" in o:
                del o["callbacks"]
                return o

        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)

        if hasattr(o, "to_json"):
            return o.to_json()

        if isinstance(o, BaseModel) and hasattr(o, "model_dump"):
            return o.model_dump()

        if isinstance(o, datetime.datetime):
            return o.isoformat()

        try:
            return str(o)
        except Exception:
            logger = logging.getLogger(__name__)
            logger.debug("Failed to serialize object of type: %s", type(o).__name__)
            return ""


def should_send_prompts():
    return (
        os.getenv(TRACELOOP_TRACE_CONTENT) or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")


def dont_throw(func):
    """
    A decorator that wraps the passed in function and logs exceptions instead of throwing them.

    @param func: The function to wrap
    @return: The wrapper function
    """
    # Obtain a logger specific to the function's module
    logger = logging.getLogger(func.__module__)

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.debug(
                "OpenLLMetry failed to trace in %s, error: %s",
                func.__name__,
                traceback.format_exc(),
            )
            if Config.exception_logger:
                Config.exception_logger(e)

    return wrapper


def should_emit_events() -> bool:
    """
    Checks if the instrumentation isn't using the legacy attributes
    and if the event logger is not None.
    """
    return not Config.use_legacy_attributes and isinstance(
        Config.event_logger, Logger
    )


def is_package_available(package_name):
    return importlib.util.find_spec(package_name) is not None


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
