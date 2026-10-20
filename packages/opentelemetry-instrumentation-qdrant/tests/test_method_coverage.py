"""Guards against the wrapped-method lists drifting from the qdrant-client API.

The instrumentor skips any method that is missing from the installed client
(see ``QdrantInstrumentor._instrument``), so a renamed or removed method
degrades silently: no error, and no spans. These tests fail loudly instead.
"""

import qdrant_client

from opentelemetry.instrumentation.qdrant import (
    ASYNC_QDRANT_CLIENT_METHODS,
    QDRANT_CLIENT_METHODS,
)

# Methods removed from qdrant-client in 1.12, kept in the lists so that users
# still on the >= 1.7 clients this package supports keep their instrumentation.
LEGACY_METHODS = {
    "add",
    "discover",
    "discover_batch",
    "query",
    "query_batch",
    "recommend",
    "recommend_batch",
    "recommend_groups",
    "search",
    "search_batch",
    "search_groups",
    "upload_records",
}

# The search/query surface a modern client must have instrumented.
REQUIRED_SEARCH_METHODS = {
    "query_points",
    "query_points_groups",
    "query_batch_points",
}


def _methods(entries):
    return {entry["method"] for entry in entries}


def _resolve(entry):
    obj = getattr(qdrant_client, entry["object"], None)
    return obj is not None and hasattr(obj, entry["method"])


def test_sync_and_async_lists_cover_the_same_methods():
    """The two lists drifted apart once; query_points was sync-only."""
    assert _methods(QDRANT_CLIENT_METHODS) == _methods(ASYNC_QDRANT_CLIENT_METHODS)


def test_required_search_methods_are_instrumented():
    for entries in (QDRANT_CLIENT_METHODS, ASYNC_QDRANT_CLIENT_METHODS):
        assert REQUIRED_SEARCH_METHODS <= _methods(entries)


def test_every_non_legacy_entry_resolves_on_the_installed_client():
    """A non-legacy entry that no longer resolves means silent span loss."""
    unresolved = [
        f"{entry['object']}.{entry['method']}"
        for entry in QDRANT_CLIENT_METHODS + ASYNC_QDRANT_CLIENT_METHODS
        if entry["method"] not in LEGACY_METHODS and not _resolve(entry)
    ]
    assert not unresolved, (
        "Wrapped methods missing from the installed qdrant-client, so no spans "
        f"will be emitted for them: {unresolved}"
    )


def test_span_names_are_unique_and_prefixed():
    entries = QDRANT_CLIENT_METHODS + ASYNC_QDRANT_CLIENT_METHODS
    for entry in entries:
        assert entry["span_name"] == f"qdrant.{entry['method']}"
