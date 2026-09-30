"""OpenTelemetry Azure AI Search instrumentation"""

import importlib
import logging
from collections.abc import Mapping
from typing import Collection
from urllib.parse import urlparse

from opentelemetry import context as context_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import SpanKind, get_tracer
from opentelemetry.trace.status import Status, StatusCode
from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.azure_search.config import Config
from opentelemetry.instrumentation.azure_search.utils import (
    dont_throw,
    set_span_attribute,
)
from opentelemetry.instrumentation.azure_search.version import __version__

logger = logging.getLogger(__name__)

_instruments = ("azure-search-documents >= 11.4.0",)

WRAPPED_SEARCH_CLIENT_METHODS = [
    {"method": "search", "span_name": "azure_search.search"},
    {"method": "get_document", "span_name": "azure_search.get_document"},
    {"method": "autocomplete", "span_name": "azure_search.autocomplete"},
    {"method": "suggest", "span_name": "azure_search.suggest"},
    {"method": "index_documents", "span_name": "azure_search.index_documents"},
    {"method": "upload_documents", "span_name": "azure_search.upload_documents"},
    {"method": "merge_documents", "span_name": "azure_search.merge_documents"},
    {"method": "merge_or_upload_documents", "span_name": "azure_search.merge_or_upload_documents"},
    {"method": "delete_documents", "span_name": "azure_search.delete_documents"},
    {"method": "get_document_count", "span_name": "azure_search.get_document_count"},
]

WRAPPED_INDEX_CLIENT_METHODS = [
    {"method": "create_index", "span_name": "azure_search.create_index"},
    {"method": "create_or_update_index", "span_name": "azure_search.create_or_update_index"},
    {"method": "delete_index", "span_name": "azure_search.delete_index"},
    {"method": "get_index", "span_name": "azure_search.get_index"},
    {"method": "list_indexes", "span_name": "azure_search.list_indexes"},
    {"method": "get_index_statistics", "span_name": "azure_search.get_index_statistics"},
    {"method": "analyze_text", "span_name": "azure_search.analyze_text"},
    {"method": "get_service_statistics", "span_name": "azure_search.get_service_statistics"},
]

WRAPPED_INDEXER_CLIENT_METHODS = [
    {"method": "create_indexer", "span_name": "azure_search.create_indexer"},
    {"method": "create_or_update_indexer", "span_name": "azure_search.create_or_update_indexer"},
    {"method": "delete_indexer", "span_name": "azure_search.delete_indexer"},
    {"method": "get_indexer", "span_name": "azure_search.get_indexer"},
    {"method": "get_indexers", "span_name": "azure_search.get_indexers"},
    {"method": "get_indexer_status", "span_name": "azure_search.get_indexer_status"},
    {"method": "run_indexer", "span_name": "azure_search.run_indexer"},
    {"method": "reset_indexer", "span_name": "azure_search.reset_indexer"},
    {"method": "create_data_source_connection", "span_name": "azure_search.create_data_source_connection"},
    {"method": "create_skillset", "span_name": "azure_search.create_skillset"},
    {"method": "get_skillset", "span_name": "azure_search.get_skillset"},
    {"method": "delete_skillset", "span_name": "azure_search.delete_skillset"},
]

WRAPPED_METHODS = WRAPPED_SEARCH_CLIENT_METHODS + WRAPPED_INDEX_CLIENT_METHODS + WRAPPED_INDEXER_CLIENT_METHODS


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


@_with_tracer_wrapper
def _wrap(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    span_name = to_wrap.get("span_name")
    with tracer.start_as_current_span(
        span_name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.VECTOR_DB_VENDOR: "azure_search",
        },
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        _set_input_attributes(span, instance, to_wrap, args, kwargs)

        try:
            response = wrapped(*args, **kwargs)
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise

        if span.is_recording():
            _set_response_attributes(span, to_wrap, response)
            span.set_status(Status(StatusCode.OK))

    return response


@dont_throw
def _set_input_attributes(span, instance, to_wrap, args, kwargs):
    method = to_wrap.get("method")

    config = getattr(instance, "_config", None)

    endpoint = getattr(instance, "_endpoint", None) or getattr(config, "endpoint", None)
    if endpoint:
        set_span_attribute(span, "server.address", urlparse(endpoint).hostname or endpoint)

    if method == "search":
        set_span_attribute(
            span,
            SpanAttributes.AZURE_SEARCH_SEARCH_TEXT,
            kwargs.get("search_text"),
        )
        set_span_attribute(span, SpanAttributes.AZURE_SEARCH_TOP, kwargs.get("top"))
        set_span_attribute(
            span,
            SpanAttributes.AZURE_SEARCH_FILTER,
            kwargs.get("filter"),
        )
    elif method == "autocomplete":
        set_span_attribute(
            span,
            SpanAttributes.AZURE_SEARCH_AUTOCOMPLETE_TEXT,
            kwargs.get("search_text"),
        )
    elif method == "suggest":
        set_span_attribute(
            span,
            SpanAttributes.AZURE_SEARCH_SUGGEST_TEXT,
            kwargs.get("search_text"),
        )
    elif method in (
        "index_documents",
        "upload_documents",
        "merge_documents",
        "merge_or_upload_documents",
        "delete_documents",
    ):
        documents = kwargs.get("documents") or (args[0] if args else [])
        set_span_attribute(
            span,
            SpanAttributes.AZURE_SEARCH_DOCUMENTS_COUNT,
            len(documents),
        )

    index_name = getattr(instance, "_index_name", None) or getattr(config, "index_name", None)
    if index_name:
        set_span_attribute(span, SpanAttributes.AZURE_SEARCH_INDEX_NAME, index_name)

    if method in ("get_index", "delete_index", "create_index", "create_or_update_index"):
        index = kwargs.get("index") or kwargs.get("index_name") or kwargs.get("name") or (args[0] if args else None)
        _set_entity_name_attribute(span, SpanAttributes.AZURE_SEARCH_INDEX_NAME, index)

    if method in (
        "get_indexer",
        "delete_indexer",
        "run_indexer",
        "reset_indexer",
        "create_indexer",
        "create_or_update_indexer",
    ):
        indexer = (
            kwargs.get("indexer") or kwargs.get("indexer_name") or kwargs.get("name") or (args[0] if args else None)
        )
        _set_entity_name_attribute(span, SpanAttributes.AZURE_SEARCH_INDEXER_NAME, indexer)
    elif method in ("get_skillset", "delete_skillset", "create_skillset"):
        skillset = (
            kwargs.get("skillset") or kwargs.get("skillset_name") or kwargs.get("name") or (args[0] if args else None)
        )
        _set_entity_name_attribute(span, SpanAttributes.AZURE_SEARCH_SKILLSET_NAME, skillset)


@dont_throw
def _set_entity_name_attribute(span, attribute, entity):
    """Record an entity name from a model instance, mapping, or plain string."""
    if isinstance(entity, str):
        set_span_attribute(span, attribute, entity)
    elif isinstance(entity, Mapping):
        set_span_attribute(span, attribute, entity.get("name"))
    elif hasattr(entity, "name"):
        set_span_attribute(span, attribute, entity.name)


@dont_throw
def _set_response_attributes(span, to_wrap, response):
    method = to_wrap.get("method")

    if method == "search" and response is not None:
        set_span_attribute(span, SpanAttributes.VECTOR_DB_OPERATION, "search")
    elif method == "get_document_count" and response is not None:
        set_span_attribute(span, SpanAttributes.AZURE_SEARCH_DOCUMENTS_COUNT, response)
    elif method in (
        "index_documents",
        "upload_documents",
        "merge_documents",
        "merge_or_upload_documents",
        "delete_documents",
    ):
        if response is not None:
            results = getattr(response, "results", response)
            results = list(results)
            succeeded = sum(1 for r in results if getattr(r, "succeeded", False))
            set_span_attribute(
                span,
                SpanAttributes.AZURE_SEARCH_SUCCEEDED_COUNT,
                succeeded,
            )
            set_span_attribute(
                span,
                SpanAttributes.AZURE_SEARCH_DOCUMENTS_COUNT,
                len(results),
            )
    elif method == "get_indexer_status" and response is not None:
        status = getattr(response, "status", None)
        set_span_attribute(span, SpanAttributes.AZURE_SEARCH_INDEXER_STATUS, status)
    elif method == "get_index_statistics" and response is not None:
        set_span_attribute(
            span,
            SpanAttributes.AZURE_SEARCH_INDEX_DOC_COUNT,
            getattr(response, "document_count", None),
        )
        set_span_attribute(
            span,
            SpanAttributes.AZURE_SEARCH_INDEX_SIZE_BYTES,
            getattr(response, "storage_size", None),
        )
    elif method == "get_service_statistics" and response is not None:
        counters = getattr(response, "counters", None)
        if counters:
            set_span_attribute(
                span,
                SpanAttributes.AZURE_SEARCH_SERVICE_USAGE,
                getattr(counters, "search_service_usage", None),
            )
            set_span_attribute(
                span,
                SpanAttributes.AZURE_SEARCH_SERVICE_LIMIT,
                getattr(counters, "search_service_limit", None),
            )
    elif method in ("create_index", "get_index", "create_or_update_index") and response is not None:
        if hasattr(response, "name"):
            set_span_attribute(span, SpanAttributes.AZURE_SEARCH_INDEX_NAME, response.name)


class AzureSearchInstrumentor(BaseInstrumentor):
    """An instrumentor for the Azure AI Search client library.

    Instruments SearchClient, SearchIndexClient, and SearchIndexerClient
    to emit OpenTelemetry spans for search, indexing, and indexer operations.
    """

    def __init__(self, exception_logger=None):
        super().__init__()
        Config.exception_logger = exception_logger

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        for wrapped_method in WRAPPED_SEARCH_CLIENT_METHODS:
            _instrument_method(
                "azure.search.documents",
                "SearchClient",
                wrapped_method,
                tracer,
            )

        for wrapped_method in WRAPPED_INDEX_CLIENT_METHODS:
            _instrument_method(
                "azure.search.documents.indexes",
                "SearchIndexClient",
                wrapped_method,
                tracer,
            )

        for wrapped_method in WRAPPED_INDEXER_CLIENT_METHODS:
            _instrument_method(
                "azure.search.documents.indexes",
                "SearchIndexerClient",
                wrapped_method,
                tracer,
            )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_SEARCH_CLIENT_METHODS:
            _uninstrument_method(
                "azure.search.documents",
                "SearchClient",
                wrapped_method,
            )

        for wrapped_method in WRAPPED_INDEX_CLIENT_METHODS:
            _uninstrument_method(
                "azure.search.documents.indexes",
                "SearchIndexClient",
                wrapped_method,
            )

        for wrapped_method in WRAPPED_INDEXER_CLIENT_METHODS:
            _uninstrument_method(
                "azure.search.documents.indexes",
                "SearchIndexerClient",
                wrapped_method,
            )


def _instrument_method(module, class_name, wrapped_method, tracer):
    method = wrapped_method.get("method")
    try:
        mod = importlib.import_module(module)
        cls = getattr(mod, class_name, None)
        if cls and hasattr(cls, method) and callable(getattr(cls, method)):
            wrap_function_wrapper(
                module,
                f"{class_name}.{method}",
                _wrap(tracer, wrapped_method),
            )
    except (ImportError, ModuleNotFoundError):
        pass


def _uninstrument_method(module, class_name, wrapped_method):
    method = wrapped_method.get("method")
    try:
        unwrap(f"{module}.{class_name}", method)
    except Exception as e:
        logger.debug(
            "Failed to unwrap %s.%s.%s: %s",
            module,
            class_name,
            method,
            e,
        )
