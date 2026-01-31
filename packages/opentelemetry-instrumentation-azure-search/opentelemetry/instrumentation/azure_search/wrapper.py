from opentelemetry import context as context_api
from opentelemetry.instrumentation.azure_search.utils import dont_throw
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
)
from opentelemetry.trace import SpanKind
from opentelemetry.semconv_ai import SpanAttributes


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


@_with_tracer_wrapper
def _wrap(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in WRAPPED_METHODS."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    method = to_wrap.get("method")

    with tracer.start_as_current_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
        },
    ) as span:
        # Set index name from instance if available
        _set_index_name_attribute(span, instance, args, kwargs)

        # Set method-specific attributes
        if method == "search":
            _set_search_attributes(span, args, kwargs)
        elif method == "get_document":
            _set_get_document_attributes(span, args, kwargs)
        elif method in ["upload_documents", "merge_documents", "delete_documents",
                        "merge_or_upload_documents"]:
            _set_document_batch_attributes(span, args, kwargs)
        elif method == "index_documents":
            _set_index_documents_attributes(span, args, kwargs)
        elif method in ["autocomplete", "suggest"]:
            _set_suggestion_attributes(span, args, kwargs)
        elif method in ["create_index", "create_or_update_index", "delete_index",
                        "get_index", "get_index_statistics"]:
            _set_index_management_attributes(span, method, args, kwargs)
        elif method == "analyze_text":
            _set_analyze_text_attributes(span, args, kwargs)

        response = wrapped(*args, **kwargs)
        if response is not None:
            span.set_status(Status(StatusCode.OK))
        return response


@dont_throw
def _set_index_name_attribute(span, instance, args, kwargs):
    """Extract and set index name from instance or arguments."""
    # SearchClient stores index name in _index_name
    index_name = getattr(instance, "_index_name", None)
    if index_name:
        _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_INDEX_NAME, index_name)


@dont_throw
def _set_search_attributes(span, args, kwargs):
    """Set attributes for search operations."""
    # search_text can be positional or keyword
    search_text = kwargs.get("search_text") or (args[0] if args else None)
    _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_SEARCH_TEXT, search_text)

    _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_SEARCH_TOP, kwargs.get("top"))
    _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_SEARCH_SKIP, kwargs.get("skip"))
    _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_SEARCH_FILTER, kwargs.get("filter"))
    _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_SEARCH_QUERY_TYPE, kwargs.get("query_type"))

    # Set top_k for vector DB convention
    top = kwargs.get("top")
    if top:
        _set_span_attribute(span, SpanAttributes.VECTOR_DB_QUERY_TOP_K, top)


@dont_throw
def _set_get_document_attributes(span, args, kwargs):
    """Set attributes for get_document operation."""
    key = kwargs.get("key") or (args[0] if args else None)
    _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_DOCUMENT_KEY, key)


@dont_throw
def _set_document_batch_attributes(span, args, kwargs):
    """Set attributes for document batch operations (upload, merge, delete)."""
    documents = kwargs.get("documents") or (args[0] if args else None)
    if documents:
        if hasattr(documents, "__len__"):
            _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT, len(documents))
        else:
            # Try to convert to list for generators
            try:
                docs_list = list(documents)
                _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT, len(docs_list))
            except (TypeError, ValueError):
                pass


@dont_throw
def _set_index_documents_attributes(span, args, kwargs):
    """Set attributes for index_documents batch operation."""
    batch = kwargs.get("batch") or (args[0] if args else None)
    if batch:
        # IndexDocumentsBatch has actions property
        actions = getattr(batch, "actions", None)
        if actions and hasattr(actions, "__len__"):
            _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT, len(actions))


@dont_throw
def _set_suggestion_attributes(span, args, kwargs):
    """Set attributes for autocomplete and suggest operations."""
    search_text = kwargs.get("search_text") or (args[0] if args else None)
    _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_SEARCH_TEXT, search_text)

    suggester_name = kwargs.get("suggester_name") or (args[1] if len(args) > 1 else None)
    _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_SUGGESTER_NAME, suggester_name)


@dont_throw
def _set_index_management_attributes(span, method, args, kwargs):
    """Set attributes for index management operations."""
    if method in ["create_index", "create_or_update_index"]:
        # index is the first argument
        index = kwargs.get("index") or (args[0] if args else None)
        if index:
            index_name = getattr(index, "name", None)
            _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_INDEX_NAME, index_name)
    elif method in ["delete_index", "get_index", "get_index_statistics"]:
        # index_name is the first argument (string)
        index_name = kwargs.get("index") or kwargs.get("index_name") or (args[0] if args else None)
        if isinstance(index_name, str):
            _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_INDEX_NAME, index_name)
        elif hasattr(index_name, "name"):
            _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_INDEX_NAME, index_name.name)


@dont_throw
def _set_analyze_text_attributes(span, args, kwargs):
    """Set attributes for analyze_text operation."""
    index_name = kwargs.get("index_name") or (args[0] if args else None)
    _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_INDEX_NAME, index_name)

    # Analyzer can be in analyze_request object or as direct kwargs
    analyze_request = kwargs.get("analyze_request") or (args[1] if len(args) > 1 else None)
    analyzer_name = None

    if analyze_request:
        # AnalyzeTextOptions has analyzer_name attribute
        analyzer_name = getattr(analyze_request, "analyzer_name", None)

    # Fallback to direct kwargs
    if not analyzer_name:
        analyzer_name = kwargs.get("analyzer_name") or kwargs.get("analyzer")

    if analyzer_name:
        # analyzer_name can be a string or an enum
        if hasattr(analyzer_name, "value"):
            analyzer_name = analyzer_name.value
        _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_ANALYZER_NAME, str(analyzer_name))
