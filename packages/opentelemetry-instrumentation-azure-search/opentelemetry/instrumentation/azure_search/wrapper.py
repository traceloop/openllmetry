import asyncio
import logging

from opentelemetry import context as context_api
from opentelemetry.instrumentation.azure_search.utils import dont_throw
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
)
from opentelemetry.trace import SpanKind
from opentelemetry.semconv_ai import SpanAttributes

logger = logging.getLogger(__name__)


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


def _set_request_attributes(span, method, instance, args, kwargs):
    """Set all pre-call span attributes based on the method being called."""
    _set_index_name_attribute(span, instance, args, kwargs)

    if method == "search":
        _set_search_attributes(span, args, kwargs)
    elif method == "get_document":
        _set_get_document_attributes(span, args, kwargs)
    elif method in [
        "upload_documents",
        "merge_documents",
        "delete_documents",
        "merge_or_upload_documents",
    ]:
        _set_document_batch_attributes(span, args, kwargs)
    elif method == "index_documents":
        _set_index_documents_attributes(span, args, kwargs)
    elif method in ["autocomplete", "suggest"]:
        _set_suggestion_attributes(span, args, kwargs)
    elif method in [
        "create_index",
        "create_or_update_index",
        "delete_index",
        "get_index",
        "get_index_statistics",
    ]:
        _set_index_management_attributes(span, method, args, kwargs)
    elif method == "analyze_text":
        _set_analyze_text_attributes(span, args, kwargs)
    elif method in [
        "create_indexer",
        "create_or_update_indexer",
        "delete_indexer",
        "get_indexer",
        "get_indexers",
        "run_indexer",
        "reset_indexer",
        "get_indexer_status",
    ]:
        _set_indexer_management_attributes(span, method, args, kwargs)
    elif method in [
        "create_data_source_connection",
        "create_or_update_data_source_connection",
        "delete_data_source_connection",
        "get_data_source_connection",
        "get_data_source_connections",
    ]:
        _set_data_source_attributes(span, method, args, kwargs)
    elif method in [
        "create_skillset",
        "create_or_update_skillset",
        "delete_skillset",
        "get_skillset",
        "get_skillsets",
    ]:
        _set_skillset_attributes(span, method, args, kwargs)


def _set_response_attributes(span, method, response, args, kwargs):
    """Set all post-call span attributes from the response."""
    if response is None:
        return

    if method == "search":
        _set_search_response_attributes(span, response)
    elif method == "get_document_count":
        _set_document_count_response_attributes(span, response)
    elif method in [
        "upload_documents",
        "merge_documents",
        "delete_documents",
        "merge_or_upload_documents",
    ]:
        _set_document_batch_response_attributes(span, response)
    elif method == "index_documents":
        _set_index_documents_response_attributes(span, response)
    elif method == "autocomplete":
        _set_autocomplete_response_attributes(span, response)
    elif method == "suggest":
        _set_suggest_response_attributes(span, response)
    elif method == "get_indexer_status":
        _set_indexer_status_attributes(span, args, kwargs, response)


@_with_tracer_wrapper
def _wrap(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in WRAPPED_METHODS."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    if asyncio.iscoroutinefunction(wrapped):
        return _async_wrap(tracer, to_wrap, wrapped, instance, args, kwargs)

    return _sync_wrap(tracer, to_wrap, wrapped, instance, args, kwargs)


def _sync_wrap(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Synchronous instrumentation wrapper."""
    name = to_wrap.get("span_name")
    method = to_wrap.get("method")

    with tracer.start_as_current_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
        },
        set_status_on_exception=False,
    ) as span:
        _set_request_attributes(span, method, instance, args, kwargs)

        try:
            response = wrapped(*args, **kwargs)
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise

        _set_response_attributes(span, method, response, args, kwargs)
        span.set_status(Status(StatusCode.OK))
        return response


async def _async_wrap(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Asynchronous instrumentation wrapper."""
    name = to_wrap.get("span_name")
    method = to_wrap.get("method")

    with tracer.start_as_current_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.VECTOR_DB_VENDOR: "Azure AI Search",
        },
        set_status_on_exception=False,
    ) as span:
        _set_request_attributes(span, method, instance, args, kwargs)

        try:
            response = await wrapped(*args, **kwargs)
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise

        _set_response_attributes(span, method, response, args, kwargs)
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

    query_type = kwargs.get("query_type")
    if query_type is not None:
        # query_type can be a string or a QueryType enum
        qt_str = query_type.value if hasattr(query_type, "value") else str(query_type)
        _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_SEARCH_QUERY_TYPE, qt_str)

    # Set top_k for vector DB convention
    top = kwargs.get("top")
    if top:
        _set_span_attribute(span, SpanAttributes.VECTOR_DB_QUERY_TOP_K, top)

    # Vector search attributes
    _set_vector_search_attributes(span, kwargs)

    # Semantic search attributes
    _set_semantic_search_attributes(span, kwargs)

    # Additional search parameters
    _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_SEARCH_MODE, kwargs.get("search_mode"))
    _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_SCORING_PROFILE, kwargs.get("scoring_profile"))

    select = kwargs.get("select")
    if select:
        if isinstance(select, (list, tuple)):
            select = ",".join(select)
        _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_SELECT, select)

    search_fields = kwargs.get("search_fields")
    if search_fields:
        if isinstance(search_fields, (list, tuple)):
            search_fields = ",".join(search_fields)
        _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_SEARCH_FIELDS, search_fields)


@dont_throw
def _set_vector_search_attributes(span, kwargs):
    """Set attributes for vector search queries."""
    vector_queries = kwargs.get("vector_queries")
    if not vector_queries:
        return

    _set_span_attribute(
        span, SpanAttributes.AZURE_SEARCH_VECTOR_QUERIES_COUNT, len(vector_queries)
    )

    # Extract details from the first vector query for primary attributes
    first_vq = vector_queries[0]

    k = getattr(first_vq, "k_nearest_neighbors", None) or getattr(first_vq, "k", None)
    _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_VECTOR_K_NEAREST_NEIGHBORS, k)

    fields = getattr(first_vq, "fields", None)
    if fields:
        if isinstance(fields, (list, tuple)):
            fields = ",".join(fields)
        _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_VECTOR_FIELDS, fields)

    exhaustive = getattr(first_vq, "exhaustive", None)
    if exhaustive is not None:
        _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_VECTOR_EXHAUSTIVE, exhaustive)

    # Vector filter mode is a top-level search param, not per-query
    vector_filter_mode = kwargs.get("vector_filter_mode")
    if vector_filter_mode is not None:
        vfm_str = vector_filter_mode.value if hasattr(vector_filter_mode, "value") else str(vector_filter_mode)
        _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_VECTOR_FILTER_MODE, vfm_str)


@dont_throw
def _set_semantic_search_attributes(span, kwargs):
    """Set attributes for semantic search configuration."""
    _set_span_attribute(
        span,
        SpanAttributes.AZURE_SEARCH_SEMANTIC_CONFIGURATION_NAME,
        kwargs.get("semantic_configuration_name"),
    )

    query_caption = kwargs.get("query_caption")
    if query_caption is not None:
        qc_str = query_caption.value if hasattr(query_caption, "value") else str(query_caption)
        _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_QUERY_CAPTION, qc_str)

    query_answer = kwargs.get("query_answer")
    if query_answer is not None:
        qa_str = query_answer.value if hasattr(query_answer, "value") else str(query_answer)
        _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_QUERY_ANSWER, qa_str)


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


@dont_throw
def _set_indexer_management_attributes(span, method, args, kwargs):
    """Set attributes for indexer management operations."""
    # Extract indexer name from various sources
    indexer_name = None

    # For create/update operations: first arg is indexer object
    if method in ["create_indexer", "create_or_update_indexer"]:
        indexer = kwargs.get("indexer") or (args[0] if args else None)
        if indexer:
            indexer_name = getattr(indexer, "name", None)

    # For other operations: first arg is indexer_name string
    else:
        indexer_name = kwargs.get("name") or kwargs.get("indexer_name") or (args[0] if args else None)

    _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_INDEXER_NAME, indexer_name)


@dont_throw
def _set_indexer_status_attributes(span, args, kwargs, response):
    """Set attributes for get_indexer_status response."""
    if response:
        # Extract status information from response
        status = getattr(response, "status", None)
        if status:
            _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_INDEXER_STATUS, str(status))

        # Extract execution counts if available
        last_result = getattr(response, "last_result", None)
        if last_result:
            items_processed = getattr(last_result, "items_processed", None)
            items_failed = getattr(last_result, "items_failed", None)
            _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_DOCUMENTS_PROCESSED, items_processed)
            _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_DOCUMENTS_FAILED, items_failed)


@dont_throw
def _set_data_source_attributes(span, method, args, kwargs):
    """Set attributes for data source operations."""
    data_source_name = None
    data_source_type = None

    # For create/update operations: first arg is data source object
    if method in ["create_data_source_connection", "create_or_update_data_source_connection"]:
        data_source = kwargs.get("data_source_connection") or (args[0] if args else None)
        if data_source:
            data_source_name = getattr(data_source, "name", None)
            # Get data source type (e.g., azureblob, azuresql, cosmosdb)
            data_source_type = getattr(data_source, "type", None)

    # For other operations: first arg is data_source_name string
    else:
        data_source_name = kwargs.get("name") or kwargs.get("data_source_name") or (args[0] if args else None)

    _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_DATA_SOURCE_NAME, data_source_name)
    _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_DATA_SOURCE_TYPE, data_source_type)


@dont_throw
def _set_skillset_attributes(span, method, args, kwargs):
    """Set attributes for skillset operations."""
    skillset_name = None
    skill_count = None

    # For create/update operations: first arg is skillset object
    if method in ["create_skillset", "create_or_update_skillset"]:
        skillset = kwargs.get("skillset") or (args[0] if args else None)
        if skillset:
            skillset_name = getattr(skillset, "name", None)
            # Count skills in skillset
            skills = getattr(skillset, "skills", None)
            if skills and hasattr(skills, "__len__"):
                skill_count = len(skills)

    # For other operations: first arg is skillset_name string
    else:
        skillset_name = kwargs.get("name") or kwargs.get("skillset_name") or (args[0] if args else None)

    _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_SKILLSET_NAME, skillset_name)
    _set_span_attribute(span, SpanAttributes.AZURE_SEARCH_SKILLSET_SKILL_COUNT, skill_count)


# --- Response attribute extraction functions ---


@dont_throw
def _set_search_response_attributes(span, response):
    """Set attributes from search response (SearchItemPaged)."""
    # get_count() returns total count when include_total_count=True
    count = getattr(response, "get_count", None)
    if callable(count):
        total = count()
        if total is not None:
            _set_span_attribute(
                span,
                SpanAttributes.AZURE_SEARCH_SEARCH_RESULTS_COUNT,
                total,
            )


@dont_throw
def _set_document_count_response_attributes(span, response):
    """Set attributes from get_document_count response (int)."""
    if isinstance(response, int):
        _set_span_attribute(
            span, SpanAttributes.AZURE_SEARCH_DOCUMENT_COUNT, response
        )


@dont_throw
def _set_document_batch_response_attributes(span, response):
    """Set attributes from document batch response (List[IndexingResult])."""
    if isinstance(response, list) and len(response) > 0:
        succeeded = sum(
            1
            for r in response
            if getattr(r, "succeeded", False)
        )
        failed = len(response) - succeeded
        _set_span_attribute(
            span,
            SpanAttributes.AZURE_SEARCH_DOCUMENT_SUCCEEDED_COUNT,
            succeeded,
        )
        _set_span_attribute(
            span,
            SpanAttributes.AZURE_SEARCH_DOCUMENT_FAILED_COUNT,
            failed,
        )


@dont_throw
def _set_index_documents_response_attributes(span, response):
    """Set attributes from index_documents response."""
    # index_documents returns an IndexDocumentsResult with a .results list
    results = getattr(response, "results", None)
    if results and isinstance(results, list):
        succeeded = sum(
            1
            for r in results
            if getattr(r, "succeeded", False)
        )
        failed = len(results) - succeeded
        _set_span_attribute(
            span,
            SpanAttributes.AZURE_SEARCH_DOCUMENT_SUCCEEDED_COUNT,
            succeeded,
        )
        _set_span_attribute(
            span,
            SpanAttributes.AZURE_SEARCH_DOCUMENT_FAILED_COUNT,
            failed,
        )


@dont_throw
def _set_autocomplete_response_attributes(span, response):
    """Set attributes from autocomplete response (list)."""
    if isinstance(response, list):
        _set_span_attribute(
            span,
            SpanAttributes.AZURE_SEARCH_AUTOCOMPLETE_RESULTS_COUNT,
            len(response),
        )


@dont_throw
def _set_suggest_response_attributes(span, response):
    """Set attributes from suggest response (list)."""
    if isinstance(response, list):
        _set_span_attribute(
            span,
            SpanAttributes.AZURE_SEARCH_SUGGEST_RESULTS_COUNT,
            len(response),
        )
