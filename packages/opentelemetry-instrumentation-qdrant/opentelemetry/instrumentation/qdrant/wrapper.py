from opentelemetry import context as context_api
from opentelemetry.instrumentation.qdrant.utils import dont_throw
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
)
from opentelemetry.trace import SpanKind
from opentelemetry.semconv_ai import SpanAttributes


def _set_span_attribute(span, name, value):
    """Set an attribute unless its value is None or an empty string."""
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, to_wrap):
        """Bind the tracer and method metadata to a wrapt callback."""
        def wrapper(wrapped, instance, args, kwargs):
            """Invoke the tracing callback with its configured tracer and method."""
            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


@_with_tracer_wrapper
def _wrap(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    method = to_wrap.get("method")
    with tracer.start_as_current_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.VECTOR_DB_VENDOR: "Qdrant",
        },
    ) as span:
        _set_request_attributes(span, method, args, kwargs)
        response = wrapped(*args, **kwargs)
        if response:
            span.set_status(Status(StatusCode.OK))
    return response


@_with_tracer_wrapper
async def _awrap(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instrument an async call until its result is available."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)

    with tracer.start_as_current_span(
        to_wrap.get("span_name"),
        kind=SpanKind.CLIENT,
        attributes={SpanAttributes.VECTOR_DB_VENDOR: "Qdrant"},
    ) as span:
        _set_request_attributes(span, to_wrap.get("method"), args, kwargs)
        response = await wrapped(*args, **kwargs)
        if response:
            span.set_status(Status(StatusCode.OK))
    return response


def _set_request_attributes(span, method, args, kwargs):
    """Record collection and operation-specific attributes from a Qdrant call."""
    _set_collection_name_attribute(span, method, args, kwargs)
    if method == "upsert":
        _set_upsert_attributes(span, args, kwargs)
    elif method == "add":
        _set_upload_attributes(span, args, kwargs, method, "documents")
    elif method == "upload_points":
        _set_upload_attributes(span, args, kwargs, method, "points")
    elif method == "upload_records":
        _set_upload_attributes(span, args, kwargs, method, "records")
    elif method == "upload_collection":
        _set_upload_attributes(span, args, kwargs, method, "vectors")
    elif method in [
        "search",
        "search_groups",
        "query",
        "query_points",
        "discover",
        "recommend",
        "recommend_groups",
    ]:
        _set_search_attributes(span, args, kwargs)
    elif method in ["search_batch", "query_batch_points", "recommend_batch", "discover_batch"]:
        _set_batch_search_attributes(span, args, kwargs, method)


@dont_throw
def _set_collection_name_attribute(span, method, args, kwargs):
    """Record the collection name using legacy attribute names for query methods."""
    attribute_method = method
    if method == "query_points":
        attribute_method = "search"
    elif method == "query_batch_points":
        attribute_method = "search_batch"

    _set_span_attribute(
        span,
        f"qdrant.{attribute_method}.collection_name",
        kwargs.get("collection_name") or args[0],
    )


@dont_throw
def _set_upsert_attributes(span, args, kwargs):
    """Record the number of points supplied as a list or batch for upsert."""
    points = kwargs.get("points") or args[1]
    if isinstance(points, list):
        length = len(points)
    else:
        length = len(
            points.ids
        )  # If using models.Batch instead of list[models.PointStruct]
    _set_span_attribute(span, SpanAttributes.QDRANT_UPSERT_POINTS_COUNT, length)


@dont_throw
def _set_upload_attributes(span, args, kwargs, method_name, param_name):
    """Record the item count from the upload method's input parameter."""
    points = list(kwargs.get(param_name) or args[1])
    _set_span_attribute(span, f"qdrant.{method_name}.points_count", len(points))


@dont_throw
def _set_search_attributes(span, args, kwargs):
    """Record the requested search limit, defaulting to ten results."""
    limit = kwargs.get("limit") or 10
    _set_span_attribute(span, SpanAttributes.VECTOR_DB_QUERY_TOP_K, limit)


@dont_throw
def _set_batch_search_attributes(span, args, kwargs, method):
    """Record the batch request count while preserving query attribute names."""
    requests = kwargs.get("requests") or []
    # Map query_batch_points to search_batch for backward compatibility
    attribute_method = "search_batch" if method == "query_batch_points" else method
    _set_span_attribute(span, f"qdrant.{attribute_method}.requests_count", len(requests))
