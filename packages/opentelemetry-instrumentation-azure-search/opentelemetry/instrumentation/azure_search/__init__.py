"""OpenTelemetry Azure AI Search instrumentation"""

import logging
import time
from typing import Collection

from wrapt import wrap_function_wrapper

from opentelemetry import context as context_api
from opentelemetry.trace import get_tracer, SpanKind
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)
from opentelemetry.instrumentation.azure_search.version import __version__

logger = logging.getLogger(__name__)

_instruments = ("azure-search-documents >= 11.0.0",)

WRAPPED_METHODS = [
    {
        "object": "SearchClient",
        "method": "search",
        "span_name": "azure_search.search",
    },
    {
        "object": "SearchClient",
        "method": "upload_documents",
        "span_name": "azure_search.upload_documents",
    },
    {
        "object": "SearchClient",
        "method": "merge_documents",
        "span_name": "azure_search.merge_documents",
    },
    {
        "object": "SearchClient",
        "method": "merge_or_upload_documents",
        "span_name": "azure_search.merge_or_upload_documents",
    },
    {
        "object": "SearchClient",
        "method": "delete_documents",
        "span_name": "azure_search.delete_documents",
    },
]


def _set_input_attributes(span, instance, method, args, kwargs):
    try:
        if hasattr(instance, "_index_name"):
            span.set_attribute("azure_search.index_name", instance._index_name)
        if hasattr(instance, "_endpoint"):
            span.set_attribute("server.address", str(instance._endpoint))
        if method == "search":
            search_text = kwargs.get("search_text") or (args[0] if args else None)
            if search_text:
                span.set_attribute("azure_search.search_text", str(search_text))
            top = kwargs.get("top")
            if top:
                span.set_attribute("azure_search.top", int(top))
            filter_expr = kwargs.get("filter")
            if filter_expr:
                span.set_attribute("azure_search.filter", str(filter_expr))
    except Exception as e:
        logger.debug("Failed to set input attributes: %s", e)


def _set_response_attributes(span, method, response):
    try:
        if method in ("upload_documents", "merge_documents",
                      "merge_or_upload_documents", "delete_documents"):
            if hasattr(response, "__iter__"):
                affected = 0
                succeeded = 0
                for r in response:
                    affected += 1
                    if getattr(r, "succeeded", False):
                        succeeded += 1
                span.set_attribute("azure_search.affected_documents", affected)
                span.set_attribute("azure_search.succeeded_documents", succeeded)
                return affected, succeeded
    except Exception as e:
        logger.debug("Failed to set response attributes: %s", e)
    return None, None


def _wrap(tracer, to_wrap):
    def wrapper(wrapped, instance, args, kwargs):
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        method = to_wrap.get("method")
        span_name = to_wrap.get("span_name")

        with tracer.start_as_current_span(
            span_name,
            kind=SpanKind.CLIENT,
            attributes={"db.system": "azure_search"},
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            if span.is_recording():
                _set_input_attributes(span, instance, method, args, kwargs)

            start_time = time.time()
            try:
                response = wrapped(*args, **kwargs)
            except Exception as e:
                span.set_attribute(ERROR_TYPE, e.__class__.__name__)
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            finally:
                duration = time.time() - start_time
                span.set_attribute("azure_search.duration", round(duration, 4))

            if span.is_recording():
                affected, succeeded = _set_response_attributes(span, method, response)
                if affected is not None and succeeded is not None and succeeded < affected:
                    span.set_status(Status(StatusCode.ERROR, f"{affected - succeeded} document(s) failed to index"))
                else:
                    span.set_status(Status(StatusCode.OK))

            return response

    return wrapper


class AzureSearchInstrumentor(BaseInstrumentor):
    """An instrumentor for Azure AI Search client library."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        for wrapped_method in WRAPPED_METHODS:
            wrap_function_wrapper(
                "azure.search.documents",
                f"{wrapped_method['object']}.{wrapped_method['method']}",
                _wrap(tracer, wrapped_method),
            )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            try:
                unwrap(
                    f"azure.search.documents.{wrapped_method['object']}",
                    wrapped_method["method"],
                )
            except Exception as e:
                logger.debug(
                    "Failed to unwrap %s.%s: %s",
                    wrapped_method["object"],
                    wrapped_method["method"],
                    e,
                )