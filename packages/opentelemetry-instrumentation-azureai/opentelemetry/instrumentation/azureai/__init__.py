"""OpenTelemetry Azure AI Search instrumentation"""

import logging
import time
from typing import Collection, Dict, Optional, Callable

from opentelemetry import context as context_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY, unwrap
from opentelemetry.metrics import Counter, Histogram, Meter, get_meter
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
    Meters,
    LLMRequestTypeValues,
)
from opentelemetry.trace import SpanKind, Tracer, get_tracer
from opentelemetry.trace.status import Status, StatusCode
from wrapt import wrap_function_wrapper

from .version import __version__
from .utils import (
    error_metrics_attributes,
    set_span_attribute,
    shared_metrics_attributes,
)

logger = logging.getLogger(__name__)

_instruments = ("azure-search-documents >= 11.4.0",)

WRAPPED_METHODS = [
    {
        "package": "azure.search.documents",
        "object": "SearchClient",
        "method": "search",
        "span_name": "azure.search",
    },
    {
        "package": "azure.search.documents",
        "object": "SearchClient",
        "method": "suggest",
        "span_name": "azure.search.suggest",
    },
    {
        "package": "azure.search.documents",
        "object": "SearchClient",
        "method": "autocomplete",
        "span_name": "azure.search.autocomplete",
    },
]

def _create_metrics(meter: Meter):
    """Create metrics collectors."""
    duration_histogram = meter.create_histogram(
        name=Meters.LLM_OPERATION_DURATION,
        unit="s",
        description="Azure AI Search operation duration",
    )

    result_counter = meter.create_counter(
        name=f"{Meters.LLM_AZURE_SEARCH_PREFIX}.results",
        unit="result",
        description="Number of results returned by search operation",
    )

    exception_counter = meter.create_counter(
        name=f"{Meters.LLM_AZURE_SEARCH_PREFIX}.exceptions",
        unit="exception",
        description="Number of exceptions occurred during search operations",
    )

    return duration_histogram, result_counter, exception_counter

def _with_search_telemetry_wrapper(func):
    """Helper for providing tracer for wrapper functions with metrics."""
    def _with_search_telemetry(
        tracer,
        duration_histogram,
        result_counter,
        exception_counter,
        to_wrap,
    ):
        def wrapper(wrapped, instance, args, kwargs):
            return func(
                tracer,
                duration_histogram,
                result_counter,
                exception_counter,
                to_wrap,
                wrapped,
                instance,
                args,
                kwargs,
            )
        return wrapper
    return _with_search_telemetry

@_with_search_telemetry_wrapper
def _wrap(
    tracer: Tracer,
    duration_histogram: Histogram,
    result_counter: Counter,
    exception_counter: Counter,
    to_wrap,
    wrapped,
    instance,
    args,
    kwargs,
):
    """Instruments and calls Azure Search methods."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "Azure AI Search",
            SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.SEARCH.value,
        },
    )

    if span.is_recording():
        # Record search parameters
        set_span_attribute(span, "azure.search.index_name", instance._index_name)
        if kwargs.get("search_text"):
            set_span_attribute(span, "azure.search.query", kwargs["search_text"])
        if kwargs.get("filter"):
            set_span_attribute(span, "azure.search.filter", kwargs["filter"])
        if kwargs.get("facets"):
            set_span_attribute(span, "azure.search.facets", str(kwargs["facets"]))
        if kwargs.get("top"):
            set_span_attribute(span, "azure.search.top", kwargs["top"])

    start_time = time.time()
    try:
        response = wrapped(*args, **kwargs)
    except Exception as e:
        end_time = time.time()
        attributes = error_metrics_attributes(e)

        if duration_histogram:
            duration = end_time - start_time
            duration_histogram.record(duration, attributes=attributes)

        if exception_counter:
            exception_counter.add(1, attributes=attributes)

        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.end()
        raise e

    end_time = time.time()

    try:
        metric_attributes = shared_metrics_attributes(response)

        if duration_histogram:
            duration = end_time - start_time
            duration_histogram.record(
                duration,
                attributes=metric_attributes,
            )

        if result_counter and hasattr(response, 'get_count'):
            result_count = response.get_count()
            if result_count is not None:
                result_counter.add(
                    result_count,
                    attributes=metric_attributes,
                )
                set_span_attribute(span, "azure.search.result_count", result_count)

    except Exception as ex:
        logger.warning(
            "Failed to set response attributes for Azure Search span, error: %s",
            str(ex),
        )

    if span.is_recording():
        span.set_status(Status(StatusCode.OK))
    span.end()
    return response

class AzureSearchInstrumentor(BaseInstrumentor):
    """An instrumentor for Azure AI Search client library."""

    def __init__(
        self,
        exception_logger: Optional[Callable] = None,
        get_common_metrics_attributes: Callable[[], Dict] = lambda: {},
    ):
        super().__init__()
        self._exception_logger = exception_logger
        self._get_common_metrics_attributes = get_common_metrics_attributes

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        meter_provider = kwargs.get("meter_provider")
        meter = get_meter(__name__, __version__, meter_provider)

        duration_histogram, result_counter, exception_counter = _create_metrics(meter)

        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")

            try:
                wrap_function_wrapper(
                    wrap_package,
                    f"{wrap_object}.{wrap_method}",
                    _wrap(
                        tracer,
                        duration_histogram,
                        result_counter,
                        exception_counter,
                        wrapped_method,
                    ),
                )
            except ModuleNotFoundError:
                pass

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            unwrap(
                f"{wrap_package}.{wrap_object}",
                wrapped_method.get("method"),
            )