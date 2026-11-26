"""OpenTelemetry Pinecone instrumentation"""

import logging
import time
import importlib.util
import pinecone
from typing import Collection
from wrapt import FunctionWrapper, wrap_function_wrapper

from opentelemetry import context as context_api
from opentelemetry.metrics import get_meter
from opentelemetry.trace import get_tracer, SpanKind
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)
from opentelemetry.instrumentation.pinecone.config import Config
from opentelemetry.instrumentation.pinecone.utils import (
    dont_throw,
    is_metrics_enabled,
    set_span_attribute,
)
from opentelemetry.instrumentation.pinecone.version import __version__
from opentelemetry.instrumentation.pinecone.query_handlers import (
    set_query_input_attributes,
    set_query_response,
)
from opentelemetry.semconv.trace import SpanAttributes
from opentelemetry.semconv_ai import Meters, SpanAttributes as AISpanAttributes

logger = logging.getLogger(__name__)

_instruments = ("pinecone >= 5.1.0, <11",)

WRAPPED_METHODS = [
    {
        "object": "GRPCIndex",
        "method": "query",
        "span_name": "pinecone.query",
    },
    {
        "object": "GRPCIndex",
        "method": "upsert",
        "span_name": "pinecone.upsert",
    },
    {
        "object": "GRPCIndex",
        "method": "delete",
        "span_name": "pinecone.delete",
    },
    {
        "object": "Index",
        "method": "query",
        "span_name": "pinecone.query",
    },
    {
        "object": "Index",
        "method": "upsert",
        "span_name": "pinecone.upsert",
    },
    {
        "object": "Index",
        "method": "delete",
        "span_name": "pinecone.delete",
    },
]


@dont_throw
def _set_input_attributes(span, instance, kwargs):
    set_span_attribute(span, SpanAttributes.SERVER_ADDRESS, instance._config.host)


@dont_throw
def _set_response_attributes(
    span, read_units_metric, write_units_metric, shared_attributes, response
):
    if response.get("usage"):
        read_units = response.get("usage").get("read_units") or 0
        write_units = response.get("usage").get("write_units") or 0

        read_units_metric.add(read_units, shared_attributes)
        span.set_attribute(AISpanAttributes.PINECONE_USAGE_READ_UNITS, read_units)

        write_units_metric.add(write_units, shared_attributes)
        span.set_attribute(AISpanAttributes.PINECONE_USAGE_WRITE_UNITS, write_units)


def _with_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(
        tracer,
        query_duration_metric,
        read_units_metric,
        write_units_metric,
        scores_metric,
        to_wrap,
    ):
        def wrapper(wrapped, instance, args, kwargs):
            return func(
                tracer,
                query_duration_metric,
                read_units_metric,
                write_units_metric,
                scores_metric,
                to_wrap,
                wrapped,
                instance,
                args,
                kwargs,
            )

        return wrapper

    return _with_tracer


@_with_wrapper
def _wrap(
    tracer,
    query_duration_metric,
    read_units_metric,
    write_units_metric,
    scores_metric,
    to_wrap,
    wrapped,
    instance,
    args,
    kwargs,
):
    """Instruments and calls every function defined in TO_WRAP."""
    return _execute_wrapped(
        tracer,
        query_duration_metric,
        read_units_metric,
        write_units_metric,
        scores_metric,
        to_wrap,
        wrapped,
        instance,
        args,
        kwargs,
    )


def _execute_wrapped(
    tracer,
    query_duration_metric,
    read_units_metric,
    write_units_metric,
    scores_metric,
    to_wrap,
    wrapped,
    instance,
    args,
    kwargs,
):
    """Shared span logic for the class-level wrapper and the per-instance wrapper."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    with tracer.start_as_current_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            AISpanAttributes.VECTOR_DB_VENDOR: "Pinecone",
        },
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        if span.is_recording():
            _set_input_attributes(span, instance, kwargs)
            if to_wrap.get("method") == "query":
                set_query_input_attributes(span, kwargs)

        shared_attributes = {}
        if hasattr(instance, "_config"):
            shared_attributes["server.address"] = instance._config.host

        start_time = time.time()
        try:
            response = wrapped(*args, **kwargs)
        except Exception as e:
            span.set_attribute(ERROR_TYPE, e.__class__.__name__)
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        end_time = time.time()

        duration = end_time - start_time
        if duration > 0 and query_duration_metric and to_wrap.get("method") == "query":
            query_duration_metric.record(duration, shared_attributes)

        if response:
            if span.is_recording():
                if to_wrap.get("method") == "query":
                    set_query_response(span, scores_metric, shared_attributes, response)

                _set_response_attributes(
                    span,
                    read_units_metric,
                    write_units_metric,
                    shared_attributes,
                    response,
                )

                span.set_status(Status(StatusCode.OK))

        return response


def _wrap_instance_methods(
    instance,
    tracer,
    query_duration_metric,
    read_units_metric,
    write_units_metric,
    scores_metric,
):
    """Wrap per-instance methods with the same span logic as the class wrapper.

    Since pinecone 10, ``Index.__init__`` rebinds ``query``/``upsert`` (and
    potentially other methods) on the instance via
    ``install_async_req_support``, so the class-level wrapping applied by
    ``_instrument`` is shadowed and never runs for those methods. Wrapping
    the instance attributes after construction restores the spans.

    Skipped when the class attribute is already wrapped: in that case the
    instance delegates to the traced canonical method pinecone stashes as
    ``_canonical_<name>``, so wrapping the instance attribute as well would
    emit every span twice.
    """
    for wrapped_method in WRAPPED_METHODS:
        method = wrapped_method.get("method")
        if method not in instance.__dict__:
            continue
        original = instance.__dict__[method]
        if getattr(original, "_otel_pinecone_traced", False):
            continue
        cls_attr = None
        for base in type(instance).__mro__:
            if method in base.__dict__:
                cls_attr = base.__dict__[method]
                break
        # NB: getattr() would return a BoundFunctionWrapper here, so the
        # raw class-dict entry is inspected instead.
        if isinstance(cls_attr, FunctionWrapper):
            continue

        def traced(
            *args,
            _original=original,
            _instance=instance,
            _wrapped_method=wrapped_method,
            **kwargs,
        ):
            return _execute_wrapped(
                tracer,
                query_duration_metric,
                read_units_metric,
                write_units_metric,
                scores_metric,
                _wrapped_method,
                _original,
                _instance,
                args,
                kwargs,
            )

        traced._otel_pinecone_traced = True
        setattr(instance, method, traced)


def _wrap_init(
    tracer,
    query_duration_metric,
    read_units_metric,
    write_units_metric,
    scores_metric,
):
    """Return a wrapper for ``Index.__init__`` that traces instance-bound methods."""

    def wrapper(wrapped, instance, args, kwargs):
        result = wrapped(*args, **kwargs)
        try:
            _wrap_instance_methods(
                instance,
                tracer,
                query_duration_metric,
                read_units_metric,
                write_units_metric,
                scores_metric,
            )
        except Exception:  # noqa: BLE001 - never break client construction
            logger.debug("Failed to wrap pinecone instance methods", exc_info=True)
        return result

    return wrapper


def _resolve_index_class(class_name):
    """Return the Index-like class, trying the known module paths in order."""
    for module_name in ("pinecone.db_data.index", "pinecone.index", "pinecone"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        cls = getattr(module, class_name, None)
        if cls is not None:
            return cls
    return None


def _wrap_class_init(
    class_name,
    tracer,
    query_duration_metric,
    read_units_metric,
    write_units_metric,
    scores_metric,
):
    """Best-effort ``__init__`` wrapping for one Index-like class."""
    cls = _resolve_index_class(class_name)
    if cls is None or "__init__" not in cls.__dict__:
        return
    try:
        wrap_function_wrapper(
            cls,
            "__init__",
            _wrap_init(
                tracer,
                query_duration_metric,
                read_units_metric,
                write_units_metric,
                scores_metric,
            ),
        )
    except (ImportError, AttributeError):
        pass


def _unwrap_class_init(class_name):
    """Best-effort ``__init__`` unwrapping for one Index-like class."""
    cls = _resolve_index_class(class_name)
    if cls is None:
        return
    try:
        unwrap(cls, "__init__")
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Failed to unwrap {class_name}.__init__: {e}")


class PineconeInstrumentor(BaseInstrumentor):
    """An instrumentor for Pinecone's client library."""

    def __init__(self, exception_logger=None):
        super().__init__()
        Config.exception_logger = exception_logger

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        if is_metrics_enabled():
            meter_provider = kwargs.get("meter_provider")
            meter = get_meter(__name__, __version__, meter_provider)

            query_duration_metric = meter.create_histogram(
                Meters.PINECONE_DB_QUERY_DURATION,
                "s",
                "Duration of query operations to Pinecone",
            )
            read_units_metric = meter.create_counter(
                Meters.PINECONE_DB_USAGE_READ_UNITS,
                "unit",
                "Number of read units consumed in serverless calls",
            )
            write_units_metric = meter.create_counter(
                Meters.PINECONE_DB_USAGE_WRITE_UNITS,
                "unit",
                "Number of write units consumed in serverless calls",
            )
            scores_metric = meter.create_histogram(
                Meters.PINECONE_DB_QUERY_SCORES,
                "score",
                "Scores returned from Pinecone calls",
            )

        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)
        for wrapped_method in WRAPPED_METHODS:
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")

            if wrap_object == "Index":
                try:
                    if importlib.util.find_spec("pinecone.db_data.index") is not None:
                        try:
                            wrap_function_wrapper(
                                "pinecone.db_data.index",
                                f"{wrap_object}.{wrap_method}",
                                _wrap(
                                    tracer,
                                    query_duration_metric,
                                    read_units_metric,
                                    write_units_metric,
                                    scores_metric,
                                    wrapped_method,
                                ),
                            )
                            continue
                        except (ImportError, AttributeError):
                            pass
                except (ModuleNotFoundError, ImportError):
                    pass

                if getattr(pinecone, wrap_object, None):
                    wrap_function_wrapper(
                        "pinecone",
                        f"{wrap_object}.{wrap_method}",
                        _wrap(
                            tracer,
                            query_duration_metric,
                            read_units_metric,
                            write_units_metric,
                            scores_metric,
                            wrapped_method,
                        ),
                    )
            elif wrap_object == "GRPCIndex":
                try:
                    if importlib.util.find_spec("pinecone.db_data.index") is not None:
                        try:
                            wrap_function_wrapper(
                                "pinecone.db_data.index",
                                f"{wrap_object}.{wrap_method}",
                                _wrap(
                                    tracer,
                                    query_duration_metric,
                                    read_units_metric,
                                    write_units_metric,
                                    scores_metric,
                                    wrapped_method,
                                ),
                            )
                            continue
                        except (ImportError, AttributeError):
                            pass
                except (ModuleNotFoundError, ImportError):
                    pass

                if getattr(pinecone, wrap_object, None):
                    wrap_function_wrapper(
                        "pinecone",
                        f"{wrap_object}.{wrap_method}",
                        _wrap(
                            tracer,
                            query_duration_metric,
                            read_units_metric,
                            write_units_metric,
                            scores_metric,
                            wrapped_method,
                        ),
                    )

        # Since pinecone 10, Index.__init__ rebinds query/upsert on the
        # instance, shadowing the class-level wrapping above. Wrap __init__
        # so instances created after instrumentation get their instance-bound
        # methods traced as well (see _wrap_instance_methods).
        for class_name in ("Index", "GRPCIndex"):
            _wrap_class_init(
                class_name,
                tracer,
                query_duration_metric,
                read_units_metric,
                write_units_metric,
                scores_metric,
            )

    def _uninstrument(self, **kwargs):
        for class_name in ("Index", "GRPCIndex"):
            _unwrap_class_init(class_name)

        for wrapped_method in WRAPPED_METHODS:
            wrap_object = wrapped_method.get("object")
            wrap_method_name = wrapped_method.get("method")

            if wrap_object == "Index":
                try:
                    unwrap("pinecone.db_data.index.Index", wrap_method_name)
                except Exception as e:
                    logger.debug(
                        f"Failed to unwrap pinecone.db_data.index.Index.{wrap_method_name}: {e}"
                    )

            if wrap_object == "GRPCIndex":
                try:
                    unwrap("pinecone.db_data.index.GRPCIndex", wrap_method_name)
                except Exception as e:
                    logger.debug(
                        f"Failed to unwrap pinecone.db_data.index.GRPCIndex.{wrap_method_name}: {e}"
                    )

            try:
                unwrap(f"pinecone.{wrap_object}", wrap_method_name)
            except Exception as e:
                logger.debug(
                    f"Failed to unwrap pinecone.{wrap_object}.{wrap_method_name}: {e}"
                )
