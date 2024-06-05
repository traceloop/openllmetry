"""OpenTelemetry Redis instrumentation"""

from opentelemetry.instrumentation.redis.config import Config
from opentelemetry.instrumentation.redis.version import __version__
from opentelemetry.instrumentation.redis.wrapper import _wrap
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer

import logging
import redis
import redis.commands.search
import redis.commands.json
from typing import Collection
from wrapt import wrap_function_wrapper


logger = logging.getLogger(__name__)

_instruments = ("redis >= 4.6.0",)

# The ping and create_index methods are not shown on the Web UI due to filtering
WRAPPED_METHODS = [
    {
        "package": redis,
        "object": "Redis",
        "method": "ping",
        "span_name": "redis.ping",
    },
    # {
    #     "package": redis,
    #     "object": "Redis",
    #     "method": "hset",
    #     "span_name": "redis.hset",
    # },
    {
        "package": redis.commands.search,
        "object": "Search",
        "method": "create_index",
        "span_name": "redis.create_index",
    },
    {
        "package": redis.commands.search,
        "object": "Search",
        "method": "search",
        "span_name": "redis.search",
    },
    {
        "package": redis.commands.search,
        "object": "Search",
        "method": "aggregate",
        "span_name": "redis.aggregate",
    },
    # {
    #     "package": redis.commands.json,
    #     "object": "JSON",
    #     "method": "set",
    #     "span_name": "redis.json.set",
    # },

]

class RedisInstrumentor(BaseInstrumentor):
    """An instrumentor for Redis's client library."""

    def __init__(self, exception_logger=None):
        super().__init__()
        Config.exception_logger = exception_logger

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            if getattr(wrap_package, wrap_object, None):
                wrap_function_wrapper(
                    wrap_package,
                    f"{wrap_object}.{wrap_method}",
                    _wrap(tracer, wrapped_method),
                )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            
            wrapped = getattr(wrap_package, wrap_object, None)
            if wrapped:
                unwrap(wrapped, wrapped_method.get("method"))
