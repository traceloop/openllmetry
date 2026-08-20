"""OpenTelemetry instrumentation for GNAP agent coordination."""

import importlib
import inspect
import logging
from typing import Collection

from opentelemetry import trace
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.trace import SpanKind
from opentelemetry.trace.status import Status, StatusCode

logger = logging.getLogger(__name__)
__version__ = "0.62.3"


def _value(value, *names):
    if isinstance(value, dict):
        for name in names:
            if name in value:
                return value[name]
    for name in names:
        result = getattr(value, name, None)
        if result is not None:
            return result
    return None


class GNAPInstrumentor(BaseInstrumentor):
    """Instrument GNAP board task lifecycle methods."""

    _methods = ("create_task", "claim_task", "complete_task")

    def instrumentation_dependencies(self) -> Collection[str]:
        return ("gnap",)

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = trace.get_tracer(__name__, __version__, tracer_provider)
        board_class = kwargs.get("board_class") or self._resolve_board_class()
        if board_class is None:
            logger.debug("GNAP board class was not found; instrumentation skipped")
            return
        self._board_class = board_class
        self._originals = {}
        for method_name in self._methods:
            original = getattr(board_class, method_name, None)
            if original is None:
                continue
            self._originals[method_name] = original
            setattr(board_class, method_name, self._wrap(tracer, method_name, original))

    def _uninstrument(self, **kwargs):
        for method_name, original in getattr(self, "_originals", {}).items():
            setattr(self._board_class, method_name, original)
        self._originals = {}

    @staticmethod
    def _resolve_board_class():
        for module_name, class_name in (("gnap", "GNAPBoard"), ("gnap.board", "GNAPBoard")):
            try:
                return getattr(importlib.import_module(module_name), class_name)
            except (ImportError, AttributeError):
                continue
        return None

    @staticmethod
    def _wrap(tracer, method_name, original):
        operation = method_name.removesuffix("_task")

        def wrapped(instance, *args, **kwargs):
            task = _value(args[0] if args else kwargs, "id", "task_id", "name")
            span = tracer.start_span(f"gnap.task.{operation}", kind=SpanKind.INTERNAL)
            span.set_attribute("gnap.operation", operation)
            if task is not None:
                span.set_attribute("gnap.task.id", str(task))
            agent = _value(instance, "agent_id", "agent") or _value(kwargs, "agent_id", "agent")
            if agent is not None:
                span.set_attribute("gnap.agent.id", str(agent))
            try:
                result = original(instance, *args, **kwargs)
                if inspect.isawaitable(result):
                    return GNAPInstrumentor._finish_async(span, result)
                span.set_attribute("gnap.operation.success", True)
                return GNAPInstrumentor._finish(span, result)
            except Exception as error:
                span.set_status(Status(StatusCode.ERROR, str(error)))
                span.record_exception(error)
                span.end()
                raise

        return wrapped

    @staticmethod
    def _finish(span, result):
        if result is not None and hasattr(result, "__len__") and not isinstance(result, (str, bytes)):
            span.set_attribute("gnap.result.size", len(result))
        span.end()
        return result

    @staticmethod
    async def _finish_async(span, result):
        try:
            value = await result
            span.set_attribute("gnap.operation.success", True)
            return GNAPInstrumentor._finish(span, value)
        except Exception as error:
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.record_exception(error)
            span.end()
            raise


__all__ = ["GNAPInstrumentor", "__version__"]
