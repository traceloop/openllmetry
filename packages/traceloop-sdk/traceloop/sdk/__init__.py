from typing import Optional
from traceloop.sdk.tracing.tracing import TracerWrapper, set_correlation_id


class Traceloop:
    __tracer_wrapper: TracerWrapper

    @staticmethod
    def init(app_name: Optional[str] = None) -> None:
        Traceloop.__tracer_wrapper = TracerWrapper()

    @staticmethod
    def set_correlation_id(correlation_id: str) -> None:
        set_correlation_id(correlation_id)
