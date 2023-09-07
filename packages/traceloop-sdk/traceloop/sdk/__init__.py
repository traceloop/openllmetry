from typing import Optional
from traceloop.sdk.tracing import Tracing


class Traceloop:
    @staticmethod
    def init(app_name: Optional[str] = None) -> None:
        Tracing.init(app_name=app_name)
