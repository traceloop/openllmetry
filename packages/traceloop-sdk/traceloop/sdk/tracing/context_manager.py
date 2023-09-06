from contextlib import contextmanager, asynccontextmanager

from traceloop.sdk.tracing.tracer import Tracing


@contextmanager
def get_tracer(flush_on_exit: bool = False):
    try:
        yield Tracing.get_tracer()
    finally:
        if flush_on_exit:
            Tracing.flush()


@asynccontextmanager
async def get_async_tracer(flush_on_exit: bool = False):
    try:
        yield Tracing.get_tracer()
    finally:
        if flush_on_exit:
            Tracing.flush()
