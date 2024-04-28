from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY


from opentelemetry.instrumentation.langchain.utils import (
    _with_tracer_wrapper,
)
from opentelemetry.instrumentation.langchain.callbacks.span import (
    SyncSpanCallbackHandler,
    AsyncSpanCallbackHandler
)

@_with_tracer_wrapper
def init_wrapper(tracer, module, wrapped, instance, args, kwargs):
    """Instruments and injects CallbackHandler to init function."""
    span_name = module.get("span_name", None)
    kind_name = module.get("kind", None)
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        print("SUPPRESS_INSTRUMENTATION_KEY", _SUPPRESS_INSTRUMENTATION_KEY)
        return wrapped(*args, **kwargs)
    # TODO: Add a logic to find if AsyncCallbackHandler should be inject or SyncCallbackHandler?
    # AsyncSpanCallbackHandler(tracer, span_name, kind_name)
    kwargs["callbacks"] = [SyncSpanCallbackHandler(tracer, span_name, kind_name)]
    return wrapped(*args, **kwargs)
