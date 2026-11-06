from anthropic.lib.streaming._beta_messages import (
    BetaAsyncMessageStreamManager,
    BetaMessageStreamManager,
)
from anthropic.lib.streaming._messages import (
    AsyncMessageStreamManager,
    MessageStreamManager,
)

from opentelemetry.instrumentation.anthropic import (
    is_async_stream_manager,
    is_stream_manager,
)


def _make(cls):
    # The manager classes take a client-bound request callable; we only need an
    # instance for type/name detection, so bypass __init__.
    return cls.__new__(cls)


def test_is_stream_manager_recognizes_all_managers():
    for cls in (
        MessageStreamManager,
        AsyncMessageStreamManager,
        BetaMessageStreamManager,
        BetaAsyncMessageStreamManager,
    ):
        assert is_stream_manager(_make(cls)), cls.__name__


def test_is_stream_manager_rejects_plain_objects():
    assert not is_stream_manager(object())


def test_is_async_stream_manager_matches_async_variants_only():
    assert is_async_stream_manager(_make(AsyncMessageStreamManager))
    # Regression for #4388: beta async streams must route to the async wrapper.
    assert is_async_stream_manager(_make(BetaAsyncMessageStreamManager))

    assert not is_async_stream_manager(_make(MessageStreamManager))
    assert not is_async_stream_manager(_make(BetaMessageStreamManager))
