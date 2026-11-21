import pytest
from llama_index.core.instrumentation import get_dispatcher
from opentelemetry import context as context_api
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY

dispatcher = get_dispatcher()


class ExplodingWorkflow:
    @dispatcher.span
    def run(self, query: str):
        raise ValueError("boom")


def test_failed_call_ends_span(instrument_legacy, span_exporter):
    """A call that raises must still end its OTel span.

    The dispatcher calls prepare_to_drop_span when the spanned call fails;
    the handler has to end the span and detach the context it attached in
    new_span, otherwise the failed call is never exported and the LLM
    content-suppression flag stays attached, untracing every following call.
    """
    with pytest.raises(ValueError, match="boom"):
        ExplodingWorkflow().run(query="hi")

    names = [span.name for span in span_exporter.get_finished_spans()]
    assert "ExplodingWorkflow.workflow" in names

    # The suppression context attached for the duration of the span must be
    # detached again, otherwise every following LLM call stays suppressed.
    assert context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY) is None
