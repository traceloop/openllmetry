from unittest.mock import patch

import pytest
from opentelemetry import trace


def test_provider_call_runs_with_the_together_span_as_current(instrument_legacy, caller, span_exporter):
    """Spans created during the provider call (e.g. HTTP client spans) must nest
    under the together span, so the call must run with it as the current span."""
    current_spans = []

    def request(*args, **kwargs):
        current_spans.append(trace.get_current_span())
        raise Exception("stop before the network")

    with patch(caller.request_patch_target(), side_effect=request):
        with pytest.raises(Exception, match="stop before the network"):
            caller.chat(model="m", messages=[{"role": "user", "content": "hi"}])

    [together_span] = span_exporter.get_finished_spans()
    assert len(current_spans) == 1
    assert current_spans[0].get_span_context() == together_span.get_span_context()
    # The span isn't left as the current span after the call.
    assert trace.get_current_span() is trace.INVALID_SPAN
