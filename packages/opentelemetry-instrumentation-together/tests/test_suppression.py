from unittest.mock import patch

import pytest
from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics as GenAIMetrics
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY

from .helpers import _data_points


@pytest.mark.parametrize(
    "suppress_key",
    [_SUPPRESS_INSTRUMENTATION_KEY, SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY],
    ids=["suppress_instrumentation", "suppress_language_model_instrumentation"],
)
def test_suppressed_calls_pass_through_without_telemetry(
    instrument_legacy, caller, span_exporter, reader, suppress_key
):
    token = context_api.attach(context_api.set_value(suppress_key, True))
    try:
        with patch(caller.request_patch_target(), side_effect=Exception("stop before the network")) as request:
            with pytest.raises(Exception, match="stop before the network"):
                caller.chat(model="m", messages=[{"role": "user", "content": "hi"}])
    finally:
        context_api.detach(token)

    # The SDK call still ran, but nothing was recorded for it.
    assert request.call_count == 1
    assert not span_exporter.get_finished_spans()
    assert not _data_points(reader, GenAIMetrics.GEN_AI_CLIENT_OPERATION_DURATION)
