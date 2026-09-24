from unittest.mock import MagicMock

import anthropic
import pytest

from opentelemetry.instrumentation.anthropic.utils import set_span_attribute


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(anthropic.NOT_GIVEN, id="not-given"),
        pytest.param(anthropic.Omit(), id="omit"),
    ],
)
def test_set_span_attribute_skips_anthropic_sentinels(value):
    span = MagicMock()

    set_span_attribute(span, "gen_ai.request.temperature", value)

    span.set_attribute.assert_not_called()
