import sys
import pytest
import replicate


@pytest.mark.skipif(sys.version_info < (3, 9), reason="requires python3.9")
@pytest.mark.vcr
def test_replicate_llama_stream(exporter):
    model_version = "meta/llama-2-70b-chat"
    for event in replicate.stream(
        model_version,
        input={
            "prompt": "tell me a joke about opentelemetry",
        },
    ):
        continue

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "replicate.stream",
    ]
