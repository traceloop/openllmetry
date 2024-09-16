import pytest
from opentelemetry.semconv_ai import SpanAttributes

import json


@pytest.mark.vcr()
def test_sagemaker_completion_string_content(exporter, smrt):
    endpoint_name = "my-llama2-endpoint"
    prompt = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure
that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not
correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

There's a llama in my garden  What should I do? [/INST]"""
    # Create request body.
    body = json.dumps(
        {
            "inputs": prompt,
            "parameters": {"temperature": 0.1, "top_p": 0.9, "max_new_tokens": 128},
        }
    )

    smrt.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=body,
        ContentType="application/json",
    )

    spans = exporter.get_finished_spans()

    meta_span = spans[0]
    assert meta_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == endpoint_name
    assert meta_span.attributes[SpanAttributes.LLM_SAGEMAKER_REQUEST] == body
