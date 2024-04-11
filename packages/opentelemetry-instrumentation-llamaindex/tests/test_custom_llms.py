import pytest
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)


@pytest.mark.vcr
def test_cpp_lmm(exporter):
    llm = LlamaCPP(
        model_path="data/llama-2-13b-chat.ggmlv3.q2_K.bin",
        temperature=0.1,
        max_new_tokens=256,
        context_window=3900,
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )
    response = llm.complete("Hello! Can you tell me a poem about cats and dogs?")
    assert response

    spans = exporter.get_finished_spans()

    llm_span = next(span for span in spans if span.name == "llama_cpp.complete")

    assert llm_span.attributes["llm.model"] == "LlamaCPP"
    assert llm_span.attributes["llm.request.model"] == "completion"
