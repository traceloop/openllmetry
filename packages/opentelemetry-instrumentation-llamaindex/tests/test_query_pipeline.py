import pytest
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    PromptTemplate,
)
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.llms.openai import OpenAI
from llama_index.postprocessor.cohere_rerank import CohereRerank
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.skip("Needs https://github.com/run-llama/llama_index/pull/14997")
@pytest.mark.vcr
def test_query_pipeline(exporter):
    docs = SimpleDirectoryReader("./data/paul_graham/").load_data()

    prompt_str = "Please generate a question about Paul Graham's life regarding the following topic {topic}"
    prompt_tmpl = PromptTemplate(prompt_str)
    llm = OpenAI(model="gpt-3.5-turbo")
    index = VectorStoreIndex.from_documents(docs)
    retriever = index.as_retriever(similarity_top_k=3)
    reranker = CohereRerank()
    summarizer = TreeSummarize(llm=llm)

    p = QueryPipeline()
    p.add_modules(
        {
            "llm": llm,
            "prompt_tmpl": prompt_tmpl,
            "retriever": retriever,
            "summarizer": summarizer,
            "reranker": reranker,
        }
    )
    p.add_link("prompt_tmpl", "llm")
    p.add_link("llm", "retriever")
    p.add_link("retriever", "reranker", dest_key="nodes")
    p.add_link("llm", "reranker", dest_key="query_str")
    p.add_link("reranker", "summarizer", dest_key="nodes")
    p.add_link("llm", "summarizer", dest_key="query_str")

    p.run(topic="YCombinator")

    spans = exporter.get_finished_spans()

    assert {
        "QueryPipeline.workflow",
        "BaseRetriever.task",
        "BaseSynthesizer.task",
        "CohereRerank.task",
        "LLM.task",
        "OpenAI.task",
        "TokenTextSplitter.task",
        "openai.chat",
        "openai.embeddings",
        "cohere.rerank",
    }.issubset({span.name for span in spans})

    query_pipeline_span = next(
        span for span in spans if span.name == "QueryPipeline.workflow"
    )
    retriever_span = next(span for span in spans if span.name == "BaseRetriever.task")
    reranker_span = next(span for span in spans if span.name == "CohereRerank.task")
    synthesizer_span = next(
        span for span in spans if span.name == "BaseSynthesizer.task"
    )
    llm_span_1, llm_span_2 = [span for span in spans if span.name == "OpenAI.task"]
    openai_span_1, openai_span_2 = [
        span for span in spans if span.name == "openai.chat"
    ]

    assert query_pipeline_span.parent is None
    assert reranker_span.parent is not None
    assert retriever_span.parent is not None
    assert synthesizer_span.parent is not None
    assert llm_span_1.parent is not None
    assert llm_span_2.parent is not None
    assert openai_span_1.parent.span_id == llm_span_1.context.span_id
    assert openai_span_2.parent.span_id == llm_span_2.context.span_id

    assert llm_span_1.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-3.5-turbo"
    assert (
        llm_span_1.attributes[SpanAttributes.LLM_RESPONSE_MODEL] == "gpt-3.5-turbo-0125"
    )
    assert llm_span_1.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == (
        "Please generate a question about Paul Graham's life regarding the following topic YCombinator"
    )
    assert llm_span_1.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.content"] == (
        "What role did Paul Graham play in the founding and development of YCombinator, and "
        "how has his involvement shaped the trajectory of the company?"
    )

    assert llm_span_2.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-3.5-turbo"
    assert (
        llm_span_2.attributes[SpanAttributes.LLM_RESPONSE_MODEL] == "gpt-3.5-turbo-0125"
    )
    assert llm_span_2.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"].startswith(
        "You are an expert Q&A system that is trusted around the world."
    )
    assert llm_span_2.attributes[
        f"{SpanAttributes.LLM_COMPLETIONS}.content"
    ].startswith(
        "Paul Graham played a pivotal role in the founding and development of Y Combinator."
    )
