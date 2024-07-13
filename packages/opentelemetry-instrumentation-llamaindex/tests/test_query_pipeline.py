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
        "query.llama_index.workflow",
        "chunking.llama_index.task",
        "embedding.llama_index.task",
        "llm.llama_index.task",
        "reranking.llama_index.task",
        "retrieve.llama_index.task",
        "synthesize.llama_index.task",
        "templating.llama_index.task",
    }.issubset({span.name for span in spans})

    query_pipeline_span = next(
        span for span in spans if span.name == "query.llama_index.workflow"
    )
    llm_span = next(span for span in spans if span.name == "llm.llama_index.task")
    reranker_span = next(span for span in spans if span.name == "reranking.llama_index.task")
    retriever_span = next(span for span in spans if span.name == "retrieve.llama_index.task")

    assert llm_span.parent.span_id == query_pipeline_span.context.span_id
    assert reranker_span.parent.span_id == query_pipeline_span.context.span_id
    assert retriever_span.parent.span_id == query_pipeline_span.context.span_id
