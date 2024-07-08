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

    assert (
        {
            "retrieve.task",
            "synthesize.task",
            "cohere.rerank",
            "llama_index_query_pipeline.workflow",
            "openai.chat",
        }
        == {
            "openai.embeddings",
            "Embedding.llamaindex.task",
            "Retriever.llamaindex.workflow",
            "LLM.llamaindex.task",
            "CohereRerank.llamaindex.workflow",
            "Embedding.llamaindex.workflow",
            "cohere.rerank",
            "NodeParser.llamaindex.workflow",
            "Synthesizer.llamaindex.workflow",
            "OpenAI.llamaindex.workflow",
            "NodePostprocessor.llamaindex.workflow",
            "SentenceSplitter.llamaindex.task",
            "TokenTextSplitter.llamaindex.task",
            "LLM.llamaindex.workflow",
            "openai.chat",
        }
        == set([span.name for span in spans])
    )

    query_pipeline_span = next(
        span for span in spans if span.name == "llama_index_query_pipeline.workflow"
    )
    retriever_span = next(span for span in spans if span.name == "retrieve.task")
    reranker_span = next(span for span in spans if span.name == "cohere.rerank")

    assert retriever_span.parent.span_id == query_pipeline_span.context.span_id
    assert reranker_span.parent.span_id == query_pipeline_span.context.span_id
