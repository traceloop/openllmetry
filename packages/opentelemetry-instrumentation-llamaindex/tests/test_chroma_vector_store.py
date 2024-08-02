import pytest
import chromadb

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_rag_with_chroma(exporter):
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("quickstart")

    # define embedding function
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")

    # load documents
    documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

    # set up ChromaVectorStore and load in data
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    service_context = ServiceContext.from_defaults(embed_model=embed_model)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, service_context=service_context
    )
    engine = index.as_query_engine()
    engine.query("What did the author do growing up?")

    spans = exporter.get_finished_spans()
    assert {
        "BaseQueryEngine.workflow",
        "BaseSynthesizer.task",
        "LLM.task",
        "OpenAI.task",
        "RetrieverQueryEngine.task",
        "openai.chat",
        "openai.embeddings",
        "chroma.add",
        "chroma.query",
        "chroma.query.segment._query",
    }.issubset({span.name for span in spans})

    query_pipeline_span = next(
        span for span in spans if span.name == "BaseQueryEngine.workflow"
    )
    synthesize_span = next(
        span for span in spans if span.name == "BaseSynthesizer.task"
    )
    llm_span = next(span for span in spans if span.name == "OpenAI.task")
    openai_chat_span = next(span for span in spans if span.name == "openai.chat")

    assert query_pipeline_span.parent is None
    assert synthesize_span.parent is not None
    assert llm_span.parent is not None
    assert openai_chat_span.parent.span_id == llm_span.context.span_id

    assert llm_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-3.5-turbo"
    assert (
        llm_span.attributes[SpanAttributes.LLM_RESPONSE_MODEL] == "gpt-3.5-turbo-0125"
    )
    assert llm_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"].startswith(
        "You are an expert Q&A system that is trusted around the world."
    )
    assert llm_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.content"] == (
        "The author worked on writing and programming before college."
    )
