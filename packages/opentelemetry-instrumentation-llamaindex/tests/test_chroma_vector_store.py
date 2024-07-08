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
        "SentenceSplitter.llamaindex.task",
        "NodeParser.llamaindex.workflow",
        "openai.embeddings",
        "Embedding.llamaindex.workflow",
        "chroma.add",
        "openai.embeddings",
        "Embedding.llamaindex.task",
        "chroma.query.segment._query",
        "chroma.query",
        "Retriever.llamaindex.task",
        "TokenTextSplitter.llamaindex.task",
        "openai.chat",
        "LLM.llamaindex.task",
        "Refine.llamaindex.task",
        "Synthesizer.llamaindex.task",
        "QueryEngine.llamaindex.workflow",
    } == set([span.name for span in spans])

    query_engine_span = next(
        span for span in spans if span.name == "QueryEngine.llamaindex.workflow"
    )
    retriever_span = next(
        span for span in spans if span.name == "Retriever.llamaindex.task"
    )
    synthesizer_span = next(
        span for span in spans if span.name == "Synthesizer.llamaindex.task"
    )

    assert retriever_span.parent.span_id == query_engine_span.context.span_id
    assert synthesizer_span.parent.span_id == query_engine_span.context.span_id
