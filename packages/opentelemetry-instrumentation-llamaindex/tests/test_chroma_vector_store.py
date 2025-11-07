from pathlib import Path

import chromadb
import pytest
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_rag_with_chroma(instrument_legacy, span_exporter):
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("quickstart")

    # define embedding function
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")

    # load documents
    current_dir = Path(__file__).parent.parent
    data_dir = current_dir.joinpath("data/paul_graham")
    documents = SimpleDirectoryReader(data_dir).load_data()

    # set up ChromaVectorStore and load in data
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model
    )
    engine = index.as_query_engine()
    engine.query("What did the author do growing up?")

    spans = span_exporter.get_finished_spans()
    assert {
        "RetrieverQueryEngine.workflow",
        "CompactAndRefine.task",
        "DefaultRefineProgram.task",
        "openai.chat",
        "RetrieverQueryEngine.task",
        "chroma.add",
        "chroma.query",
        "chroma.query.segment._query",
    }.issubset({span.name for span in spans})

    query_pipeline_span = next(
        span for span in spans if span.name == "RetrieverQueryEngine.workflow"
    )
    synthesize_span = next(
        span for span in spans if span.name == "CompactAndRefine.task"
    )
    llm_span = next(span for span in spans if span.name == "openai.chat")

    assert query_pipeline_span.parent is None
    assert synthesize_span.parent is not None
    assert llm_span.parent is not None

    assert llm_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-3.5-turbo"
    assert (
        llm_span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "gpt-3.5-turbo-0125"
    )
    assert llm_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"].startswith(
        "You are an expert Q&A system that is trusted around the world."
    )
    assert llm_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"] == (
        "The author worked on writing and programming before college."
    )
    assert llm_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 10
    assert llm_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 2070
    assert llm_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 2080
