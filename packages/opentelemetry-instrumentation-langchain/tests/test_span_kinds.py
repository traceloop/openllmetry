"""Test new span kinds functionality."""

import pytest
from opentelemetry.semconv_ai import SpanAttributes, TraceloopSpanKindValues
from opentelemetry.instrumentation.langchain.callback_handler import TraceloopCallbackHandler
from uuid import uuid4


class TestSpanKinds:
    """Test span kind detection and creation."""

    @pytest.fixture
    def callback_handler(self, span_exporter, tracer_provider, meter_provider):
        from opentelemetry.trace import get_tracer
        from opentelemetry.metrics import get_meter
        from opentelemetry.semconv_ai import Meters

        tracer = get_tracer(__name__, tracer_provider=tracer_provider)
        meter = get_meter(__name__, meter_provider=meter_provider)

        duration_histogram = meter.create_histogram(
            name=Meters.LLM_OPERATION_DURATION,
            unit="s",
            description="GenAI operation duration",
        )
        token_histogram = meter.create_histogram(
            name=Meters.LLM_TOKEN_USAGE,
            unit="token",
            description="Measures number of input and output tokens used",
        )

        return TraceloopCallbackHandler(tracer, duration_histogram, token_histogram)

    def test_determine_llm_span_kind_embedding(self, callback_handler):
        """Test detection of embedding models."""
        serialized = {
            "id": ["langchain", "embeddings", "openai", "OpenAIEmbeddings"],
            "name": "OpenAIEmbeddings"
        }

        kind = callback_handler._determine_llm_span_kind(serialized)
        assert kind == TraceloopSpanKindValues.EMBEDDING

    def test_determine_llm_span_kind_generation(self, callback_handler):
        """Test detection of generation models."""
        serialized = {
            "id": ["langchain", "llms", "openai", "OpenAI"],
            "name": "OpenAI"
        }

        kind = callback_handler._determine_llm_span_kind(serialized)
        assert kind == TraceloopSpanKindValues.GENERATION

    def test_determine_llm_span_kind_no_serialized(self, callback_handler):
        """Test default behavior when no serialized data."""
        kind = callback_handler._determine_llm_span_kind(None)
        assert kind == TraceloopSpanKindValues.GENERATION

    def test_retrieval_qa_chain(self, callback_handler):
        """Test that RetrievalQA chain is classified as TASK"""
        serialized = {
            "id": ["langchain", "chains", "retrieval_qa", "base", "RetrievalQA"],
            "name": "RetrievalQA"
        }

        kind = callback_handler._determine_chain_span_kind(serialized, "RetrievalQA")
        assert kind == TraceloopSpanKindValues.TASK

    def test_conversational_retrieval_chain(self, callback_handler):
        """Test that ConversationalRetrievalChain is classified as TASK"""
        serialized = {
            "id": ["langchain", "chains", "conversational_retrieval", "base", "ConversationalRetrievalChain"],
            "name": "ConversationalRetrievalChain"
        }

        kind = callback_handler._determine_chain_span_kind(serialized, "ConversationalRetrievalChain")
        assert kind == TraceloopSpanKindValues.TASK

    def test_reranker_by_class(self, callback_handler):
        """Test detection of reranker by class name."""
        serialized = {
            "id": ["langchain", "retrievers", "document_compressors", "LLMChainExtractor"],
            "name": "DocumentReranker"
        }

        kind = callback_handler._determine_chain_span_kind(serialized, "reranker")
        assert kind == TraceloopSpanKindValues.RERANKER

    def test_reranker_by_name(self, callback_handler):
        """Test detection of reranker by name."""
        serialized = {
            "id": ["langchain", "chains", "base", "Chain"],
            "name": "Chain"
        }

        kind = callback_handler._determine_chain_span_kind(serialized, "document_reranker")
        assert kind == TraceloopSpanKindValues.RERANKER

    def test_tool_by_name(self, callback_handler):
        """Test tool detection by name in chain callbacks."""
        serialized = {
            "id": ["langchain_core", "runnables", "base", "RunnableLambda"],
            "name": "RunnableLambda"
        }

        # Test name-based tool detection
        kind = callback_handler._determine_chain_span_kind(serialized, "add_numbers_tool")
        assert kind == TraceloopSpanKindValues.TOOL

        kind = callback_handler._determine_chain_span_kind(serialized, "calculator_function")
        assert kind == TraceloopSpanKindValues.TOOL

    def test_tool_by_tags(self, callback_handler):
        """Test tool detection by tags."""
        serialized = {
            "id": ["langchain_core", "runnables", "base", "RunnableLambda"],
            "name": "RunnableLambda"
        }

        kind = callback_handler._determine_chain_span_kind(serialized, "runnable", tags=["tool", "calculator"])
        assert kind == TraceloopSpanKindValues.TOOL

    def test_embedding_by_name(self, callback_handler):
        """Test embedding detection by name in chain callbacks."""
        serialized = {
            "id": ["langchain_core", "runnables", "base", "RunnableLambda"],
            "name": "RunnableLambda"
        }

        # Test name-based detection
        kind = callback_handler._determine_chain_span_kind(serialized, "OpenAIEmbeddings")
        assert kind == TraceloopSpanKindValues.EMBEDDING

        kind = callback_handler._determine_chain_span_kind(serialized, "document_embedder")
        assert kind == TraceloopSpanKindValues.EMBEDDING

    def test_agent_executor(self, callback_handler):
        """Test that AgentExecutor is classified as AGENT (real LangChain component)."""
        serialized = {
            "id": ["langchain", "agents", "agent", "AgentExecutor"],
            "name": "AgentExecutor"
        }

        kind = callback_handler._determine_chain_span_kind(serialized, "AgentExecutor")
        assert kind == TraceloopSpanKindValues.AGENT

    def test_default_task(self, callback_handler):
        """Test default behavior returns TASK."""
        serialized = {
            "id": ["langchain", "chains", "llm", "LLMChain"],
            "name": "LLMChain"
        }

        kind = callback_handler._determine_chain_span_kind(serialized, "llm_chain")
        assert kind == TraceloopSpanKindValues.TASK

    def test_workflow_span(self, callback_handler, span_exporter):
        """Test workflow span kind detection."""
        run_id = uuid4()
        serialized = {
            "id": ["langchain", "chains", "sequential", "SequentialChain"],
            "name": "SequentialChain"
        }

        callback_handler.on_chain_start(
            serialized=serialized,
            inputs={"input": "test"},
            run_id=run_id,
            parent_run_id=None
        )

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 0  # Span not finished yet

        span_holder = callback_handler.spans[run_id]
        assert span_holder.span.name == "SequentialChain.workflow"
        assert span_holder.span.attributes[SpanAttributes.TRACELOOP_SPAN_KIND] == "workflow"

    def test_agent_executor_in_workflow(self, callback_handler, span_exporter):
        """Test that AgentExecutor creates agent spans when used as child chain."""
        parent_run_id = uuid4()
        run_id = uuid4()

        callback_handler.on_chain_start(
            serialized={"id": ["langchain", "workflows", "rag"], "name": "RAGWorkflow"},
            inputs={},
            run_id=parent_run_id,
            parent_run_id=None
        )

        # AgentExecutor as child component
        serialized = {
            "id": ["langchain", "agents", "agent", "AgentExecutor"],
            "name": "AgentExecutor"
        }

        callback_handler.on_chain_start(
            serialized=serialized,
            inputs={"input": "test query"},
            run_id=run_id,
            parent_run_id=parent_run_id
        )

        span_holder = callback_handler.spans[run_id]
        assert span_holder.span.name == "AgentExecutor.agent"
        assert span_holder.span.attributes[SpanAttributes.TRACELOOP_SPAN_KIND] == "agent"

    def test_embedding_model(self, callback_handler, span_exporter):
        """Test embedding span kind detection in LLM."""
        run_id = uuid4()
        serialized = {
            "id": ["langchain", "embeddings", "openai", "OpenAIEmbeddings"],
            "name": "OpenAIEmbeddings"
        }

        callback_handler.on_llm_start(
            serialized=serialized,
            prompts=["test prompt"],
            run_id=run_id,
            parent_run_id=None
        )

        span_holder = callback_handler.spans[run_id]
        assert span_holder.span.name == "OpenAIEmbeddings.completion"
        assert span_holder.span.attributes[SpanAttributes.TRACELOOP_SPAN_KIND] == "embedding"

    def test_chat_model(self, callback_handler, span_exporter):
        """Test generation span kind detection in chat model."""
        run_id = uuid4()
        serialized = {
            "id": ["langchain", "chat_models", "openai", "ChatOpenAI"],
            "name": "ChatOpenAI"
        }

        from langchain_core.messages import HumanMessage
        messages = [[HumanMessage(content="test message")]]

        callback_handler.on_chat_model_start(
            serialized=serialized,
            messages=messages,
            run_id=run_id,
            parent_run_id=None
        )

        span_holder = callback_handler.spans[run_id]
        assert span_holder.span.name == "ChatOpenAI.chat"
        assert span_holder.span.attributes[SpanAttributes.TRACELOOP_SPAN_KIND] == "generation"

    def test_runnable_lambda_embedding_creates_embedding_span(self, callback_handler, span_exporter):
        """Test RunnableLambda with embedding name creates EMBEDDING span."""
        parent_run_id = uuid4()
        run_id = uuid4()

        callback_handler.on_chain_start(
            serialized={"id": ["pipeline"], "name": "EmbeddingPipeline"},
            inputs={},
            run_id=parent_run_id,
            parent_run_id=None
        )

        # RunnableLambda with embedding name
        serialized = {
            "id": ["langchain_core", "runnables", "base", "RunnableLambda"],
            "name": "RunnableLambda"
        }

        callback_handler.on_chain_start(
            serialized=serialized,
            inputs={"texts": ["doc1", "doc2"]},
            run_id=run_id,
            parent_run_id=parent_run_id,
            name="OpenAIEmbeddings"  # Name passed via kwargs
        )

        span_holder = callback_handler.spans[run_id]
        assert span_holder.span.name == "OpenAIEmbeddings.embedding"
        assert span_holder.span.attributes[SpanAttributes.TRACELOOP_SPAN_KIND] == "embedding"

    def test_vector_store_retriever(self, callback_handler, span_exporter):
        """Test VectorStoreRetriever creates proper RETRIEVER spans (real LangChain component)."""
        run_id = uuid4()

        serialized = {
            "id": ["langchain_core", "vectorstores", "base", "VectorStoreRetriever"],
            "name": "VectorStoreRetriever"
        }

        callback_handler.on_retriever_start(
            serialized=serialized,
            query="test query",
            run_id=run_id,
            parent_run_id=None
        )

        span_holder = callback_handler.spans[run_id]
        assert span_holder.span.name == "VectorStoreRetriever.retriever"
        assert span_holder.span.attributes[SpanAttributes.TRACELOOP_SPAN_KIND] == "retriever"

        from langchain_core.documents import Document
        callback_handler.on_retriever_end(
            documents=[Document(page_content="test doc")],
            run_id=run_id,
            parent_run_id=None
        )

        # Check span was ended
        assert run_id not in callback_handler.spans
