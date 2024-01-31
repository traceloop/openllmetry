from enum import Enum


class SpanAttributes:
    # LLM
    LLM_VENDOR = "llm.vendor"
    LLM_REQUEST_TYPE = "llm.request.type"
    LLM_REQUEST_MODEL = "llm.request.model"
    LLM_RESPONSE_MODEL = "llm.response.model"
    LLM_REQUEST_MAX_TOKENS = "llm.request.max_tokens"
    LLM_USAGE_TOTAL_TOKENS = "llm.usage.total_tokens"
    LLM_USAGE_COMPLETION_TOKENS = "llm.usage.completion_tokens"
    LLM_USAGE_PROMPT_TOKENS = "llm.usage.prompt_tokens"
    LLM_TEMPERATURE = "llm.temperature"
    LLM_USER = "llm.user"
    LLM_HEADERS = "llm.headers"
    LLM_TOP_P = "llm.top_p"
    LLM_TOP_K = "llm.top_k"
    LLM_FREQUENCY_PENALTY = "llm.frequency_penalty"
    LLM_PRESENCE_PENALTY = "llm.presence_penalty"
    LLM_PROMPTS = "llm.prompts"
    LLM_COMPLETIONS = "llm.completions"
    LLM_CHAT_STOP_SEQUENCES = "llm.chat.stop_sequences"
    LLM_REQUEST_FUNCTIONS = "llm.request.functions"

    # Vector DB
    VECTOR_DB_VENDOR = "vector_db.vendor"
    VECTOR_DB_QUERY_TOP_K = "vector_db.query.top_k"

    # LLM Workflows
    TRACELOOP_SPAN_KIND = "traceloop.span.kind"
    TRACELOOP_WORKFLOW_NAME = "traceloop.workflow.name"
    TRACELOOP_ENTITY_NAME = "traceloop.entity.name"
    TRACELOOP_ASSOCIATION_PROPERTIES = "traceloop.association.properties"

    # Deprecated
    TRACELOOP_CORRELATION_ID = "traceloop.correlation.id"

    # Watson/genai LLM
    LLM_DECODING_METHOD = "llm.watsonx.decoding_method"
    LLM_RANDOM_SEED = "llm.watsonx.random_seed"
    LLM_MAX_NEW_TOKENS = "llm.watsonx.max_new_tokens"
    LLM_MIN_NEW_TOKENS = "llm.watsonx.min_new_tokens"
    LLM_REPETITION_PENALTY = "llm.watsonx.repetition_penalty"


class Events(Enum):
    DB_QUERY_EMBEDDINGS = "db.query.embeddings"
    DB_PINECONE_QUERY_RESULT = "db.pinecone.query.result"
    DB_CRHOMADB_QUERY_RESULT = "db.chromadb.query.result"


class EventAttributes(Enum):
    # Query Embeddings
    DB_QUERY_EMBEDDINGS_VECTOR = "db.query.embeddings.vector"

    # Query Result (from ChromaDB)
    DB_CHROMADB_QUERY_RESULT_IDS = "db.chromadb.query.result.ids"
    DB_CHROMADB_QUERY_RESULT_DISTANCES = "db.chromadb.query.result.distances"
    DB_CHROMADB_QUERY_RESULT_METADATA = "db.chromadb.query.result.metadata"
    DB_CHROMADB_QUERY_RESULT_DOCUMENTS = "db.chromadb.query.result.documents"

    # Query Result (from Pinecone)
    DB_PINECONE_QUERY_RESULT_ID = "db.pinecone.query.result.id"
    DB_PINECONE_QUERY_RESULT_SCORE = "db.pinecone.query.result.score"
    DB_PINECONE_QUERY_RESULT_VECTOR = "db.pinecone.query.result.vector"


class LLMRequestTypeValues(Enum):
    COMPLETION = "completion"
    CHAT = "chat"
    RERANK = "rerank"
    EMBEDDING = "embedding"
    UNKNOWN = "unknown"


class TraceloopSpanKindValues(Enum):
    WORKFLOW = "workflow"
    TASK = "task"
    AGENT = "agent"
    TOOL = "tool"
    UNKNOWN = "unknown"
