import logging
import os
from typing import Set

logger = logging.getLogger(__name__)

# Track which warnings have already been shown (avoid spam)
_warned_instrumentors: Set[str] = set()

# Mapping from instrument name to pip install extra name
INSTRUMENT_TO_EXTRA = {
    "openai": "openai",
    "anthropic": "anthropic",
    "azure_search": "azure-search",
    "mistral": "mistralai",
    "cohere": "cohere",
    "google_generativeai": "google-generativeai",
    "bedrock": "bedrock",
    "sagemaker": "sagemaker",
    "vertexai": "vertexai",
    "watsonx": "watsonx",
    "ollama": "ollama",
    "together": "together",
    "groq": "groq",
    "replicate": "replicate",
    "writer": "writer",
    "alephalpha": "alephalpha",
    "langchain": "langchain",
    "llama_index": "llamaindex",
    "crewai": "crewai",
    "haystack": "haystack",
    "agno": "agno",
    "openai_agents": "openai-agents",
    "mcp": "mcp",
    "transformers": "transformers",
    "pinecone": "pinecone",
    "qdrant": "qdrant",
    "lancedb": "lancedb",
    "chroma": "chromadb",
    "milvus": "milvus",
    "marqo": "marqo",
    "weaviate": "weaviate",
}


def warn_missing_instrumentation(
    instrument_name: str,
    target_library_installed: bool,
) -> None:
    """
    Log a helpful warning when an instrumentation package is not installed
    but the user appears to want to use it.

    Args:
        instrument_name: The name of the instrument (from Instruments enum value)
        target_library_installed: Whether the target library (e.g., 'openai') is installed
    """
    if os.getenv("TRACELOOP_SUPPRESS_WARNINGS", "false").lower() == "true":
        return

    if instrument_name in _warned_instrumentors:
        return

    _warned_instrumentors.add(instrument_name)

    extra_name = INSTRUMENT_TO_EXTRA.get(instrument_name.lower(), instrument_name.lower())

    if target_library_installed:
        logger.info(
            f"Traceloop: '{instrument_name}' library detected but instrumentation "
            f"not installed. To enable tracing, run: "
            f"pip install 'traceloop-sdk[{extra_name}]'"
        )


def reset_warnings() -> None:
    """Reset the warned instrumentors set. Useful for testing."""
    _warned_instrumentors.clear()
