from enum import Enum


class Instruments(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    PINECONE = "pinecone"
    CHROMA = "chroma"
    LANGCHAIN = "langchain"
    LLAMA_INDEX = "llama_index"
    TRANSFORMERS = "transformers"
    REQUESTS = "requests"
    URLLIB3 = "urllib3"
    PYMYSQL = "pymysql"
    BEDROCK = "bedrock"
    REPLICATE = "replicate"
    VERTEXAI = "vertexai"
    WATSONX = "watsonx"
    WEAVIATE = "weaviate"
